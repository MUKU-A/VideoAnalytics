import pandas as pd
import numpy as np
import cv2
import json
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# ==========================================
# 1. ユーザー設定
# ==========================================
TARGET_VIDEO_NAME = "asagaya_251213_05.mp4"
FPS = 30
SMOOTHING_WINDOW = 5

# --- ID再結合パラメータ（研究向け安全設定） ---
MAX_FRAME_GAP = 3           # 最大フレーム欠損
MAX_DIST_M = 1.0            # 最大距離 (m)
MAX_SPEED_DIFF = 0.8        # 速度差 (m/s)
MAX_DIR_DIFF_DEG = 30       # 方向差 (度)

# ==========================================
# 2. パス設定
# ==========================================
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    project_name = os.path.splitext(TARGET_VIDEO_NAME)[0]

    config_path = os.path.join(project_root, "configs", f"{project_name}.json")
    tracking_csv_path = os.path.join(project_root, "data", "projects", project_name, "tracking", "tracking.csv")
    calibration_img_path = os.path.join(project_root, "data", "projects", project_name, "calibration", "calibration_warped.jpg")
    output_dir = os.path.join(project_root, "data", "projects", project_name, "trajectory")
    os.makedirs(output_dir, exist_ok=True)

    return config_path, tracking_csv_path, calibration_img_path, output_dir

# ==========================================
# 3. ユーティリティ
# ==========================================
def load_config(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data["homography_matrix"], dtype=np.float32), data["real_width_m"], data["real_height_m"]

def pixel_to_meter(df, H):
    u = (df['x1'] + df['x2']) / 2
    v = df['y2']
    pts = np.column_stack((u, v)).astype(np.float32).reshape(-1, 1, 2)
    pts_m = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df['x_meter'] = pts_m[:, 0]
    df['y_meter'] = pts_m[:, 1]
    return df

def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

# ==========================================
# 4. ID再結合ロジック
# ==========================================
def reconnect_ids(df):
    df = df.sort_values(['frame'])
    new_id = 0
    id_map = {}
    active_tracks = {}

    df['new_track_id'] = -1

    for idx, row in df.iterrows():
        f = row['frame']
        x, y = row['x_meter'], row['y_meter']

        best_match = None
        best_dist = float('inf')

        for tid, last in active_tracks.items():
            frame_gap = f - last['frame']
            if frame_gap > MAX_FRAME_GAP:
                continue

            dist = np.hypot(x - last['x'], y - last['y'])
            if dist > MAX_DIST_M:
                continue

            if last['speed'] > 0 and row['speed'] > 0:
                if abs(row['speed'] - last['speed']) > MAX_SPEED_DIFF:
                    continue

            if last['dir'] is not None and row['direction'] is not None:
                if angle_diff(row['direction'], last['dir']) > MAX_DIR_DIFF_DEG:
                    continue

            if dist < best_dist:
                best_dist = dist
                best_match = tid

        if best_match is None:
            df.at[idx, 'new_track_id'] = new_id
            active_tracks[new_id] = {
                'frame': f,
                'x': x,
                'y': y,
                'speed': row['speed'],
                'dir': row['direction']
            }
            new_id += 1
        else:
            df.at[idx, 'new_track_id'] = best_match
            active_tracks[best_match] = {
                'frame': f,
                'x': x,
                'y': y,
                'speed': row['speed'],
                'dir': row['direction']
            }

    return df

# ==========================================
# 5. 速度・方向計算
# ==========================================
def compute_motion(df):
    df = df.sort_values('frame')
    dx = df['x_meter'].diff()
    dy = df['y_meter'].diff()
    dt = df['frame'].diff() / FPS

    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(vx**2 + vy**2)
    direction = (np.degrees(np.arctan2(vy, vx)) + 360) % 360

    df['vx'] = vx.fillna(0)
    df['vy'] = vy.fillna(0)
    df['speed'] = speed.fillna(0)
    df['direction'] = direction  # NaNはそのまま維持

    df['x_meter'] = df['x_meter'].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
    df['y_meter'] = df['y_meter'].rolling(SMOOTHING_WINDOW, min_periods=1).mean()

    return df

# ==========================================
# 6. メイン処理
# ==========================================
def main():
    config_path, tracking_csv_path, calib_img_path, output_dir = setup_paths()
    H, real_w, real_h = load_config(config_path)

    df = pd.read_csv(tracking_csv_path)
    df = pixel_to_meter(df, H)

    # 初期速度計算（再結合用）
    df = df.groupby('track_id', group_keys=False).apply(compute_motion)

    # ID再結合
    df = reconnect_ids(df)

    # 再結合後に再計算
    df = df.groupby('new_track_id', group_keys=False).apply(compute_motion)

    out_cols = ['frame', 'new_track_id', 'x_meter', 'y_meter', 'vx', 'vy', 'speed', 'direction']
    df_out = df[out_cols].sort_values(['new_track_id', 'frame'])

    csv_path = os.path.join(output_dir, "trajectory_ver2.csv")
    df_out.to_csv(csv_path, index=False)

    # ======================================
    # 可視化
    # ======================================
    fig, ax = plt.subplots(figsize=(14, 14))

    if os.path.exists(calib_img_path):
        bg = cv2.imdecode(np.fromfile(calib_img_path, np.uint8), cv2.IMREAD_COLOR)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        ax.imshow(bg, extent=[0, real_w, 0, real_h], alpha=0.6)

    ids = df_out['new_track_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(ids)))

    for i, tid in enumerate(ids):
        d = df_out[df_out['new_track_id'] == tid]

        y_plot = real_h - d['y_meter']

        ax.scatter(
            d['x_meter'],
            y_plot,
            s=50,                      # 点サイズ（元ver1相当）
            alpha=0.9,
            color=colors[i % len(colors)],
            label=f"ID {tid}"
        )


    ax.set_title("Trajectory (ID Reconnected)", fontsize=16)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    img_path = os.path.join(output_dir, "trajectory_ver2.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    print("Trajectory ver2 完了")
    print(f"CSV: {csv_path}")
    print(f"Image: {img_path}")

if __name__ == "__main__":
    main()
