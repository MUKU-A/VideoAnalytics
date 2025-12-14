import pandas as pd
import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt
import cv2
import json

# ==========================================
# 1. ユーザー設定（研究で変更しうるパラメータ）
# ==========================================
TARGET_VIDEO_NAME = "asagaya_251213_05.mp4"

FPS = 30

# 空間分割
MESH_SIZE_M = 0.5      # メッシュサイズ [m]

# 時間分割
TIME_WINDOW_SEC = 3  # 秒
TIME_WINDOW_FRAME = TIME_WINDOW_SEC * FPS

# 静止判定閾値
STOP_SPEED_THRESHOLD = 0.3  # [m/s]

# ==========================================
# 2. パス設定
# ==========================================
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    project_name = os.path.splitext(TARGET_VIDEO_NAME)[0]

    trajectory_csv = os.path.join(
        project_root,
        "data", "projects", project_name, "trajectory", "trajectory.csv"
    )

    config_path = os.path.join(
        project_root, "configs", f"{project_name}.json"
    )

    calib_img_path = os.path.join(
        project_root,
        "data", "projects", project_name, "calibration", "calibration_warped.jpg"
    )

    output_dir = os.path.join(
        project_root,
        "data", "projects", project_name, "features"
    )
    os.makedirs(output_dir, exist_ok=True)

    return trajectory_csv, config_path, calib_img_path, output_dir

# ==========================================
# 3. 設定ファイル読込
# ==========================================
def load_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    real_w = data["real_width_m"]
    real_h = data["real_height_m"]
    
    # 鳥瞰図画像のオフセット情報（存在する場合）
    warped_offset = data.get("warped_image_offset", None)
    
    return real_w, real_h, warped_offset

# ==========================================
# 4. 特徴量計算関数
# ==========================================
def direction_coherence(vx, vy):
    """
    方向収束度（論文の主要特徴量）

    R = ||Σv⃗|| / Σ||v⃗||

    ・人の移動方向が揃っているほど 1 に近づく
    ・滞留・混雑・交錯が多いほど 0 に近づく
    """
    sum_vx = vx.sum()
    sum_vy = vy.sum()
    norm_sum = np.sqrt(sum_vx**2 + sum_vy**2)
    sum_norm = np.sqrt(vx**2 + vy**2).sum()

    if sum_norm == 0:
        return 0.0
    return norm_sum / sum_norm

# ==========================================
# 5. メイン処理
# ==========================================
def main():
    traj_csv, config_path, calib_img_path, output_dir = setup_paths()

    if not os.path.exists(traj_csv):
        print("trajectory_.csv が見つかりません")
        sys.exit(1)

    real_w, real_h, warped_offset = load_config(config_path)

    # --------------------------------------
    # データ読込
    # --------------------------------------
    df = pd.read_csv(traj_csv)

    # 時間窓ID
    df["time_window_id"] = df["frame"] // TIME_WINDOW_FRAME

    # メッシュID
    df["mesh_x"] = (df["x_meter"] // MESH_SIZE_M).astype(int)
    df["mesh_y"] = (df["y_meter"] // MESH_SIZE_M).astype(int)

    feature_rows = []

    # --------------------------------------
    # 時間窓 × メッシュごとの特徴量
    # --------------------------------------
    grouped = df.groupby(["time_window_id", "mesh_x", "mesh_y"])

    for (tw, mx, my), g in grouped:
        if len(g) == 0:
            continue

        # ------------------------------
        # 1. 密度（人数）
        # ------------------------------
        density = g["new_track_id"].nunique()

        # ------------------------------
        # 2. 平均速度
        # ------------------------------
        mean_speed = g["speed"].mean()

        # ------------------------------
        # 3. 平均速度ベクトル
        # ------------------------------
        mean_vx = g["vx"].mean()
        mean_vy = g["vy"].mean()

        # ------------------------------
        # 4. 方向収束度
        # ------------------------------
        dir_coh = direction_coherence(g["vx"], g["vy"])

        # ------------------------------
        # 5. 静止率
        # ------------------------------
        stop_ratio = (g["speed"] < STOP_SPEED_THRESHOLD).mean()

        feature_rows.append([
            tw,
            mx,
            my,
            density,
            mean_speed,
            mean_vx,
            mean_vy,
            dir_coh,
            stop_ratio
        ])

    # --------------------------------------
    # 出力
    # --------------------------------------
    feature_df = pd.DataFrame(
        feature_rows,
        columns=[
            "time_window_id",
            "mesh_x",
            "mesh_y",
            "density",
            "mean_speed",
            "mean_vx",
            "mean_vy",
            "direction_coherence",
            "stop_ratio"
        ]
    )

    csv_out = os.path.join(output_dir, "features.csv")
    feature_df.to_csv(csv_out, index=False)

    # ======================================
    # 確認用可視化（density）
    # 座標系：4点目（左下）を原点(0,0)、左上方向がY軸正、右下方向がX軸正
    # 5枚出力：最初、1/4、1/2、3/4、最後
    # ======================================
    
    # 背景画像を読み込み
    bg = None
    # Calibrationの座標系: 左上(0,0), 右下(real_w, real_h)
    # 変換後の座標系: 左下(0,0), 左上(0, real_h), 右下(real_w, 0)
    # Y座標変換: y_new = real_h - y_old
    
    # 背景画像のオフセット（Calibration座標系）
    bg_x_min = 0
    bg_x_max = real_w
    bg_y_min_calib = 0
    bg_y_max_calib = real_h
    
    if os.path.exists(calib_img_path):
        bg = cv2.imdecode(np.fromfile(calib_img_path, np.uint8), cv2.IMREAD_COLOR)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        
        # オフセット情報がある場合は、選択範囲外も含む画像として配置
        if warped_offset is not None:
            bg_x_min = warped_offset["x_min_m"]
            bg_x_max = warped_offset["x_max_m"]
            bg_y_min_calib = warped_offset["y_min_m"]
            bg_y_max_calib = warped_offset["y_max_m"]
            print(f"背景画像の範囲(Calib座標): X=[{bg_x_min:.2f}, {bg_x_max:.2f}]m, Y=[{bg_y_min_calib:.2f}, {bg_y_max_calib:.2f}]m")

    # データ範囲を取得（Calibration座標系）
    mesh_x_min = int(feature_df["mesh_x"].min())
    mesh_x_max = int(feature_df["mesh_x"].max()) + 1
    mesh_y_min_calib = int(feature_df["mesh_y"].min())
    mesh_y_max_calib = int(feature_df["mesh_y"].max()) + 1

    n_mesh_x = mesh_x_max - mesh_x_min
    n_mesh_y = mesh_y_max_calib - mesh_y_min_calib

    # 座標変換: y_new = real_h - y_old
    # Calibration座標系のy_min -> 新座標系の real_h - y_min
    # Calibration座標系のy_max -> 新座標系の real_h - y_max
    
    # 表示範囲（新座標系：Y軸上向き）
    margin = MESH_SIZE_M
    x_min_plot = min(bg_x_min, mesh_x_min * MESH_SIZE_M) - margin
    x_max_plot = max(bg_x_max, mesh_x_max * MESH_SIZE_M) + margin
    # Y座標を変換
    y_min_plot = min(real_h - bg_y_max_calib, real_h - mesh_y_max_calib * MESH_SIZE_M) - margin
    y_max_plot = max(real_h - bg_y_min_calib, real_h - mesh_y_min_calib * MESH_SIZE_M) + margin

    # 出力するtime_windowを選択（最初、1/4、1/2、3/4、最後）
    unique_tw = sorted(feature_df["time_window_id"].unique())
    n_tw = len(unique_tw)
    
    if n_tw >= 5:
        indices = [0, n_tw // 4, n_tw // 2, 3 * n_tw // 4, n_tw - 1]
    else:
        indices = list(range(n_tw))  # 5未満なら全部
    
    selected_tw = [unique_tw[i] for i in indices]

    from matplotlib.patches import Rectangle

    for tw in selected_tw:
        fig, ax = plt.subplots(figsize=(14, 14))

        # 背景画像（新座標系に変換して配置）
        if bg is not None:
            # extent = [left, right, bottom, top]
            # Calibration座標系のY軸を反転して新座標系に変換
            # 画像の上端（Calib y_min）-> 新座標系の (real_h - y_min)
            # 画像の下端（Calib y_max）-> 新座標系の (real_h - y_max)
            bg_extent_new = [
                bg_x_min,
                bg_x_max,
                real_h - bg_y_max_calib,  # bottom
                real_h - bg_y_min_calib   # top
            ]
            ax.imshow(bg, extent=bg_extent_new, alpha=0.6)

        d = feature_df[feature_df["time_window_id"] == tw]

        # 2D配列を作成（0で初期化：データがないメッシュは密度0）
        density_grid = np.zeros((n_mesh_y, n_mesh_x))

        # データを2D配列に格納
        for _, row in d.iterrows():
            mx, my = int(row["mesh_x"]), int(row["mesh_y"])
            grid_x = mx - mesh_x_min
            grid_y = my - mesh_y_min_calib
            if 0 <= grid_x < n_mesh_x and 0 <= grid_y < n_mesh_y:
                density_grid[grid_y, grid_x] = row["density"]

        # メッシュの塗りつぶし表示（新座標系）
        # Calibration座標系のY座標を変換
        grid_extent_new = [
            mesh_x_min * MESH_SIZE_M,
            mesh_x_max * MESH_SIZE_M,
            real_h - mesh_y_max_calib * MESH_SIZE_M,  # bottom
            real_h - mesh_y_min_calib * MESH_SIZE_M   # top
        ]
        im = ax.imshow(
            density_grid,
            extent=grid_extent_new,
            cmap="Reds",
            alpha=0.7,
            interpolation="nearest",
            vmin=0
        )

        # キャリブレーション領域を枠で表示（緑の点線）
        # 新座標系: 左下(0,0), 左上(0, real_h), 右下(real_w, 0), 右上(real_w, real_h)
        calib_rect = Rectangle(
            (0, 0), real_w, real_h,
            linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
        )
        ax.add_patch(calib_rect)

        # 表示範囲を設定
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)

        # フレーム範囲を計算
        frame_start = int(tw * TIME_WINDOW_FRAME)
        frame_end = int((tw + 1) * TIME_WINDOW_FRAME - 1)

        plt.colorbar(im, ax=ax, label="Density")
        ax.set_title(f"Density Map (time window {tw}, frame {frame_start}-{frame_end})\nGreen dashed: Calibration area ({real_w}m x {real_h}m)", fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        # ファイル名にフレーム情報を追加
        img_out = os.path.join(output_dir, f"feature_tw{tw:03d}_frame{frame_start:06d}-{frame_end:06d}.png")
        plt.tight_layout()
        plt.savefig(img_out, dpi=150)
        plt.close()
        
        print(f"画像保存: {img_out}")

    print("05_FeatureExtraction 完了")
    print(f"CSV: {csv_out}")

# ==========================================
if __name__ == "__main__":
    main()
