import pandas as pd
import numpy as np
import cv2
import json
import os
import sys
import matplotlib.pyplot as plt

# ==========================================
# 1. ユーザー設定セクション
# ==========================================
# 対象とする動画ファイル名 (ここからプロジェクト名を特定します)
TARGET_VIDEO_NAME = "asagaya_251213_05.mp4"

# 動画のフレームレート (速度計算の時間単位として使用)
# ※画像ベース分析で間引かれている場合でも、元の動画のFPSを指定してください
FPS = 30

# スムージングのウィンドウサイズ (移動平均の幅)
# ノイズが多い場合は大きくしてください (例: 5 ~ 10)
SMOOTHING_WINDOW = 5

# ==========================================
# 2. パス設定・初期化
# ==========================================
def setup_paths():
    """パス関係の設定を行う関数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # プロジェクト名 (拡張子なし)
    project_name = os.path.splitext(TARGET_VIDEO_NAME)[0]
    
    # 入力1: 設定ファイル (ホモグラフィ行列)
    config_path = os.path.join(project_root, "configs", f"{project_name}.json")
    
    # 入力2: トラッキングデータ
    tracking_csv_path = os.path.join(project_root, "data", "projects", project_name, "tracking", "tracking.csv")
    
    # 入力3: キャリブレーション後の鳥瞰図画像
    calibration_img_path = os.path.join(project_root, "data", "projects", project_name, "calibration", "calibration_warped.jpg")
    
    # 出力ディレクトリ
    output_dir = os.path.join(project_root, "data", "projects", project_name, "trajectory")
    os.makedirs(output_dir, exist_ok=True)
    
    return config_path, tracking_csv_path, calibration_img_path, output_dir

# ==========================================
# 3. データ処理関数
# ==========================================
def load_config(json_path):
    """JSONファイルからホモグラフィ行列と実空間サイズを読み込む"""
    if not os.path.exists(json_path):
        print(f"エラー: 設定ファイルが見つかりません: {json_path}")
        print("02_Calibration を実行してください。")
        sys.exit(1)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if "homography_matrix" not in data:
        print("エラー: JSONに 'homography_matrix' が含まれていません。")
        sys.exit(1)
    
    H_matrix = np.array(data["homography_matrix"], dtype=np.float32)
    real_width = data.get("real_width_m", 10.0)
    real_height = data.get("real_height_m", 10.0)
        
    return H_matrix, real_width, real_height

def pixel_to_meter(df, H_matrix):
    """
    ピクセル座標 (足元) をメートル座標に変換する
    """
    # 足元の中心座標 (u, v) を計算
    # u = (x1 + x2) / 2, v = y2 (足元)
    u = (df['x1'] + df['x2']) / 2
    v = df['y2']
    
    # cv2.perspectiveTransform 用に形状を変換 (N, 1, 2)
    points_pixel = np.column_stack((u, v)).astype(np.float32)
    points_pixel = points_pixel.reshape(-1, 1, 2)
    
    # 変換実行
    points_meter = cv2.perspectiveTransform(points_pixel, H_matrix)
    
    # 形状を戻す (N, 2)
    points_meter = points_meter.reshape(-1, 2)
    
    # DataFrameに追加
    df['x_meter'] = points_meter[:, 0]
    df['y_meter'] = points_meter[:, 1]
    
    return df

def calculate_velocity_and_smooth(df_group):
    """
    IDごとのグループに対してスムージングと速度計算を行う
    """
    # 1. 座標のスムージング (移動平均)
    # min_periods=1 に設定することで、データ数がWindow未満でも計算する
    df_group['x_meter_smooth'] = df_group['x_meter'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    df_group['y_meter_smooth'] = df_group['y_meter'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    
    # 2. 変位の計算 (現在のフレーム - 1つ前のデータ)
    dx = df_group['x_meter_smooth'].diff()
    dy = df_group['y_meter_smooth'].diff()
    
    # 3. 時間差の計算 (フレーム番号の差分 / FPS)
    # フレームが間引かれている場合(5, 10...)にも対応するため、indexの差分を使う
    d_frame = df_group['frame'].diff()
    dt = d_frame / FPS
    
    # dtが0またはNaNの場合は速度計算できないため除外（あるいは0にする）
    # ここでは計算結果を格納
    vx = dx / dt
    vy = dy / dt
    
    # 4. 速度 (スカラー)
    speed = np.sqrt(vx**2 + vy**2)
    
    # 結果をDataFrameに戻す
    df_group['x_meter'] = df_group['x_meter_smooth'] # スムージング後の値を採用
    df_group['y_meter'] = df_group['y_meter_smooth']
    df_group['vx'] = vx
    df_group['vy'] = vy
    df_group['speed'] = speed
    
    # diff計算で最初の行はNaNになるため、0で埋めるかNaNのままにする
    # ここでは後続処理のためにNaNを0埋めする
    df_group = df_group.fillna(0)
    
    return df_group

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    config_path, tracking_csv_path, calibration_img_path, output_dir = setup_paths()
    
    print("-" * 50)
    print("軌跡変換処理 (Trajectory Processing) を開始します")
    print(f"対象動画: {TARGET_VIDEO_NAME}")
    print(f"FPS: {FPS}, Smoothing Window: {SMOOTHING_WINDOW}")
    print("-" * 50)

    # 1. データ読み込み
    print("データを読み込んでいます...")
    H_matrix, real_width, real_height = load_config(config_path)
    
    if not os.path.exists(tracking_csv_path):
        print(f"エラー: トラッキングデータが見つかりません: {tracking_csv_path}")
        print("03_Tracking を実行してください。")
        sys.exit(1)
        
    df = pd.read_csv(tracking_csv_path)
    if len(df) == 0:
        print("エラー: トラッキングデータが空です。")
        sys.exit(1)

    print(f"トラッキングデータ数: {len(df)} 行")

    # 2. 座標変換 (Pixel -> Meter)
    print("実空間座標へ変換中...")
    df = pixel_to_meter(df, H_matrix)

    # 3. 速度・方向計算とスムージング
    print("速度計算とスムージングを実行中...")
    # track_id ごとにグループ化して適用
    # group_keys=False は元のインデックス構造を維持するため
    df = df.groupby('track_id', group_keys=False).apply(calculate_velocity_and_smooth)
    
    # 不要なカラムを削除・整理 (x1, y1等はもう不要だが、検証用に残しても良い)
    # 要件にある出力カラムのみに絞る場合:
    output_columns = ['frame', 'track_id', 'x_meter', 'y_meter', 'vx', 'vy', 'speed']
    df_out = df[output_columns].sort_values(['track_id', 'frame'])

    # 4. CSV出力
    csv_output_path = os.path.join(output_dir, "trajectory.csv")
    df_out.to_csv(csv_output_path, index=False)
    print(f"CSV保存完了: {csv_output_path}")

    # 5. 確認用画像出力 (背景に鳥瞰図を表示)
    print("確認用プロットを作成中...")
    plot_output_path = os.path.join(output_dir, "trajectory.png")
    
    # 図のサイズを大きくする
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 背景にキャリブレーション後の鳥瞰図画像を表示
    # 座標系：4点目（左下）を原点、左上方向がY軸正、右下方向がX軸正
    if os.path.exists(calibration_img_path):
        # 日本語パス対応で画像を読み込む
        bg_img = cv2.imdecode(np.fromfile(calibration_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bg_img is not None:
            # BGRからRGBに変換
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            # 画像をメートル座標系に合わせて表示
            # extent = [left, right, bottom, top]
            # 鳥瞰図の左下（4点目）が(0,0)、左上（1点目）が(0,real_height)に対応
            ax.imshow(bg_img, extent=[0, real_width, 0, real_height], alpha=0.6)
            print(f"背景画像を読み込みました: {calibration_img_path}")
        else:
            print(f"警告: 背景画像の読み込みに失敗しました")
    else:
        print(f"警告: 背景画像が見つかりません: {calibration_img_path}")
    
    # IDごとに色を変えてプロット
    unique_ids = df_out['track_id'].unique()
    print(f"検出されたユニークID数: {len(unique_ids)}")
    
    # カラーマップを使用して色を割り当て
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    
    for i, tid in enumerate(unique_ids):
        track_data = df_out[df_out['track_id'] == tid]
        # Y座標を反転（左下を原点にするため）
        y_plot = real_height - track_data['y_meter']
        ax.plot(track_data['x_meter'], y_plot, 
                marker='o', markersize=8, linestyle='-', linewidth=3, 
                alpha=0.9, color=colors[i % len(colors)], label=f"ID {tid}")

    ax.set_title(f"Trajectory Plot (Meter Scale) - {TARGET_VIDEO_NAME}", fontsize=16)
    ax.set_xlabel("X Position (m)", fontsize=14)
    ax.set_ylabel("Y Position (m)", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 表示範囲：キャリブレーション範囲とデータ範囲の両方を含める
    # Y座標を反転（左下を原点にするため: y_plot = real_height - y_meter）
    y_plot_min = real_height - df_out['y_meter'].max()
    y_plot_max = real_height - df_out['y_meter'].min()
    x_min_data, x_max_data = df_out['x_meter'].min(), df_out['x_meter'].max()
    
    margin = 0.5  # 余白（メートル）
    x_min = min(0, x_min_data) - margin
    x_max = max(real_width, x_max_data) + margin
    y_min = min(0, y_plot_min) - margin
    y_max = max(real_height, y_plot_max) + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)  # 数学座標系（左下が原点、Y軸は上向き正）
    ax.set_aspect('equal')
    
    # 凡例はIDが多すぎると邪魔になるので、数が少なければ表示
    if len(unique_ids) < 20:
        ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150)
    plt.close() # メモリ解放
    
    print(f"プロット画像保存完了: {plot_output_path}")
    print("-" * 50)
    print("処理が正常に完了しました。")

if __name__ == "__main__":
    main()