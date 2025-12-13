import cv2
import numpy as np
import json
import os
import sys
import re

# ==========================================
# 1. ユーザー設定セクション
# ==========================================
# 対象とする動画ファイル名 (data/raw_videos 内にあるものと仮定)
# ※ 01_FrameSampling で使用したものと同じ名前にしてください
TARGET_VIDEO_NAME = "oimachi_251113_2.mp4"

# プレビュー表示時のスケール（1メートルあたり何ピクセルで表示するか）
# これが小さいとプレビュー画像が小さくなりすぎます
PREVIEW_SCALE_PX_PER_METER = 100

# ==========================================
# 2. パス設定・初期化
# ==========================================
def setup_paths():
    """
    パス関係の設定を行う関数
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # プロジェクト名 (拡張子なし)
    project_name = os.path.splitext(TARGET_VIDEO_NAME)[0]
    
    # フレーム画像のディレクトリ
    frames_dir = os.path.join(project_root, "data", "projects", project_name, "frames")
    
    # frames ディレクトリ内から最も若い番号のファイルを選択
    input_image_path = get_first_frame(frames_dir)
    
    # 設定ファイルの保存ディレクトリ
    configs_dir = os.path.join(project_root, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    
    # 出力JSONパス
    output_json_path = os.path.join(configs_dir, f"{project_name}.json")
    
    # キャリブレーション画像の保存ディレクトリ
    calibration_dir = os.path.join(project_root, "data", "projects", project_name, "calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    
    return input_image_path, output_json_path, calibration_dir


def get_first_frame(frames_dir):
    """
    frames ディレクトリ内から最も若い番号のフレームファイルを取得する関数
    """
    if not os.path.exists(frames_dir):
        print(f"エラー: フレームディレクトリが見つかりません: {frames_dir}")
        print("ヒント: 先に 01_FrameSampling を実行してください。")
        sys.exit(1)
    
    # frame_XXXXXX.jpg 形式のファイルを検索
    pattern = re.compile(r'^frame_(\d+)\.jpg$')
    frame_files = []
    
    for filename in os.listdir(frames_dir):
        match = pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            frame_files.append((frame_num, filename))
    
    if not frame_files:
        print(f"エラー: フレームファイルが見つかりません: {frames_dir}")
        print("ヒント: 先に 01_FrameSampling を実行してください。")
        sys.exit(1)
    
    # 番号順にソートして最も若いものを取得
    frame_files.sort(key=lambda x: x[0])
    first_frame = frame_files[0][1]
    
    return os.path.join(frames_dir, first_frame)

# ==========================================
# 3. GUI関連処理
# ==========================================
clicked_points = []
temp_img = None

def mouse_callback(event, x, y, flags, param):
    """
    マウスクリックイベントを処理するコールバック関数
    """
    global clicked_points, temp_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"ポイント追加: ({x}, {y}) - 残り {4 - len(clicked_points)} 点")

def run_point_selection(image_path):
    """
    画像を読み込み、GUIで4点を指定させる処理
    """
    global clicked_points, temp_img
    
    # 画像読み込み
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        print("ヒント: 先に 01_FrameSampling を実行してください。")
        sys.exit(1)
    
    # 日本語パス対応: np.fromfileを使用
    original_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original_img is None:
        print("エラー: 画像の読み込みに失敗しました。")
        sys.exit(1)

    print("-" * 60)
    print("【操作方法】")
    print("  マウス左クリック : 点を追加")
    print("  順番           : ①左上 -> ②右上 -> ③右下 -> ④左下")
    print("  r キー         : リセット (点を消去)")
    print("  Enter キー     : 確定 (4点選択後)")
    print("  q キー         : 中断して終了")
    print("-" * 60)

    window_name = "Calibration: Select 4 Points (TL->TR->BR->BL)"
    # WINDOW_NORMAL: ウィンドウをリサイズ可能にする
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # 画像サイズを取得し、画面に収まるようにウィンドウサイズを設定
    img_h, img_w = original_img.shape[:2]
    max_window_width = 1280
    max_window_height = 720
    
    # アスペクト比を維持してリサイズ
    scale = min(max_window_width / img_w, max_window_height / img_h, 1.0)
    window_w = int(img_w * scale)
    window_h = int(img_h * scale)
    cv2.resizeWindow(window_name, window_w, window_h)

    while True:
        # 描画用画像のコピー
        display_img = original_img.copy()
        
        # クリックされた点を描画
        for i, pt in enumerate(clicked_points):
            # 点を描画 (赤色)
            cv2.circle(display_img, tuple(pt), 5, (0, 0, 255), -1)
            # 順番を表示
            cv2.putText(display_img, str(i+1), (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 線で結ぶ
        if len(clicked_points) > 1:
            for i in range(len(clicked_points) - 1):
                cv2.line(display_img, tuple(clicked_points[i]), tuple(clicked_points[i+1]), (0, 255, 0), 2)
        # 4点揃ったら最後と最初も結ぶ
        if len(clicked_points) == 4:
            cv2.line(display_img, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 0), 2)

        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(20) & 0xFF
        
        # Enterキー (13) で確定
        if key == 13:
            if len(clicked_points) == 4:
                print("4点が選択されました。座標を確定します。")
                break
            else:
                print(f"まだ {len(clicked_points)} 点しか選択されていません。4点選択してください。")
        
        # 'r' キーでリセット
        elif key == ord('r'):
            clicked_points = []
            print("リセットしました。")
            
        # 'q' キーで終了
        elif key == ord('q'):
            print("処理を中断します。")
            sys.exit(0)

    cv2.destroyAllWindows()
    return original_img, clicked_points

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    input_image_path, output_json_path, calibration_dir = setup_paths()
    
    print(f"キャリブレーションを開始します: {TARGET_VIDEO_NAME}")
    
    # 1. 点の指定
    img, src_pts_list = run_point_selection(input_image_path)
    
    # 2. 実距離の入力
    print("\n" + "="*40)
    print("【実空間距離の入力】")
    print("選択した長方形エリアの現実世界でのサイズを入力してください (メートル単位)")
    print("例: 3.5")
    try:
        real_width = float(input("横幅 (メートル) > "))
        real_height = float(input("縦幅 (メートル) > "))
    except ValueError:
        print("エラー: 数値を入力してください。")
        sys.exit(1)

    # 3. 行列計算
    # 変換元座標 (画像上のピクセル)
    src_pts = np.float32(src_pts_list)
    
    # 変換先座標 (実空間座標: メートル)
    # 左上(0,0), 右上(w,0), 右下(w,h), 左下(0,h)
    dst_pts_meter = np.float32([
        [0, 0],
        [real_width, 0],
        [real_width, real_height],
        [0, real_height]
    ])
    
    # ホモグラフィ行列の計算 (ピクセル -> メートル)
    # cv2.getPerspectiveTransform(src, dst)
    H_meter = cv2.getPerspectiveTransform(src_pts, dst_pts_meter)

    print("\n変換行列を計算しました。")

    # 4. 変換結果の確認 (プレビュー)
    # プレビュー用に、メートルをピクセルにスケールアップした行列を作る
    # 例: 5メートル -> 500ピクセル
    dst_pts_preview = dst_pts_meter * PREVIEW_SCALE_PX_PER_METER
    H_preview = cv2.getPerspectiveTransform(src_pts, dst_pts_preview)
    
    # 出力画像のサイズ計算
    preview_w = int(real_width * PREVIEW_SCALE_PX_PER_METER)
    preview_h = int(real_height * PREVIEW_SCALE_PX_PER_METER)
    
    # 画像変換 (鳥瞰図作成)
    warped_img = cv2.warpPerspective(img, H_preview, (preview_w, preview_h))
    
    print("変換結果のプレビューを表示します。確認したら何かキーを押して閉じてください。")
    cv2.imshow("Preview: Bird's Eye View", warped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5. キャリブレーション画像の保存
    # 4点を描画した元画像を作成
    points_img = img.copy()
    for i, pt in enumerate(src_pts_list):
        cv2.circle(points_img, tuple(pt), 8, (0, 0, 255), -1)
        cv2.putText(points_img, str(i+1), (pt[0]+15, pt[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    # 線で結ぶ
    for i in range(len(src_pts_list)):
        cv2.line(points_img, tuple(src_pts_list[i]), tuple(src_pts_list[(i+1) % 4]), (0, 255, 0), 3)
    
    # 画像を保存（日本語パス対応）
    points_img_path = os.path.join(calibration_dir, "calibration_points.jpg")
    warped_img_path = os.path.join(calibration_dir, "calibration_warped.jpg")
    
    # 4点描画画像を保存
    ret, buf = cv2.imencode('.jpg', points_img)
    if ret:
        buf.tofile(points_img_path)
        print(f"4点描画画像を保存しました: {points_img_path}")
    
    # 鳥瞰図画像を保存
    ret, buf = cv2.imencode('.jpg', warped_img)
    if ret:
        buf.tofile(warped_img_path)
        print(f"鳥瞰図画像を保存しました: {warped_img_path}")

    # 6. JSON保存
    # numpyのデータ型はjsonで保存できないためlistに変換
    config_data = {
        "target_video": TARGET_VIDEO_NAME,
        "real_width_m": real_width,
        "real_height_m": real_height,
        "src_points": src_pts_list,  # [[x,y], ...]
        "homography_matrix": H_meter.tolist() # ピクセル -> メートルの変換行列
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
        
    print("\n" + "="*40)
    print("キャリブレーション設定を保存しました。")
    print(f"設定ファイル: {output_json_path}")
    print(f"画像フォルダ: {calibration_dir}")
    print("="*40)

if __name__ == "__main__":
    main()