import cv2
import os
import sys
import numpy as np

# ==========================================
# 1. ユーザー設定セクション
# ==========================================
# 入力する動画ファイル名 (data/raw_videos 内にあるファイルを指定)
TARGET_VIDEO_NAME = "oimachi_251113_2.mp4" 

# 何フレームごとに保存するか (例: 5 = 5フレームに1回保存, 30 = 1秒に1回程度)
FRAME_STEP = 10

# 画像のリサイズ倍率 (1.0 = そのまま, 0.5 = 半分)
RESIZE_SCALE = 1.0

# ==========================================
# 2. パス設定・フォルダ自動生成
# ==========================================
def setup_paths():
    """
    現在のファイルの場所を基準に、入力・出力パスを設定する関数
    """
    # このスクリプトがあるディレクトリ (src/01_FrameSampling)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # プロジェクトのルートディレクトリ (VideoAnalytics)
    # src/01_FrameSampling から2階層上に戻る
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # 入力動画のディレクトリ
    input_dir = os.path.join(project_root, "data", "raw_videos")
    
    # 入力動画のフルパス
    video_path = os.path.join(input_dir, TARGET_VIDEO_NAME)
    
    # 出力先のベースディレクトリ (data/projects - 既存フォルダを使用)
    output_base_dir = os.path.join(project_root, "data", "projects")
    
    # プロジェクト名 (拡張子なしの動画ファイル名)
    project_name = os.path.splitext(TARGET_VIDEO_NAME)[0]
    
    # 最終的な画像の保存先 (data/projects/[project_name]/frames)
    output_dir = os.path.join(output_base_dir, project_name, "frames")
    
    return video_path, output_dir

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    # パスの取得とフォルダ作成
    video_path, output_dir = setup_paths()
    
    print("-" * 50)
    print(f"処理を開始します: FrameSampling_ver1")
    print(f"対象動画: {video_path}")
    print(f"間引き設定: {FRAME_STEP} フレームごと")
    print("-" * 50)

    # 動画ファイルの存在確認
    if not os.path.exists(video_path):
        print(f"エラー: 動画ファイルが見つかりません。\nパスを確認してください: {video_path}")
        sys.exit(1)

    # 出力ディレクトリの作成 (存在していてもエラーにしない)
    os.makedirs(output_dir, exist_ok=True)
    print(f"出力ディレクトリを確認しました: {output_dir}")

    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("エラー: 動画を開けませんでした。ファイル形式やコーデックを確認してください。")
        sys.exit(1)

    # 総フレーム数の取得（目安）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"総フレーム数: {total_frames}")

    frame_count = 0  # 現在のフレーム番号
    saved_count = 0  # 保存した枚数

    while True:
        ret, frame = cap.read()
        
        # 動画の最後または読み込みエラーで終了
        if not ret:
            break

        # 指定された間隔（FRAME_STEP）のときだけ保存
        if frame_count % FRAME_STEP == 0:
            
            # リサイズ処理（必要な場合のみ）
            if RESIZE_SCALE != 1.0:
                height, width = frame.shape[:2]
                new_dim = (int(width * RESIZE_SCALE), int(height * RESIZE_SCALE))
                frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_LINEAR)

            # ファイル名の生成 (frame_000005.jpg のように6桁ゼロ埋め)
            filename = f"frame_{frame_count:06d}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            # 保存（日本語パス対応: cv2.imencodeを使用）
            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                buf.tofile(save_path)
                saved_count += 1
            else:
                print(f"警告: フレーム {frame_count} のエンコードに失敗しました")
            
            # ログ出力 (10枚保存するごとに進捗表示)
            if saved_count > 0 and saved_count % 10 == 0:
                print(f"進捗: {saved_count} 枚保存済み (現在フレーム: {frame_count}/{total_frames})")

        frame_count += 1

    # 後処理
    cap.release()
    
    # 実際に保存されたファイル数を確認
    if os.path.exists(output_dir):
        actual_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        actual_count = len(actual_files)
    else:
        actual_count = 0
        print(f"エラー: 出力ディレクトリが存在しません: {output_dir}")
    
    print("-" * 50)
    print("処理が完了しました。")
    print(f"保存試行回数: {saved_count} 枚")
    print(f"実際に保存されたファイル数: {actual_count} 枚")
    print(f"画像の保存先フォルダ:\n{output_dir}")
    if actual_count > 0:
        print(f"保存されたファイル例: {actual_files[0] if actual_files else 'なし'}")
    print("-" * 50)

if __name__ == "__main__":
    main()