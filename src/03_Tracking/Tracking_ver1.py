"""
Tracking_frames_ReID.py
ReID付きトラッカー（BoT-SORT）による人物トラッキング
フレーム間が飛んでもIDを維持する設計
"""

import cv2
import os
import glob
import re
import csv
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

# ==============================
# 設定
# ==============================

TARGET_VIDEO_NAME = "asagaya_251213_60.mp4"
MODEL_NAME = "yolov8s.pt"

YOLO_CONF = 0.1
YOLO_IOU = 0.9

FRAME_SAMPLE_INTERVAL = 10   # フレーム間が飛ぶ前提

OUTPUT_FPS = 6  # 30 * 0.2 = 6 (0.2倍速再生)
PREVIEW_LIMIT = 500

# ==============================
# ユーティリティ
# ==============================

def get_frame_number(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1

def load_frames(dir_path):
    exts = ["*.jpg", "*.png", "*.jpeg"]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(dir_path, e))
    return sorted(files, key=get_frame_number)

# ==============================
# メイン
# ==============================

def main():

    project = os.path.splitext(TARGET_VIDEO_NAME)[0]
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    frames_dir = os.path.join(base, "data", "projects", project, "frames")
    out_dir = os.path.join(base, "data", "projects", project, "tracking")
    os.makedirs(out_dir, exist_ok=True)

    frames = load_frames(frames_dir)
    assert frames, "フレームが見つかりません"

    first = cv2.imread(frames[0])
    h, w = first.shape[:2]

    # モデル
    model = YOLO(MODEL_NAME)

    # 出力
    csv_path = os.path.join(out_dir, "tracking.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])

    video_path = os.path.join(out_dir, "preview.mp4")
    vw = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (w, h)
    )

    print("=== ReID Tracking (BoT-SORT) ===")
    print(f"処理フレーム数: {len(frames)}")

    # 1フレームずつ処理（ファイル数が多い場合のメモリ対策）
    for i, frame_path in enumerate(tqdm(frames, total=len(frames))):
        # 画像を読み込み
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # トラッキング実行（persist=True で状態を維持）
        results = model.track(
            source=img,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            classes=[0],
            tracker="botsort.yaml",
            persist=True,
            verbose=False
        )

        r = results[0]
        if r.boxes.id is None:
            continue

        frame_idx = get_frame_number(frame_path)
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, tid, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            tid = int(tid)

            writer.writerow([
                frame_idx, tid,
                f"{x1:.1f}", f"{y1:.1f}",
                f"{x2:.1f}", f"{y2:.1f}",
                f"{conf:.3f}"
            ])

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                img, f"ID:{tid}",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2
            )

        if i < PREVIEW_LIMIT:
            vw.write(img)

    vw.release()
    csv_f.close()

    print("=== 完了 ===")
    print(csv_path)
    print(video_path)

if __name__ == "__main__":
    main()
