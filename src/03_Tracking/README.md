# 03_Tracking

BoT-SORT（ReID付きトラッカー）を使用した人物トラッキングスクリプト。

## 使い方

1. `Tracking_ver1.py` を開き、設定を編集：

```python
TARGET_VIDEO_NAME = "動画ファイル名.mp4"  # 対象動画（01_FrameSamplingと同じ）
MODEL_NAME = "yolov8x.pt"                 # 検出モデル
YOLO_CONF = 0.1                           # 検出信頼度閾値
YOLO_IOU = 0.9                            # NMS IoU閾値
```

2. 実行：

```bash
python Tracking_ver1.py
```

## 入出力

| 項目 | パス |
|------|------|
| 入力 | `data/projects/[プロジェクト名]/frames/` |
| 出力 | `data/projects/[プロジェクト名]/tracking/` |

## 出力ファイル

| ファイル名 | 説明 |
|-----------|------|
| `tracking.csv` | トラッキング結果 |
| `preview.mp4` | 確認用動画（0.2倍速） |

### CSVカラム

```
frame, track_id, x1, y1, x2, y2, conf
```

## 依存ライブラリ

- ultralytics
- opencv-python
- numpy
- tqdm
