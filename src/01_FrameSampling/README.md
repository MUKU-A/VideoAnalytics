# 01_FrameSampling

動画から一定間隔でフレームを抽出し、画像ファイルとして保存するスクリプト。

## 使い方

1. `FrameSampling_ver1.py` を開き、設定を編集：

```python
TARGET_VIDEO_NAME = "動画ファイル名.mp4"  # 対象動画
FRAME_STEP = 6                           # 保存間隔（10 = 10フレームごと）
RESIZE_SCALE = 1.0                        # リサイズ倍率（1.0 = そのまま）
```

2. 実行：

```bash
python FrameSampling_ver1.py
```

## 入出力

| 項目 | パス |
|------|------|
| 入力 | `data/raw_videos/[動画ファイル名]` |
| 出力 | `data/projects/[プロジェクト名]/frames/` |

## 出力ファイル例

```
frame_000000.jpg
frame_000010.jpg
frame_000020.jpg
...
```

## 依存ライブラリ

- opencv-python
- numpy
