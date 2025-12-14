# 04_Trajectory

トラッキングデータをメートル座標に変換し、軌跡を可視化するスクリプト。

## バージョン

| ファイル | 説明 |
|----------|------|
| `Trajectory_ver1.py` | 基本的な軌跡変換・速度計算 |
| `Trajectory_ver2.py` | ID再結合機能付き（途切れたIDを統合） |

## 使い方

1. スクリプトを開き、設定を編集：

```python
TARGET_VIDEO_NAME = "動画ファイル名.mp4"
FPS = 30                    # 元動画のFPS
SMOOTHING_WINDOW = 5        # スムージング幅
```

2. ver2のみ：ID再結合パラメータ

```python
MAX_FRAME_GAP = 3           # 最大フレーム欠損
MAX_DIST_M = 1.0            # 最大距離 (m)
MAX_SPEED_DIFF = 0.8        # 速度差 (m/s)
MAX_DIR_DIFF_DEG = 30       # 方向差 (度)
```

3. 実行：

```bash
python Trajectory_ver1.py
# または
python Trajectory_ver2.py
```

## 入出力

| 項目 | パス |
|------|------|
| 入力（設定） | `configs/[プロジェクト名].json` |
| 入力（トラッキング） | `data/projects/[プロジェクト名]/tracking/tracking.csv` |
| 入力（背景画像） | `data/projects/[プロジェクト名]/calibration/calibration_warped.jpg` |
| 出力 | `data/projects/[プロジェクト名]/trajectory/` |

## 出力ファイル

| ファイル名 | 説明 |
|-----------|------|
| `trajectory.csv` / `trajectory_ver2.csv` | 軌跡データ |
| `trajectory.png` / `trajectory_ver2.png` | 確認用プロット |

### CSVカラム

```
frame, track_id, x_meter, y_meter, vx, vy, speed, direction
```

## 依存ライブラリ

- pandas
- numpy
- opencv-python
- matplotlib
