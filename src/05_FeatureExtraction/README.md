# 05_FeatureExtraction

軌跡データから時空間的な特徴量を抽出する。

## 入力

| ファイル | パス |
|---------|------|
| 軌跡CSV | `data/projects/{project_name}/trajectory/trajectory.csv` |
| 設定JSON | `configs/{project_name}.json` |
| 鳥瞰図画像 | `data/projects/{project_name}/calibration/calibration_warped.jpg` |

## 出力

| ファイル | パス |
|---------|------|
| 特徴量CSV | `data/projects/{project_name}/features/features.csv` |
| 密度マップ画像 | `data/projects/{project_name}/features/feature_tw*.png` |

## パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `TARGET_VIDEO_NAME` | - | 対象動画ファイル名 |
| `FPS` | 30 | 動画のフレームレート |
| `MESH_SIZE_M` | 0.5 | メッシュサイズ [m] |
| `TIME_WINDOW_SEC` | 3 | 時間窓の長さ [秒] |
| `STOP_SPEED_THRESHOLD` | 0.3 | 静止判定の閾値 [m/s] |

## 特徴量

| カラム名 | 説明 |
|---------|------|
| `time_window_id` | 時間窓ID |
| `mesh_x` | メッシュX座標 |
| `mesh_y` | メッシュY座標 |
| `density` | 密度（ユニーク人数） |
| `mean_speed` | 平均速度 [m/s] |
| `mean_vx` | 平均速度ベクトルX成分 |
| `mean_vy` | 平均速度ベクトルY成分 |
| `direction_coherence` | 方向収束度（0〜1） |
| `stop_ratio` | 静止率（0〜1） |

## 実行

```bash
python FeatureExtraction_ver1.py
```
