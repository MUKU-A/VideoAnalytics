# 02_Calibration

画像上の4点を指定し、実空間座標への変換行列（ホモグラフィ行列）を計算するスクリプト。

## 使い方

1. `Calibration_ver1.py` を開き、設定を編集：

```python
TARGET_VIDEO_NAME = "動画ファイル名.mp4"  # 対象動画（01_FrameSamplingと同じ）
PREVIEW_SCALE_PX_PER_METER = 100          # プレビュー時のスケール
```

2. 実行：

```bash
python Calibration_ver1.py
```

3. 操作方法：
   - **マウス左クリック**: 点を追加（①左上 → ②右上 → ③右下 → ④左下）
   - **r キー**: リセット
   - **Enter キー**: 確定
   - **q キー**: 中断

4. 実空間の距離（メートル）を入力

## 入出力

| 項目 | パス |
|------|------|
| 入力 | `data/projects/[プロジェクト名]/frames/` 内の最初のフレーム |
| 出力（設定） | `configs/[プロジェクト名].json` |
| 出力（画像） | `data/projects/[プロジェクト名]/calibration/` |

## 出力ファイル

### JSON設定ファイル
```json
{
    "target_video": "動画名.mp4",
    "real_width_m": 3.5,
    "real_height_m": 5.0,
    "src_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "homography_matrix": [[...], [...], [...]]
}
```

### 確認用画像
- `calibration_points.jpg` - 選択した4点を描画した画像
- `calibration_warped.jpg` - 変換後の鳥瞰図

## 依存ライブラリ

- opencv-python
- numpy
