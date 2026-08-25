# Dataset Summary 初始模板

本表根据当前目录初步统计生成，后续需要继续补充 `species_en`、`scene`、`COLMAP registered`、`mask`、`manual traits` 和 `usage`。

## 1. 初始帧数与 GT 可用性

| sample_id | source_name | raw_frames | fft_frames | fft_retained_ratio | has_manual_gt | 初步用途 |
|---|---|---:|---:|---:|---|---|
| S01 | BaiZhang | 250 | 214 | 85.6% | no | reconstruction / visualization |
| S02 | CaoMei1 | 250 | 210 | 84.0% | yes | main / ablation / phenotype |
| S03 | CaoMei2 | 250 | 210 | 84.0% | no | reconstruction / visualization |
| S04 | ChangShouHua1 | 250 | 213 | 85.2% | no | reconstruction / visualization |
| S05 | ChangShouHua2 | 250 | 212 | 84.8% | yes | main / ablation / phenotype |
| S06 | ChangShouHua3 | 250 | 210 | 84.0% | no | reconstruction / visualization |
| S07 | DouBanLv1 | 250 | 215 | 86.0% | yes | main / ablation / phenotype |
| S08 | DouBanLv2 | 250 | 210 | 84.0% | no | reconstruction / visualization |
| S09 | DouBanLv3 | 250 | 210 | 84.0% | no | reconstruction / visualization |
| S10 | HongZhang | 250 | 213 | 85.2% | no | reconstruction / visualization |
| S11 | KongQueZhuYu | 250 | 210 | 84.0% | yes | main / ablation / phenotype |
| S12 | WanNianQing1 | 252 | 213 | 84.5% | no | reconstruction / visualization |
| S13 | WanNianQing2 | 250 | 215 | 86.0% | no | reconstruction / visualization |
| S14 | WangWenCao1 | 250 | 211 | 84.4% | no | reconstruction / visualization |
| S15 | WangWenCao2 | 250 | 210 | 84.0% | no | reconstruction / visualization |
| S16 | XianKeLai1 | 250 | 208 | 83.2% | yes | main / ablation / phenotype |
| S17 | XianKeLai2 | 250 | 208 | 83.2% | no | reconstruction / visualization |
| S18 | XianKeLai3 | 250 | 206 | 82.4% | no | reconstruction / visualization |
| S19 | XiangPiShu1 | 250 | 208 | 83.2% | no | reconstruction / visualization |
| S20 | XiangPiShu2 | 250 | 211 | 84.4% | no | reconstruction / visualization |

## 2. 当前可用于表型验证的候选样本

```text
CaoMei1
ChangShouHua2
DouBanLv1
KongQueZhuYu
XianKeLai1
```

这些样本优先用于：

- Table 7 表型测量准确性。
- M5 leaf width bias 验证。
- Fig. 9 phenotype validation。

## 3. 待补字段

| 字段 | 来源建议 |
|---|---|
| `species_cn/species_en` | 由样本中文名映射或实验记录补充 |
| `scene` | 固定转台/黑布、复杂室内、温室近似场景 |
| `acquisition_mode` | turntable 或 handheld |
| `colmap_registered` | 从 COLMAP `images.txt/bin` 或日志统计 |
| `has_fsam3_mask` | 检查 FSAM3/SAM3 输出目录 |
| `has_2dgs` | 检查 `05-2DGS-*` |
| `has_mesh` | 检查 `06-MESH-new` |
| `has_sugar` | 检查 `07-SuGaR-*` |
| `manual_traits` | plant height/canopy width/leaf length/leaf width |

## 4. 注意

本表中的 `raw_frames` 和 `fft_frames` 是按当前文件夹中图片数量统计，后续应与实验日志交叉验证。
