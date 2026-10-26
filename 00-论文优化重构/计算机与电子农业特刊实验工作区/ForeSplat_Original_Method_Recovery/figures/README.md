# figures/ — 由真实运行产物生成的图

生成脚本：`scripts/build_figures.py`（`--only chain,gcfc,sgmps,ablation,downstream,mesh`）。
所有面板都从 run 目录里读图，没有任何手绘内容。上游产物还没落盘的图会自动跳过并打印 `[skip]`。

| 文件 | 数据来源 | 内容 |
|---|---|---|
| `fig1_rap_stage_chain_0075.png` | `runs/R6_RAP_FULL_real/*` + 输入 JPEG | 单帧（0075，SG-MPS 在此帧真实切换 P2→P1）依次经过：输入 RGB → P1/P2 最佳候选 → 选择后（SG-MPS+CDR）→ SB-FBPR 后 → 残差修复后 → GCFC 后最终掩膜 → 叠加图 |
| `fig2_gcfc_stages_0050.png` | `runs/R6_RAP_FULL_real/几何修正提示图/` | GCFC 内部五个量：before → 支撑场 H → 边界带 B → 新增像素 → after（0050 帧 GCFC 新增 456 px，是该 5 帧样本中最大的一次） |
| `fig3_sgmps_confidence_vs_softmax.png` | `runs/R6_RAP_FULL_real/逐帧报告.csv` | SG-MPS 的门控置信度 $c_i^{(k)}$ 与提示词级 softmax $a^{(k)}$ 左右对照。直方显示五个提示词相差仅 0.01–0.03，τ_s=0.25 下 softmax 贴近 1/5 均匀线（灰色虚线）——这是消融中 SG-MPS 自伤的直接可视化证据 |
| `fig4_dev_ablation_f1.png` | `metrics/dev_ablation_per_scene.csv` | 跨场景 dev 集 10 档 × 4 场景的 F1 分组柱状图（依赖评分进程写出的 `dev_ablation_per_scene.csv`，尚未生成时会自动跳过） |
| `fig5_downstream_2x2.png` | `metrics/downstream_2x2.csv` | 掩膜来源 × M2M3-GO 的 2×2，四个指标（PSNR_fg/SSIM_fg/LPIPS_fg/outside ratio）并排。蓝=冻结 C_R1，红=完整 RAP。直观显示"换 RAP 掩膜后三项变差、泄漏翻倍" |
| `fig6_mesh_phenotype_geometry.png` | `metrics/mesh_geometry.json` + `metrics/mesh_silhouette_*.npz` | 左/中：`C_R1_baseline` 与 `RAP_FULL` 的最大连通域在相机上轴–水平面内的点云侧影（COLMAP 单位）；右：四档高度与冠幅柱状图及相对基线百分比（≤0.54%）。**图上标注"not cm"**，因为本场景没有经过核验的真实尺度 |

## 复现

```bash
cd 00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery
/home/test/.venvs/sam3/bin/python scripts/build_figures.py
```

依赖：Pillow（拼图）、matplotlib（柱状图，解释器 `/home/test/.venvs/sam3`，版本 3.10.8）。

`fig5` / `fig6` 需要先跑完下游训练与网格提取：

```bash
python scripts/build_downstream_2x2.py
2d-gaussian-splatting-m2m3-floor40/venv/bin/python scripts/run_mesh_phenotype.py --measure-only
```
