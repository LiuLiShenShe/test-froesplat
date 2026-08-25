# M1 H-VQG 运行说明

更新日期：2026-05-20

## 目标

为新版 Plant-aware 2DGS 提供三层式视图质量门控的第一版可运行实现，并记录当前 hard filtering 负证据：

- Gate 1：Raw-view quality
- Gate 2：Mask reliability
- Gate 3：Geometry reliability

输出 retained list、逐帧分数表和汇总报告，供后续 2DGS 训练入口通过 `--raw_gate_list / --mask_gate_list / --geo_gate_list` 使用。

当前结论：已有 `A6+M1/H-VQG hard filtering` 和 `A6+M1-reject-only/raw-mask` 结果不能证明 M1 有效，反而说明 retained-list hard filtering 会破坏植物 foreground-object reconstruction。该机制保留为负对照，下一版 M1 应优先转向 soft view weighting；若必须删图，则采用 coverage-balanced anchor-view selection。

## 当前实现

### Gate 1

输入：

- `02-FFT/<sample>/filter_log.json`
- 最终锁定位姿目录下的原始 RGB 图像

指标：

- FFT `combined_norm`
- exposure score
- contrast score
- entropy score

输出：

- `raw_quality_scores.csv`
- `raw_gate_retained.txt`

### Gate 2

输入：

- `03-SAM/<sample>/mask_*.png`

指标：

- foreground ratio
- largest component ratio
- hole ratio
- boundary ratio
- temporal consistency

输出：

- `mask_reliability_scores.csv`
- `mask_gate_retained.txt`

### Gate 3

输入：

- `03-final_locked/<sample>/sparse/0/images.bin`
- `03-final_locked/<sample>/sparse/0/points3D.bin`

指标：

- registration success
- matched points
- inlier-like ratio
- reprojection error proxy
- coverage / trajectory balance（下一版补强）

输出：

- `geometry_reliability_scores.csv`
- `geometry_gate_retained.txt`
- `hvqg_report.json`

## 配置位置

```text
数据管理/07-运行脚本与超参/S6-M1-HVQG验证/configs/kongquezhu_hvqg.json
```

## 脚本位置

```text
数据管理/07-运行脚本与超参/S6-M1-HVQG验证/scripts/run_hvqg.py
```

## 当前边界

- CLIP-IQA 当前未接入，因为环境里没有现成的 `open_clip/clip` 包。
- Gate 3 的 reprojection 使用当前可取的 COLMAP proxy，而不是完整 bundle-level per-image error。
- 当前版本优先保证 retained list 可运行、可追溯；但 hard delete 策略已经在 A6 下显示为负证据，不应作为最终 M1。
- 当前 retained list 缺少显式 azimuth / trajectory coverage 约束，可能导致覆盖断裂。

## 已完成 retained-list 结果

样本：

```text
KongQueZhuYu
```

输出目录：

```text
数据管理/05-评测结果/KongQueZhuYu/M1_hvqg/
```

当前结果：

- Raw Gate retained：187
- Mask Gate retained：187
- Geometry Gate retained：75
- final keep ratio vs Raw Gate：0.4011

说明：

- 当前阈值偏保守，更适合先验证三层 gate 链路和 retained list 接口。
- 后续需要补 `keep_ratio` / score threshold 灵敏度分析，避免几何 gate 过强。

## A6+M1 retained-list hard filtering 负对照

正式对照记录：

```text
数据管理/05-评测结果/S14_M1_on_A6/kongquezhu_A6_vs_A6_HVQG_summary.md
```

训练 / 渲染加载时：

| gate | train retained | test retained |
|---|---:|---:|
| raw gate | 163/183 | 24/27 |
| mask gate | 163/163 | 24/24 |
| geometry gate | 123/163 | 17/24 |

foreground-object 指标：

| Version | PSNR_fg | SSIM_fg | LPIPS_fg | outside | leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 |
| A6+M1/H-VQG hard filtering | 12.5478 | 0.6018 | 0.1179 | 0.1743 | 0.3020 | 597116 |
| A6+M1-reject-only/raw-mask | 13.4557 | 0.6244 | 0.1115 | 0.1450 | 0.2848 | 579612 |

结论：

- hard filtering 不是可靠 M1；
- 去掉 geometry hard delete 后，raw/mask retained-list 仍明显损伤 foreground-object reconstruction；
- 两个 retained-list 分支均使 leakage / outside 超过 foreground-only 分离阈值；
- M1 下一版不能继续只按质量分数删图，必须优先保留视角覆盖，并通过 soft weighting 调节视角贡献。

## 下一版优先方向

```text
M1-v1 reject-only severe degradation
M1-v2 soft view weighting
M1-v3 coverage-balanced anchor selection
```

下一版报告项：

- retained views；
- azimuth / trajectory coverage；
- empty angular bins；
- min views per bin；
- pose graph connectivity；
- foreground-object metrics；
- train time and Gaussian count。
