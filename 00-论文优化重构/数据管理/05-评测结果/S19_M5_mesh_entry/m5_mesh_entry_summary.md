# S19 M5 mesh-only structural and efficiency summary

## Purpose

S19 is the current M5 mesh-only structural and efficiency evaluation stage. It reuses completed `A6+M1-soft+M4` / `Ours-full` 30k models and runs mesh-only extraction variants:

```text
standard TSDF
smaller truncation TSDF
post-boundary mask-guided cleanup
```

This stage evaluates how different meshing variants affect mesh scale, connected structure, boundary stability, and mesh-only wall time after S18 closed the Gaussian representation story.

S19 must not be interpreted as evidence that leaf width or phenotype accuracy has improved.

## Source Models

| Sample | Role | Model path | Iteration |
|---|---|---|---:|
| `KongQueZhuYu` | main sample / complex background | `数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4` | 30000 |
| `XianKeLai1` | thin-leaf / fine-structure stress test | `数据管理/06-实验输出/XianKeLai1/A6_M1_soft_M4` | 30000 |

Common mesh settings: `voxel_size=0.02`, `depth_trunc=6.0`, `num_cluster=20`, `mesh_res=256`.

## Results

| Sample | Mesh variant | Vertices | Components | Largest comp. ratio | Small comps | Boundary edges | Boundary consistency | Mean disp. | P95 disp. | Mesh time/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | standard TSDF | 167789 | 8 | 0.9920 | 5 | 12088 | - | - | - | 53.33 |
| KongQueZhuYu | smaller truncation TSDF | 147665 | 20 | 0.9350 | 12 | 25086 | - | - | - | 56.52 |
| KongQueZhuYu | post-boundary cleanup | 167789 | 8 | 0.9920 | 5 | 12088 | 0.9631 | 0.0041 | 0.0222 | 58.26 |
| XianKeLai1 | standard TSDF | 74753 | 6 | 0.9488 | 0 | 6956 | - | - | - | 78.15 |
| XianKeLai1 | smaller truncation TSDF | 66138 | 12 | 0.9487 | 5 | 9763 | - | - | - | 78.57 |
| XianKeLai1 | post-boundary cleanup | 74753 | 6 | 0.9488 | 0 | 6956 | 0.8278 | 0.0121 | 0.0376 | 97.10 |

`P95 disp.` is computed between the standard post-processed mesh and the post-boundary post-processed mesh for each sample. It is a structural proxy for boundary adjustment magnitude, not a phenotype error.

`Mesh time/s` is measured as `render.py` mesh-only wall time with `--skip_train --skip_test`. It includes model/camera loading, radiance reconstruction, TSDF fusion, mesh extraction, cleanup, and optional post-boundary processing; it is not a decomposition into TSDF/cleanup substages.

## Interpretation

The mesh-only M5 path is operational on both the main sample and the thin-leaf stress sample.

`small_trunc` is the stronger structural simplification in both samples. It reduces the post-processed mesh from `167789` to `147665` vertices on `KongQueZhuYu` (`11.99%`) and from `74753` to `66138` vertices on `XianKeLai1` (`11.52%`). However, it also increases connected components from `8` to `20` on `KongQueZhuYu` and from `6` to `12` on `XianKeLai1`, while increasing boundary-edge counts. Therefore, smaller truncation produces a more compact mesh, but the current structural metrics indicate possible fragmentation risk rather than guaranteed mesh-quality improvement.

`post_boundary` keeps the same TSDF and connected-component cleanup structure as standard TSDF. It preserves component counts, largest-component ratio, and boundary-edge count in both samples, while applying conservative mask-guided vertex displacement. On `KongQueZhuYu`, the mean projected mask consistency is `0.9631`, mean displacement is `0.0041`, and P95 displacement is `0.0222`. On `XianKeLai1`, consistency drops to `0.8278`, mean displacement rises to `0.0121`, and P95 displacement rises to `0.0376`, which confirms that the thin-leaf sample is a stronger boundary stress case.

For timing, smaller truncation is similar to standard TSDF in this mesh-only wall-time measurement. Post-boundary adds modest overhead on `KongQueZhuYu` (`58.26s` vs `53.33s`) and larger overhead on `XianKeLai1` (`97.10s` vs `78.15s`). This should be reported as practical runtime evidence only, not as a geometric-quality claim.

Timing overhead relative to standard TSDF:

| Sample | Variant | Extra time/s | Relative overhead |
|---|---|---:|---:|
| KongQueZhuYu | smaller truncation TSDF | +3.19 | +5.98% |
| KongQueZhuYu | post-boundary cleanup | +4.93 | +9.24% |
| XianKeLai1 | smaller truncation TSDF | +0.42 | +0.54% |
| XianKeLai1 | post-boundary cleanup | +18.95 | +24.25% |

Current paper-safe wording:

```text
S19 has been upgraded to a mesh-only structural and efficiency evaluation. Smaller truncation reduces mesh vertices but increases fragmentation risk, while post-boundary cleanup preserves connected-component structure at the cost of additional wall time, especially on the thin-leaf XianKeLai1 sample. Current S19 results support mesh-readiness analysis but should not yet be interpreted as phenotypic accuracy improvement.
```

中文写法：

```text
S19 已升级为 mesh-only 结构与效率评价。smaller truncation 能减少网格顶点，但会增加碎片化风险；post-boundary cleanup 能保持连通域结构，但会带来额外耗时，尤其在薄叶样本 XianKeLai1 上更明显。当前 S19 结果可以支撑 mesh-readiness 分析，但不能写成表型测量精度提升。
```

## Paper Draft

为进一步评估 M5 网格化阶段对显式植物网格结构和处理效率的影响，本文在 KongQueZhuYu 和 XianKeLai1 两个样本上比较了 standard TSDF、smaller truncation TSDF 和 post-boundary cleanup 三种 mesh-only 变体。结果显示，smaller truncation 在两个样本上均减少了约 12% 的网格顶点数量，但同时增加了连通域数量和边界边数量。KongQueZhuYu 的 connected components 从 8 增加至 20，XianKeLai1 从 6 增加至 12，说明 smaller truncation 虽然能够生成更紧凑的网格，但存在碎片化风险，不能直接解释为网格质量提升。

与此不同，post-boundary cleanup 在两个样本上均保持了与 standard TSDF 相同的连通域数量，说明其主要执行局部边界几何调整，而没有破坏整体网格连通结构。在 KongQueZhuYu 上，post-boundary 的 boundary consistency 为 0.9631，mean displacement 为 0.0041，P95 displacement 为 0.0222；而在 XianKeLai1 上，boundary consistency 降至 0.8278，mean displacement 和 P95 displacement 分别升至 0.0121 和 0.0376。该结果表明，薄叶和细结构样本对边界调整更加敏感。

从处理时间看，post-boundary cleanup 在两个样本上均引入额外耗时，尤其在 XianKeLai1 上 mesh wall time 从 78.15 s 增加至 97.10 s。需要说明的是，该时间为 `render.py --skip_train --skip_test` 的整体 wall time，包含模型加载、radiance reconstruction、TSDF 融合、mesh extraction、cleanup 以及可选 post-boundary 操作，并非各子阶段的拆分计时。因此，当前 S19 结果应解释为 mesh-only 结构与效率证据，而不是表型测量精度提升证据。

## Current Status

This is an M5 entry result, not yet a paper-level phenotype result. The next formal checks should add:

- visual boundary zoom-ins for standard vs small-trunc vs post-boundary;
- edge thickness metrics;
- leaf width bias / MAE / MAPE against available GT;
- manual phenotype comparison for leaf width / leaf length when GT alignment is ready.

## Archived Outputs

```text
数据管理/05-评测结果/S19_M5_mesh_entry/KongQueZhuYu_A6_M1_soft_M4_standard/
数据管理/05-评测结果/S19_M5_mesh_entry/KongQueZhuYu_A6_M1_soft_M4_small_trunc/
数据管理/05-评测结果/S19_M5_mesh_entry/KongQueZhuYu_A6_M1_soft_M4_post_boundary/
数据管理/05-评测结果/S19_M5_mesh_entry/XianKeLai1_A6_M1_soft_M4_standard/
数据管理/05-评测结果/S19_M5_mesh_entry/XianKeLai1_A6_M1_soft_M4_small_trunc/
数据管理/05-评测结果/S19_M5_mesh_entry/XianKeLai1_A6_M1_soft_M4_post_boundary/
数据管理/05-评测结果/S19_M5_mesh_entry/m5_mesh_time.csv
数据管理/05-评测结果/S19_M5_mesh_entry/mesh_time_logs/
```
