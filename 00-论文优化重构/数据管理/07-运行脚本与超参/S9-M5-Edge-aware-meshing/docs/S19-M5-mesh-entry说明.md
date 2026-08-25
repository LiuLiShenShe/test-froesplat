# S19 M5 mesh-only 结构与效率评价说明

## 目的

S19 在不重新训练的前提下，复用已经完成的 `A6+M1-soft+M4` 30k checkpoint，进行 M5 mesh-only 结构与效率评价。

当前对比三种 mesh variant：

```text
standard TSDF
smaller truncation TSDF
post-boundary mask-guided cleanup
```

该阶段评价不同 mesh variant 对网格规模、连通结构、边界稳定性和 mesh-only wall time 的影响，不把结果直接写成 leaf width 或 phenotype 改进。

当前安全定性：

```text
S19 已升级为 mesh-only 结构与效率评价。当前结果用于比较不同网格化策略带来的网格结构和处理耗时变化，尚不能用于证明表型测量精度提升。
```

## 输入模型

| Sample | Role | Model path |
|---|---|---|
| `KongQueZhuYu` | 主样本 / 复杂背景 | `数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4` |
| `XianKeLai1` | 薄叶 / 细结构压力测试 | `数据管理/06-实验输出/XianKeLai1/A6_M1_soft_M4` |

该模型对应 S18 的 `Ours-full / Ours-compact = A6 + M1-soft + M4`。

## 运行命令模板

直接调用 `render.py` 时需要补齐 2DGS 的 `PYTHONPATH`：

```bash
export PYTHONPATH=/data/fj/F2DMAS/2d-gaussian-splatting-main:/data/fj/F2DMAS/2d-gaussian-splatting-main/submodules/diff-surfel-rasterization:/data/fj/F2DMAS/2d-gaussian-splatting-main/submodules/simple-knn
```

Standard TSDF：

```bash
CUDA_VISIBLE_DEVICES=1 python render.py \
  --model_path <A6_M1_soft_M4_model_path> \
  --iteration 30000 \
  --skip_train --skip_test \
  --voxel_size 0.02 --depth_trunc 6.0 --sdf_trunc 0.08 \
  --num_cluster 20 --mesh_res 256 \
  --meshing_mode standard
```

Smaller truncation TSDF：

```bash
CUDA_VISIBLE_DEVICES=1 python render.py \
  --model_path <A6_M1_soft_M4_model_path> \
  --iteration 30000 \
  --skip_train --skip_test \
  --voxel_size 0.02 --depth_trunc 6.0 --sdf_trunc 0.08 \
  --num_cluster 20 --mesh_res 256 \
  --meshing_mode small_trunc --edge_truncation_scale 0.5
```

Post-boundary cleanup：

```bash
CUDA_VISIBLE_DEVICES=1 python render.py \
  --model_path <A6_M1_soft_M4_model_path> \
  --iteration 30000 \
  --skip_train --skip_test \
  --voxel_size 0.02 --depth_trunc 6.0 --sdf_trunc 0.08 \
  --num_cluster 20 --mesh_res 256 \
  --meshing_mode post_boundary --boundary_shrink_ratio 0.08
```

注意：`render.py` 会把 `mesh_metrics.json` 写入 model 目录根部，因此每个 variant 跑完后都需要立刻归档 `mesh_metrics.json` 和对应 PLY，避免被下一次 mesh run 覆盖。

## S19 当前结果

| Sample | Mesh variant | Vertices | Components | Largest comp. ratio | Small comps | Boundary edges | Consistency | Mean disp. | P95 disp. | Mesh time/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | standard TSDF | 167789 | 8 | 0.9920 | 5 | 12088 | - | - | - | 53.33 |
| KongQueZhuYu | smaller truncation TSDF | 147665 | 20 | 0.9350 | 12 | 25086 | - | - | - | 56.52 |
| KongQueZhuYu | post-boundary cleanup | 167789 | 8 | 0.9920 | 5 | 12088 | 0.9631 | 0.0041 | 0.0222 | 58.26 |
| XianKeLai1 | standard TSDF | 74753 | 6 | 0.9488 | 0 | 6956 | - | - | - | 78.15 |
| XianKeLai1 | smaller truncation TSDF | 66138 | 12 | 0.9487 | 5 | 9763 | - | - | - | 78.57 |
| XianKeLai1 | post-boundary cleanup | 74753 | 6 | 0.9488 | 0 | 6956 | 0.8278 | 0.0121 | 0.0376 | 97.10 |

解释：

- smaller truncation 在两个样本上均减少约 12% post vertices，但同时提高 connected components 和 boundary edge count，因此只能说明网格更紧凑，且存在碎片化风险，不能直接说明几何更准。
- post-boundary 在两个样本上不改变 connected-component cleanup 后的顶点数量和连通域结构，主要体现为保守的局部几何调整。
- `XianKeLai1` 的 boundary consistency 低于 `KongQueZhuYu`，mean displacement 更高，说明薄叶/细结构确实是更强的边界压力测试。
- mesh time 是 `render.py --skip_train --skip_test` 的 mesh-only wall time，包含加载、radiance reconstruction、TSDF、mesh extraction 和 cleanup，不拆分 TSDF / cleanup 子阶段。

相对 standard TSDF 的耗时变化：

| Sample | Variant | Extra time/s | Relative overhead |
|---|---|---:|---:|
| KongQueZhuYu | smaller truncation TSDF | +3.19 | +5.98% |
| KongQueZhuYu | post-boundary cleanup | +4.93 | +9.24% |
| XianKeLai1 | smaller truncation TSDF | +0.42 | +0.54% |
| XianKeLai1 | post-boundary cleanup | +18.95 | +24.25% |

归档结果：

```text
数据管理/05-评测结果/S19_M5_mesh_entry/
```

## 下一步

1. 对三种 mesh 输出做边界 zoom-in 和 edge thickness 统计。
2. 如需要，后续再做更细的 TSDF / cleanup 子阶段计时。
3. 接入 GT 后再报告 leaf width bias、MAE 和 MAPE。
