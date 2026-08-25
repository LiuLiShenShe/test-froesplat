# M5：Edge-aware Thin-leaf Meshing

## 1. 模块定位

M5 是新论文最有针对性的薄叶片创新模块。它直接回应原稿中 leaf width 存在约 9% amplification bias 的问题。

目标：

- 缓解 leaf boundary expansion。
- 降低 leaf width overestimation。
- 保持 plant height 和 canopy width 等宏观指标稳定。

## 2. 背景动机

原论文已经指出：

- leaf width 的 R2 最低，约 0.956。
- virtual measurement 会随物理尺寸增大而约 9% 高估 leaf width。
- 原因与 2DGS 边界斜视角深度不确定性、TSDF 对边缘深度方差敏感有关。

新论文应将这一点从 limitation 转化为 method motivation：

> Because phenotypic traits such as leaf width are highly sensitive to boundary dilation, we introduce edge-aware thin-leaf meshing to reduce unreliable fusion near mask boundaries and glancing views.

## 3. 推荐实现路线

优先级：

1. Edge-aware TSDF Fusion。
2. Boundary Confidence Correction。
3. Post-mesh Boundary Correction。

如果时间有限，先实现 3 作为最小可执行版本，再逐步升级到 1。

## 4. 做法一：Edge-aware TSDF Fusion

从 FSAM3 mask 中提取边界：

```text
E = Boundary(M)
```

计算像素到边界距离：

```text
d_edge = DistanceTransform(E)
```

修正 TSDF fusion weight：

```text
w' = w * f(d_edge, theta)
```

其中：

| 符号 | 含义 |
|---|---|
| `d_edge` | 当前像素/体素到 mask edge 的距离 |
| `theta` | view ray 与 normal 的夹角 |
| `w'` | 修正后的融合权重 |

直觉：

- 边界附近降低不可靠斜视角深度权重。
- 边界附近使用更小 truncation distance。
- 避免 TSDF 把叶片边界向外融合膨胀。

## 5. 做法二：Boundary Confidence Correction

计算多视角 mask 边界一致性：

- 同一空间区域被多个视角观察到的边界位置是否一致。
- 如果边界位置跨视角不稳定，则降低 TSDF fusion 权重。
- 高置信度边界保留，低置信度边界弱化或轻微收缩。

## 6. 做法三：Post-mesh Boundary Correction

最小可执行版本：

1. 识别 mesh 中薄叶片边界区域。
2. 基于 mask projection 或局部曲率定位边界 vertices。
3. 沿局部内侧方向或法向相关方向小幅收缩。
4. 对收缩幅度设置上限，避免叶片变碎或过窄。

推荐参数：

```bash
--meshing_mode post_boundary
--boundary_shrink_ratio <float>
--boundary_band_px <int>
--max_shrink_cm <float>
```

## 7. 实现接口

推荐参数：

```bash
--meshing_mode {standard,small_trunc,edge_aware,post_boundary}
--edge_band_px <int>
--edge_weight_min <float>
--edge_truncation_scale <float>
--edge_angle_threshold <float>
--boundary_confidence_threshold <float>
--boundary_shrink_ratio <float>
```

默认：

```bash
--meshing_mode standard
```

## 8. 输出文件

```text
outputs/<sample>/<method_tag>/meshing/
├── mesh_standard.ply
├── mesh_edge_aware.ply
├── edge_distance_maps/
├── boundary_confidence_maps/
├── leaf_width_measurements.csv
├── edge_meshing_report.json
└── zoom_leaf_boundary.png
```

## 9. 消融实验

| Method | 说明 |
|---|---|
| standard TSDF | 原始 TSDF |
| smaller truncation TSDF | 全局减小 truncation distance |
| edge-aware TSDF | 仅边界区域自适应 |
| post-boundary correction | mesh 后处理边界修正 |
| edge-aware TSDF + pruning | 完整组合 |

## 10. 指标

表型指标是核心：

- leaf width MAE
- leaf width RMSE
- leaf width MAPE
- leaf width bias
- leaf length MAE
- plant height MAE
- canopy width MAE

几何指标：

- Chamfer Distance，如果有 pseudo-GT 或人工清理 mesh。
- F-score。
- Normal Consistency。
- mesh connected components。
- leaf edge thickness。
- boundary visual quality。

## 11. 验收标准

M5 最好证明：

- leaf width bias 明显下降。
- leaf width MAPE 明显下降。
- plant height 和 canopy width 不恶化。
- mesh connected components 不显著增加。
- leaf boundary zoom-in 图中边界更贴近图像/人工测量。

最低可接受：

- 在有 GT 的样本上 leaf width bias 有一致下降趋势。
- 标准 TSDF 与 edge-aware/post-boundary 的对比图能肉眼看到边界膨胀被缓解。

## 12. 风险与备选

风险：

- 过度收缩导致 leaf width 被低估。
- 边界误差来自 mask，M5 可能放大 mask 错误。
- leaf edge thickness 指标如果定义不清，会导致结果难复现。

备选：

- 对边界修正设置极小幅度，只用于降低系统性 bias。
- 先对草莓等 leaf width GT 完整样本做小范围验证。
- 将 M5 写成 conservative correction，而不是声称完全解决边界膨胀。

