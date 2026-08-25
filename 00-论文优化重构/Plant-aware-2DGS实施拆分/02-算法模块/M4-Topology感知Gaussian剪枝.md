# M4：Topology-aware Gaussian Pruning

## 1. 模块定位

M4 是第二个方法模块，目标是把原稿中的 brightness-based pruning 升级为 multi-criteria topology-aware pruning。

原始版本：

- 主要基于 Gaussian base color / brightness 删除暗色伪影。

新版本：

- 结合 mask consistency、opacity、view coverage、brightness、connected component 等信息，删除背景粘连、漂浮点、暗色伪影和小碎片。

## 2. 核心评分

建议定义：

```text
S_i = alpha * M_i + beta * O_i + gamma * V_i + delta * B_i + eta * C_i
```

字段：

| 符号 | 含义 |
|---|---|
| `M_i` | mask consistency，Gaussian 投影是否主要位于植物 mask 内 |
| `O_i` | opacity，Gaussian 不透明度 |
| `V_i` | view coverage，被多少视角稳定观测 |
| `B_i` | brightness/color abnormality，颜色异常程度 |
| `C_i` | connected component，是否属于主体连通结构 |

## 3. 删除规则

Gaussian 满足以下任一条件可删除：

1. 多视角投影大部分落在 background mask。
2. opacity 低于阈值。
3. brightness 异常且 view coverage 低。
4. 不属于最大或主要 connected component。
5. 对最终渲染贡献极低。

建议实现时从简单到复杂：

- V1：opacity + brightness + mask consistency。
- V2：加入 view coverage。
- V3：加入 connected component。

## 4. 实现接口

推荐参数：

```bash
--pruning_mode {none,opacity,brightness,mask,topology}
--pruning_start_iter <int>
--pruning_interval <int>
--pruning_opacity_threshold <float>
--pruning_brightness_threshold <float>
--pruning_mask_threshold <float>
--pruning_min_view_coverage <int>
--pruning_keep_largest_components <int>
--pruning_score_weights <alpha,beta,gamma,delta,eta>
--save_pruning_report
```

默认：

```bash
--pruning_mode none
```

## 5. 输出文件

```text
outputs/<sample>/<method_tag>/pruning/
├── pruning_config.yaml
├── pruning_report.json
├── removed_gaussians.ply
├── kept_gaussians.ply
├── score_histograms.png
└── before_after_render_grid.png
```

`pruning_report.json` 字段：

```json
{
  "gaussians_before": 0,
  "gaussians_after": 0,
  "pruning_ratio": 0.0,
  "removed_by_opacity": 0,
  "removed_by_brightness": 0,
  "removed_by_mask": 0,
  "removed_by_component": 0,
  "psnr_before": null,
  "psnr_after": null
}
```

## 6. 消融实验

| Pruning method | 说明 |
|---|---|
| no pruning | 不剪枝 |
| opacity pruning | 仅基于 opacity |
| brightness pruning | 原稿已有基础 |
| mask pruning | 仅基于 mask consistency |
| topology-aware pruning | 多条件联合剪枝 |

组合：

```text
baseline+M4
baseline+M2+M4
baseline+M2+M3+M4
full: baseline+M1+M2+M3+M4+M5
```

## 7. 指标

结构：

- Gaussian number
- pruning ratio
- connected component count
- floating artifact ratio
- background Gaussian ratio

渲染：

- PSNR
- SSIM
- LPIPS

效率：

- mesh time
- memory
- total pipeline time

表型：

- trait MAE
- leaf width MAPE
- plant height bias

## 8. 论文写法

推荐表述：

> Instead of pruning Gaussians solely according to opacity or color statistics, we estimate whether each primitive is topologically consistent with the plant foreground across multiple views. This strategy removes background-adhered and weakly observed primitives while preserving organ-level structures required for phenotypic measurement.

## 9. 验收标准

M4 需要证明：

- Gaussian 数量减少。
- mesh time 下降。
- floating artifacts 减少。
- 表型误差不增加，最好下降。
- PSNR/SSIM/LPIPS 不明显下降。

注意：

- 剪枝不能只追求 Gaussian 数量减少。
- 如果 pruning 后 leaf width 或 leaf length 误差上升，必须回退阈值或改用更保守策略。
- 必须可视化 removed vs kept Gaussians，证明没有剪掉主体叶片。

## 10. 风险与备选

风险：

- mask consistency 对细叶柄过于严苛。
- connected component 可能误删被遮挡导致的真实分离结构。
- brightness pruning 可能误删阴影叶片。

备选：

- 对 leaf/stem 区域设置更低 pruning 强度。
- 最大 component 之外保留 top-K component，而不是只保留最大 component。
- 对低 brightness 但 high view coverage 的 Gaussian 保留。

