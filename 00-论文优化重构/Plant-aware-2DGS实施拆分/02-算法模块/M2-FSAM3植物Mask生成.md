# M2：FSAM3 Plant Mask Generation

## 1. 模块定位

M2 提供植物 foreground mask，是 M3 mask-constrained optimization、M4 mask pruning、M5 edge-aware meshing 的共同先验。

论文中角色：

- 从“好看的分割结果”升级为“后续 3D reconstruction 的可用几何先验”。
- 通过 prompt sensitivity 和 downstream reconstruction 指标证明 mask 的实际价值。

工程中角色：

- 生成 binary mask、alpha image、foreground-only image。
- 记录 prompt、post-processing 参数和 mask 质量。
- 默认不改变 baseline；只有 `--mask_mode preprocess/alpha` 或后续模块显式需要时才使用。

## 2. 推荐 prompt 组合

| Prompt ID | Prompt |
|---|---|
| P1 | green plant |
| P2 | entire plant excluding pot |
| P3 | leaves and stems |
| P4 | crop seedling |
| P5 | plant body without background |

建议默认策略：

- 先用 P2 作为主 prompt。
- 对代表性样本做 P1-P5 sensitivity。
- 如果单 prompt 不稳定，再设计 prompt ensemble。

## 3. 后处理

推荐流程：

1. FSAM3 生成初始 mask。
2. 对连续帧做 tracking/propagation，保持多视角一致。
3. morphological closing 填补孔洞。
4. 删除极小 connected components。
5. 生成：
   - binary mask
   - RGBA alpha image
   - foreground-only image
   - boundary overlay visualization

## 4. 输出文件

建议目录：

```text
outputs/<sample>/M2_masks/<prompt_id>/
├── masks/
├── rgba/
├── foreground_rgb/
├── overlays/
├── mask_metrics.json
├── prompt.txt
└── postprocess_config.yaml
```

`mask_metrics.json` 字段：

```json
{
  "prompt_id": "P2",
  "f1": null,
  "miou": null,
  "hd95": null,
  "boundary_error": null,
  "component_count_mean": null,
  "hole_area_ratio_mean": null
}
```

## 5. 实验设计

### 5.1 分割性能

对比：

| Method | 说明 |
|---|---|
| HSV/ExG/Otsu | 传统颜色 baseline |
| SAM/SAM2/SAM3 | foundation model baseline |
| SEEM | 原论文已有 baseline |
| FSAM3 | 主方法 mask |
| FSAM3 + post-processing | 最终输入 |

指标：

- F1-score
- mIoU
- HD95
- boundary error
- component count

### 5.2 Prompt sensitivity

| Prompt | F1 | mIoU | HD95 | downstream PSNR | leaf width MAPE |
|---|---:|---:|---:|---:|---:|
| P1 |  |  |  |  |  |
| P2 |  |  |  |  |  |
| P3 |  |  |  |  |  |
| P4 |  |  |  |  |  |
| P5 |  |  |  |  |  |

### 5.3 Downstream impact

关键对比：

```text
raw 2DGS
FSAM3-preprocessed 2DGS
FSAM3 + mask-constrained 2DGS
```

## 6. 与其他模块接口

M3 需要：

- binary mask aligned with training images
- optional alpha supervision target
- camera/image filename mapping

M4 需要：

- mask projection consistency
- background/foreground lookup for Gaussian projections

M5 需要：

- mask boundary map
- distance transform to mask edge
- boundary confidence across views

## 7. 论文写法

推荐表达：

> FSAM3 is used not only as a foreground extractor but also as a source of multi-view plant priors. The generated masks provide supervision for Gaussian opacity, support topology-aware pruning, and define uncertain boundary regions during thin-leaf meshing.

## 8. 验收标准

M2 成功标准：

- 每个样本能生成与训练图像一一对应的 mask。
- prompt 和后处理参数可追溯。
- 至少完成一个 prompt sensitivity test。
- 能证明 FSAM3 mask 对 downstream reconstruction 或 phenotype accuracy 有正向作用。

风险：

- prompt 对 pot/soil 是否保留可能影响 plant height 和 canopy bottom。
- 过度 closing 可能导致叶间孔洞被填满，影响叶宽。
- mask 边界误差会传递到 M3 和 M5，因此必须保留 boundary metric。

