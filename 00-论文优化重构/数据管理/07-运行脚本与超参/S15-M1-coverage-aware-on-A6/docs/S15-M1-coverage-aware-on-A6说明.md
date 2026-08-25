# S15 M1 Retained-list 负证据 on A6 说明

## 目的

S14 已证明 `A6+M1/H-VQG hard filtering` 会显著破坏 `KongQueZhuYu` foreground-object reconstruction。S15 的目标是在不改动 A6 foreground objective 的前提下，验证去掉 geometry hard delete、只保留 raw/mask retained list 后，当前 M1 是否能避免 hard filtering 的覆盖断裂问题。

## 当前先跑分支

```text
A6_M1_reject_only_raw_mask
```

设计：

- 复用 S6 的 raw / mask retained list；
- 不启用 geometry hard delete；
- 保留 A6 的 foreground track init、foreground RGB loss、alpha mask loss、bg opacity loss；
- 训练 / 渲染 / full metrics 与 A6、S14 M1-hard 保持同一 runner 和同一评价口径；
- 训练完成后补 foreground-object eval。

## 对照关系

| Variant | M1 policy | 作用 |
|---|---|---|
| A6 | none | 主基线 |
| A6+M1-hard | raw + mask + geometry hard delete | 强负对照 |
| A6+M1-reject-only/raw-mask | raw + mask only, no geometry hard delete | 弱负对照；检验失败是否只来自 geometry gate |

## 预期判断

若 `A6_M1_reject_only_raw_mask` 的 foreground 指标接近 A6，且明显优于 M1-hard，则说明 S14 的失败主要来自 geometry hard delete / 覆盖断裂，而不是所有输入质量控制都不可用。

若该分支仍显著退化，则说明问题不只是 geometry gate 过强，而是当前 retained-list hard filtering 机制本身不适合作为有效 M1。下一步应优先改为 soft view weighting，而不是继续调 retained list。

## 已完成结果

结果目录：

```text
数据管理/05-评测结果/S15_M1_coverage_aware_on_A6/kongquezhu_A6_M1_reject_only_summary.md
```

| Variant | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians | Eval images |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 | 27 |
| A6+M1-hard | 12.5478 | 0.6018 | 0.1179 | 0.1743 | 0.3020 | 597116 | 17 |
| A6+M1-reject-only/raw-mask | 13.4557 | 0.6244 | 0.1115 | 0.1450 | 0.2848 | 579612 | 24 |

结论：

- 关闭 geometry hard delete 后，指标相对 M1-hard 略有恢复；
- 但该分支仍远低于 A6，leakage 仍显著超过 foreground-only 阈值；
- 因此失败不只来自 geometry hard delete，现有 raw/mask retained-list hard filtering 也不可靠；
- 当前 M1 retained-list 机制不能作为有效模块进入主方法；
- 下一步不应继续微调 hard filtering，应优先转向 soft view weighting；若必须删图，则需要真正 coverage-balanced anchor-view selection。

正式定性：

```text
M1-hard 和 M1-reject-only/raw-mask 均为负结果。
当前证据表明，直接删除视角并不能稳定提升 foreground-object reconstruction，
反而可能破坏多视角覆盖或训练/评估视角分布。
M1 后续应从“筛掉低分图像”改为“在保持视角覆盖的前提下调节视角贡献”。
```
