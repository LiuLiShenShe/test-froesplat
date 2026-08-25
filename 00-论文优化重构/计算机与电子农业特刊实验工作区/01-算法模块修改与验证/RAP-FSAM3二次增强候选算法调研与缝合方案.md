# RAP-FSAM3 二次增强候选算法调研与缝合方案

## 当前判断

孔雀竹芋 GT5 已经能证明 A2 结构化正负提示有效；但现有 A3-A5 在当前标注子集上更多是保护性流程和日志机制，独立增益不明显。若论文只强调 A2，创新点偏单薄。因此建议把 RAP-FSAM3 从“单一结构化提示修正”升级为三层适配框架：

```text
语义目标定位 -> 边界细化 -> 多视角/重建一致性反馈
```

这三层都能贴合植物前景 2DGS 重建，不只是把外部模型堆在一起。

## 候选算法筛选标准

| 标准 | 说明 |
| --- | --- |
| 可落地 | 能接到当前 `生成RAP-FSAM3掩膜.py` 的候选、评分、后处理或几何反馈节点 |
| 可消融 | 能形成清楚的 A6/A7/A8 或替换 A1/A3/A5 的实验设置 |
| 与任务相关 | 直接改善植物前景、花盆/背景泄漏、细结构边界或多视角一致性 |
| 不喧宾夺主 | 不把 ForeSplat 变成外部模型横向拼盘，而是作为农业重建适配策略 |

## 推荐优先级

### R1：语义定位先验，修正提示词选择

**可借鉴方法**：Grounding DINO / Grounded SAM 思路。

**动机**：仙客来 GT1 说明当前 A1 评分可能选错提示词：P2 candidate 最好，但 A1 选到 P3 后召回下降。这个问题不是 A2 能完全补救的，应该在候选选择前加一个“目标定位锚”。

**缝合方式**：

- 对输入图像运行开放词汇检测，提示词可设为：
  - positive：`potted plant`, `plant`, `leaves and stems`
  - negative/distractor：`pot`, `table`, `background plant`, `support`
- 输出植物目标框、花盆框和邻近干扰框。
- 在候选评分里加入：
  - `box_iou`：候选掩膜 bbox 与植物框一致性。
  - `pot_overlap_penalty`：候选与花盆框重叠惩罚。
  - `distractor_side_penalty`：候选触及侧边邻近植物框的惩罚。
- 作为 A1 的替换增强：`A1+semantic_box_gate`。

**预期贡献表述**：

> We introduce a semantic target gate that converts open-vocabulary grounding cues into mask reliability terms, preventing prompt selection from drifting to under-segmented or distractor-biased candidates.

**实现成本**：中等。若暂不引入 Grounding DINO 权重，可先用 SAM3 概念框/当前候选 bbox 的伪 box 做同构接口，后续替换为 Grounding DINO。

### R2：边界质量 refiner，替换目前效果弱的 A3 形态学残差修复

**可借鉴方法**：HQ-SAM、BiRefNet。

**动机**：A3 当前只靠开闭运算残差，孔雀竹芋和仙客来上都没有明显二值增益。更像工程小修，不够像论文贡献。植物叶缘、叶尖、叶柄属于高分辨率细边界问题，更适合引入边界/梯度参考的 mask refiner。

**缝合方式**：

- 输入：A2 掩膜 + 原图。
- 只在 A2 边界不确定环带内细化，避免 refiner 改动整株主体。
- 可选实现路线：
  - `HQ-SAM`：用 A2 bbox/点提示重新预测高质量 mask。
  - `BiRefNet`：生成高分辨率二值前景图，再与 A2 做置信融合。
  - 轻量 fallback：当前实现先保留形态学 A3，但新增 `boundary_refine_backend=none|hq_sam|birefnet|edge_band`。
- 输出：
  - `边界细化掩膜/`
  - `边界不确定环带/`
  - `boundary_refine_delta.csv`

**预期贡献表述**：

> Instead of applying global morphological repair, we refine only the uncertain boundary band using high-quality segmentation priors, preserving thin plant structures while suppressing background leakage.

**实现成本**：中等到偏高，取决于本机是否安装权重。可先做接口和 `edge_band` 版本，再接 HQ-SAM/BiRefNet。

### R3：多视角一致性传播，把 A4/A5 从“标记机制”升级为“修正机制”

**可借鉴方法**：SAM 2 视频分割、Cutie VOS、SA3D/Gaussian Grouping 的跨视图一致性思想。

**动机**：当前 A4/A5 只标记突变帧或几何 low，并没有真正把邻帧/重建信息反向修正掩膜。因此指标上很难超过 A2。需要把“检测”升级为“传播/反馈修正”。

**缝合方式**：

- 选择高置信关键帧：A2/A3 得分高、几何分数高、面积稳定的帧。
- 在相邻视角序列中传播前景：
  - 短期路线：SAM2/Cutie 风格的 mask propagation，生成 `propagated_mask`。
  - 轻量路线：用相邻帧掩膜 IoU、光流/特征匹配或 COLMAP track projection 做投影一致性约束。
- 对低置信帧执行：
  - 欠分割：把落在 GT-like 几何前景点但不在 mask 内的区域转为正点/正框。
  - 过分割：把几何无支撑的大块区域转为负点/负框。
- 输出 `A5_repaired`，而不只是 `reprompt_flag`。

**预期贡献表述**：

> We close the loop between foundation-model masks and reconstruction by projecting multi-view foreground evidence back to uncertain masks, converting reconstruction feedback from a diagnostic score into corrective prompts.

**实现成本**：中等。最建议优先做 COLMAP track projection 的 corrective prompts，因为现有脚本已有几何反馈和 mask stem 匹配。

### R4：3D Gaussian foreground identity field，作为中期高创新模块

**可借鉴方法**：Gaussian Grouping、GARField、SA3D。

**动机**：如果要把论文主线从“2D 掩膜适配”抬升到“重建感知前景对象”，最强的新意是让 2DGS 中的 Gaussian 学到 foreground identity，而不是只在训练前/后使用 mask。

**缝合方式**：

- 在 2DGS 训练中给每个 Gaussian 增加 foreground/object logit。
- 用 RAP-FSAM3 mask 监督渲染出的 identity map。
- 训练后将 identity map 反投影回图像，生成 mask consistency map。
- 用该 consistency map：
  - 评估背景泄漏；
  - 指导剪枝；
  - 作为 A5 的几何反馈升级版。

**预期贡献表述**：

> We transform 2D visual foundation model masks into a 3D foreground identity prior for Gaussian splats, enabling object-level reconstruction and pruning.

**实现成本**：高。适合实验四/ForeSplat 主线，不建议马上塞进实验三。

## 不建议优先缝合的方向

| 方向 | 原因 |
| --- | --- |
| 单纯更多文本提示词 | 仙客来已显示提示词选择会选错，只加提示词可能放大不稳定 |
| 只调形态学核大小 | 难形成论文创新，且对同连通域泄漏无效 |
| 纯最大连通域/面积阈值 | 已验证对孔雀竹芋粘连泄漏几乎无效 |
| 全量引入多个大模型并投票 | 容易被审稿人认为是 ensemble 堆料，贡献不清 |

## 建议新增消融版本

| 编号 | 名称 | 核心新增 | 目标 |
| --- | --- | --- | --- |
| A1s | Semantic-gated selection | 语义目标框/花盆框约束提示词选择 | 修复仙客来 P3 选错、孔雀竹芋花盆泄漏 |
| A3b | Boundary-aware refinement | HQ-SAM/BiRefNet/edge-band 边界细化 | 替代弱形态学残差修复，提高 Boundary F1、HD95 |
| A5c | Corrective geometry feedback | 几何反馈转正负提示修正 | 让 A5 产生真实 mask delta，而不只是日志 |
| B5i | Gaussian foreground identity | 2DGS foreground identity field | 强化实验四，成为 ForeSplat 主创新 |

## 最推荐的近期执行顺序

1. **先做 A1s**：因为它直接解决仙客来 GT1 暴露的问题，也能解释孔雀竹芋 P2/P4/P3 的提示词差异。
2. **再做 A5c**：沿用现有 COLMAP 几何反馈，把 low/ok 从记录变成正负提示生成。
3. **最后做 A3b**：如果环境能快速接 HQ-SAM/BiRefNet，就做；否则先用 edge-band refiner 作为轻量版。

## 可写成论文贡献的版本

推荐把 RAP-FSAM3 的贡献重新组织为：

1. **Semantic-gated multi-prompt selection**：面向植物目标和干扰物的开放语义可靠性评分。
2. **Structured positive-negative refinement**：已由孔雀竹芋 GT5 证明有效的 A2。
3. **Reconstruction-consistent corrective prompting**：把 COLMAP/2DGS 多视角证据反馈为修正提示，而不是只做异常标记。
4. **Mask-guided Gaussian foreground identity**：放到实验四/ForeSplat 主线，不一定并入实验三。

这样比“只有 A2 起效”更完整，也能避免硬吹 A3-A5。

## 参考来源

- Grounding DINO: open-set detector with language/referring-expression grounding, suitable for semantic target and distractor boxes. https://arxiv.org/abs/2303.05499
- HQ-SAM: high-quality SAM variant for intricate structures and fine mask details. https://arxiv.org/abs/2306.01567
- BiRefNet: high-resolution dichotomous segmentation with localization/reconstruction and gradient references. https://arxiv.org/abs/2401.03407
- SAM 2: promptable segmentation for images and videos with streaming memory, suitable for mask propagation. https://arxiv.org/abs/2408.00714
- Cutie: object-level memory for robust video object segmentation under distractors. https://arxiv.org/abs/2310.12982
- SA3D: one-shot SAM-based 3D segmentation over NeRFs, useful as cross-view segmentation inspiration. https://jumpat.github.io/SA3D/
- Gaussian Grouping: object grouping/segmentation in 3D Gaussian scenes. https://arxiv.org/abs/2312.00732
