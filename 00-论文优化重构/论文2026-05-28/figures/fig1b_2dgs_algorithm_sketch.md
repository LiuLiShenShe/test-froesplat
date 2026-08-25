# Fig. 1b 2DGS Algorithm Sketch for ForeSplat

这张图的目标不是做数据集验证，而是把 `2.4-2.6` 里的方法改动画清楚。

## 可复用的底图思路

可以参考 2DGS 官方论文里那种很直白的三段式流程：

1. 输入多视角 RGB 和相机
2. 2D Gaussian 初始化与训练
3. 网格/表面提取输出

这种底图是可以用的，因为它结构清楚，读者一眼能看出原始 2DGS 的训练逻辑。

## 需要替换或新增的 ForeSplat 元素

不要把这张图画成数据覆盖图，也不要画成结果对比拼图。它应该突出方法改动：

1. 在输入端加入 `FSAM3 mask`
2. 在初始化端加入 `foreground track init`
3. 在优化端加入 `foreground RGB supervision`
4. 在损失端加入 `alpha mask loss` 和 `background opacity loss`
5. 在训练调度端加入 `view quality soft weighting`
6. 在后处理端加入 `mask-guided pruning`
7. 在输出端保留 `TSDF mesh`，但强调输出的是 `plant-only mesh`

## 推荐版式

建议画成横向流程图，分 3 到 4 个区块注重Block 2和Block 3：

### Block 1: Input and prior

- 多视角 RGB
- COLMAP poses
- FSAM3 masks

### Block 2: 2DGS training core

- Gaussian initialization
- foreground-only RGB loss
- alpha / background regularization
- soft view weighting

### Block 3: Post-training cleanup

- mask-guided pruning
- TSDF mesh extraction

### Block 4: Output

- plant-only mesh
- measurement-ready representation

## 视觉重点

1. 原始 2DGS 流程用灰色或浅蓝色表达。
2. ForeSplat 新增模块用绿色或橙色强调。
3. 箭头尽量少而清楚，不要堆太多分支。
4. 如果想保留官方 2DGS 图的影子，可以只借它的“输入 -> 训练 -> 网格”骨架，不要直接把数据集验证图裁来改。
5. 图注里要明确这是 `training pipeline with ForeSplat modifications`，不是实验结果图。

## 推荐草图文案

- `Input views`
- `COLMAP poses`
- `FSAM3 plant masks`
- `Foreground track initialization`
- `Foreground RGB supervision`
- `Alpha/background constraints`
- `Soft view weighting`
- `Mask-guided pruning`
- `TSDF mesh`

## 你现在这篇文章里怎么落位

这张图应放在 `2.4` 前后，作为方法段的算法图。
它和你已有的 `图 1` 总览分工如下：

- `图 1`：整个 ForeSplat 工作流总览
- `图 1b`：2DGS 算法层面的改造示意
- `图 2`：FSAM3 前景先验质量或后续结果图

