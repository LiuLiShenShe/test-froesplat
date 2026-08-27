# 阶段十 A6/A7 + 证据引导 SPNP 四样本产物整理

整理日期：2026-08-27

## 目录结构

```
阶段十A6A7验证产物/
├── 成功产物/          # F1>=0.9 且 GT 口径干净，可直接用于论文
├── GT混标待核/        # F1 高但 GT 标注口径需人工核对
├── 失败产物/          # F1<0.6，SAM3 候选级失败（管道无法纠正）
├── 冒烟证据图/        # KongQueZhuYu 10帧冒烟 A6/A7/证据引导SPNP 三联/四联图
└── 阶段十A6A7产物整理README.md
```

## 成功产物（10 帧）

| 样本_帧 | 备注 | 图片 |
| --- | --- | --- |
| CaoMei1_0025 | F1=0.955，去盆干净 | `成功产物/CaoMei1_0025_P2|GT|P6.png` |
| CaoMei1_0050 | F1=0.969，去盆干净 | `成功产物/CaoMei1_0050_P2|GT|P6.png` |
| ChangShouHua2_0000 | F1=0.821，去盆基本干净 | `成功产物/ChangShouHua2_0000_P2|GT|P6.png` |
| ChangShouHua2_0025 | F1=0.978，去盆干净 | `成功产物/ChangShouHua2_0025_P2|GT|P6.png` |
| ChangShouHua2_0050 | F1=0.984，去盆干净 | `成功产物/ChangShouHua2_0050_P2|GT|P6.png` |
| DouBanLv1_0025 | F1=0.974，去盆干净 | `成功产物/DouBanLv1_0025_P2|GT|P6.png` |
| DouBanLv1_0050 | F1=0.986，去盆干净 | `成功产物/DouBanLv1_0050_P2|GT|P6.png` |
| DouBanLv1_0075 | F1=0.977，去盆干净 | `成功产物/DouBanLv1_0075_P2|GT|P6.png` |
| DouBanLv1_0100 | F1=0.970，去盆干净 | `成功产物/DouBanLv1_0100_P2|GT|P6.png` |
| XianKeLai1_0000 | F1=0.971，A6A7SPNP提升最大帧 | `成功产物/XianKeLai1_0000_P2|GT|P6.png` |

## GT 混标待核（3 帧）

> 这些帧 F1 数值不低，但 GT 标注口径（去盆 vs 带盆）存在混标，需人工确认后再决定是否纳入主表。

| 样本_帧 | 备注 | 图片 |
| --- | --- | --- |
| CaoMei1_0000 | F1=0.928，GT去盆/带盆混标，需人工核对 | `GT混标待核/CaoMei1_0000_P2|GT|P6.png` |
| CaoMei1_0075 | F1=0.951，掩膜含盆但GT去盆→F1虚高，需核对 | `GT混标待核/CaoMei1_0075_P2|GT|P6.png` |
| ChangShouHua2_0075 | F1=0.758，GT口径混标，需核对 | `GT混标待核/ChangShouHua2_0075_P2|GT|P6.png` |

## 失败产物（3 帧）

> 这些帧为 SAM3 候选级失败（前端分割即失败，A6/A7/SPNP 无法纠正），属模型能力边界，非管道 bug。

| 样本_帧 | 备注 | 图片 |
| --- | --- | --- |
| CaoMei1_0100 | F1=0.229，SAM3候选失败(只割到盆丢植株) | `失败产物/CaoMei1_0100_P2|GT|P6.png` |
| ChangShouHua2_0100 | F1=0.436，GT含盆+深色叶漏分 | `失败产物/ChangShouHua2_0100_P2|GT|P6.png` |
| DouBanLv1_0000 | F1=0.481，GT口径混标(标了盆) | `失败产物/DouBanLv1_0000_P2|GT|P6.png` |

## 冒烟证据图（KongQueZhuYu 10 帧）

- `final_*_3panel.png`：A1s选中 | A6共识投票 | 最终掩膜（粘连切割验证）
- `a7_*_4panel.png`：A1s选中 | A6共识 | A7记忆传播 | 最终掩膜
- `evspnp_*_3panel.png`：A6共识 | 证据引导SPNP细化 | 最终掩膜（花盆剔除+邻株消失）
- `bridge_*_3panel.png`：粘连桥切割诊断三联图

## 读图约定

- 三联图（P2|GT|P6）：左=P2去盆最终掩膜叠加，中=P2叠加+GT绿色轮廓，右=P6带盆最终掩膜叠加
- 四联图（A1s|共识|记忆|最终）：从左到右即管线顺序，红色半透明为掩膜覆盖区
- 真正用于论文和 3D 重建的是最右图（或三联图右图）

## 四样本 GT 帧专项对比图（冒烟证据图/）

以下为 P2|GT|P6 三联图之外的补充证据（`/tmp` 固化）：

| 文件 | 内容 |
| --- | --- |
| `caomei1_0075_gt.png` | CaoMei1 0075：掩膜含盆 vs GT去盆 混标对照 |
| `caomei1_0100_gt.png` | CaoMei1 0100：SAM3只割到盆丢植株（失败） |
| `csh2_0075_gt.png` | ChangShouHua2 0075：GT口径混标对照 |
| `csh2_0100_gt.png` | ChangShouHua2 0100：GT含盆+深色叶漏分（失败） |
| `dbl1_0000_gt.png` | DouBanLv1 0000：GT标了盆（混标/失败边界） |
| `0007_compare.png` | KongQueZhuYu 0007 粘连帧早期对比 |

## 分类阈值说明

- **成功**：F1 ≥ 0.9 且 GT 口径干净（去盆样本的 GT 确实去盆，P2/P6 均接近 GT）
- **GT 混标待核**：F1 数值不低，但 GT 标注口径存在去盆/带盆混标，需人工确认后再决定是否纳入主表
- **失败**：F1 < 0.6，属 SAM3 候选级失败（前端分割即失败，A6/A7/SPNP 无法纠正），是模型能力边界而非管道 bug
