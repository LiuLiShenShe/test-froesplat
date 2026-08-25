

## 一、对 SAM3/FSAM3 可以新增的 4 个改进模块

### 1. Prompt Ensemble and Selection：提示词集成与自动选择

你现在已经用了 P1-P5 五个 prompt，并且发现 P2 和 P5 最稳定，P1 容易包含绿色背景，P3 容易漏掉粗茎或花叶混合结构，P4 容易低分割成熟植物。这个结果已经很好，但目前只是“比较 prompt”。建议把它升级成方法模块：**不是人工选一个 prompt，而是让系统自动从多 prompt 输出中选择或融合最可靠的 plant foreground prior。**你现在稿子里已经有 P1-P5 的定义和 P2 的选择理由，可以直接扩展。

具体做法：

对每一帧 (I_i)，用多个 prompt 得到候选掩膜：

[
\mathcal{M}_i={M_i^{(1)},M_i^{(2)},...,M_i^{(K)}}.
]

然后计算每个候选掩膜的可靠性分数：

[
S(M_i^{(k)})=
\lambda_1 Q_{\text{area}}
+\lambda_2 Q_{\text{comp}}
+\lambda_3 Q_{\text{edge}}
+\lambda_4 Q_{\text{temp}}
+\lambda_5 Q_{\text{contrast}}.
]

其中：

* (Q_{\text{area}})：前景面积是否在合理范围内，防止把背景或花盆全选进去；
* (Q_{\text{comp}})：连通域数量是否过多；
* (Q_{\text{edge}})：边界是否稳定清晰；
* (Q_{\text{temp}})：相邻帧 mask IoU 是否平稳；
* (Q_{\text{contrast}})：前景和背景的 RGB 差异是否足够。

最终选择最高分 mask：

[
M_i^{*}=\arg\max_{M_i^{(k)}\in \mathcal{M}_i}S(M_i^{(k)}),
]

或者用加权投票融合：

[
\hat{M}*i(p)=
\mathbb{1}\left[
\sum*{k=1}^{K} w_k M_i^{(k)}(p) \geq \tau
\right].
]

这就是你对 SAM3 的第一处“改进”：**从单 prompt SAM3 变成农业重建可靠性驱动的 multi-prompt SAM3 adapter。**

---

### 2. Positive-Negative Structural Prompting：植物结构平衡正负提示

IPENS 的“正-负-正”提示策略是为了解决 SAM2 单中心点提示时正负样本不平衡的问题。你可以借鉴这个思想，但不能完全照搬，因为你的目标不是谷粒级多目标分割，而是整株盆栽 foreground reconstruction。你可以改成：

**结构平衡正负提示策略，Structural Positive-Negative Prompting，SPNP。**

做法是先用 SAM3 text prompt 得到一个初始 mask，然后从这个 mask 自动生成点提示：

正样本点不要只取中心点，而是从植物 mask 的距离变换图中取多个高响应点：

[
P^+ = \operatorname{TopK}\left(D(M_i)\right),
]

其中 (D(M_i)) 是 mask 内部到边界的距离。这样正点会落在叶片主体、茎叶连接区域、冠层主体区域，而不是只有一个中心点。

负样本点从 mask 外圈生成：

[
P^- = \operatorname{Sample}\left(\operatorname{Dilate}(M_i,r)-M_i\right).
]

如果花盆区域明显，可以额外在植物下方或 pot candidate region 放负点，明确告诉 SAM3/SAM2 refinement：花盆不是 plant foreground。

最终用：

[
\mathcal{P}_i={P_i^+, P_i^-}
]

作为第二轮 prompt refinement。这个模块可以解决你的核心问题：**花盆、土壤、支撑物、桌面和绿色背景容易被 SAM3 纳入前景。**

注意，如果 SAM3 当前接口主要支持 text/concept prompt，不方便直接用点提示，你可以把这一模块写成 **SAM-family prompt refinement**，实现时用 SAM2/SAM 或支持 point prompt 的接口做二次细化。不要硬写“修改了 SAM3 内部结构”，否则审稿人会追问源码和训练细节。

---

### 3. Boundary-Preserving Residual Mask Repair：保细结构的残差掩膜修复

你现在 FSAM3 已经有 morphological closing、8-connected component analysis、小连通域过滤和 PCA main-foreground refinement。 但特刊需要看出你对 VFM 输出做了农业任务适配，所以建议把这部分升级为一个明确算法，而不是一句话后处理。

IPENS 用闭运算和开运算修复残缺与噪声。你可以改成更适合盆栽叶片的版本：

**先闭运算修补孔洞，再开运算去除孤立噪声，但加入 thin-structure preservation，避免把叶柄、细叶尖、窄叶片腐蚀掉。**

具体流程：

1. 对原始 mask 做 closing，修复孔洞和断裂；
2. 对结果做 opening，去除孤立噪声；
3. 计算骨架或细结构区域；
4. 对被 opening 删除但与主植物连通、且位于叶片边缘附近的细长区域进行恢复。

可以写成：

[
M_{\text{close}}=(M\oplus B_c)\ominus B_c,
]

[
M_{\text{open}}=(M_{\text{close}}\ominus B_o)\oplus B_o,
]

[
M_{\text{repair}}=M_{\text{open}}\cup R_{\text{thin}},
]

其中 (R_{\text{thin}}) 表示被常规形态学操作删除、但满足细长结构和主连通域邻接条件的区域。

这比普通 morphological post-processing 更像一个植物表型任务里的“VFM residual correction”。

---

### 4. Reconstruction-Aware Re-Prompt Frame Detection：重建感知的重提示帧检测

IPENS 用 SSIM 找目标重新出现的后视帧。你的盆栽植物不是谷粒级目标消失，但会出现另一个问题：**随着相机绕行或转台旋转，某些视角下叶片重叠、花盆遮挡、强反光、绿色背景相似，会导致 SAM3 mask 突然变差。**

所以你可以提出一个更适合 ForeSplat 的版本：

**不是检测“目标重新出现”，而是检测“mask reliability drop frame”。**

检测指标可以包括：

[
D_i =
\alpha(1-\operatorname{IoU}(M_i,M_{i-1}))
+\beta |\operatorname{Area}(M_i)-\operatorname{Area}(M_{i-1})|
+\gamma(1-\operatorname{SSIM}(I_i,I_{i-1}))
+\delta E_{\text{edge}}.
]

当 (D_i>\tau_d) 时，判定该帧为 prompt-sensitive frame，需要重新运行 prompt ensemble 或结构化正负提示。

更进一步，你可以加入重建反馈：

COLMAP 后把 sparse tracks 投影回图像，如果大量 foreground sparse points 落在 mask 外，说明 mask 过小；如果大量 mask 区域没有任何几何支持，说明 mask 可能包含背景。这个指标非常适合 ForeSplat，因为你的后续 2DGS 本身依赖 mask-defined foreground sparse-track initialization。ForeSplat 当前已经用 multiview mask consistency 过滤 sparse points，这正好可以反向用来评估 mask 是否可靠。

可以定义：

[
G_i=
\frac{1}{|X_i|}
\sum_{X_j\in X_i} M_i(\pi_i(X_j)).
]

如果 (G_i) 很低，说明该帧的 mask 与多视角几何不一致，应进入 re-prompt 或降低 view weight。

这就是你比 IPENS 更有新意的地方：**IPENS 是 SSIM 找补提示帧；ForeSplat 可以是 SSIM + mask stability + SfM/2DGS geometry feedback 找重建敏感帧。**

---


---

