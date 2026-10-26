# 07 — M2M3-GO 接入与回归

生成时间：2026-10-25
状态：**已完成。** 参数核对、2×2 回归（四档 COMPLETE）、TSDF 网格提取均已执行；结论部分为负（见 §3.1、§6）。

---

## 1. 论文公式 ↔ 代码核对

M2M3-GO 定义见原稿 §2.7，全部 8 个公式在 `2d-gaussian-splatting-m2m3-floor40/train.py` 的 `run_m2m3_go_update()` 中有对应实现。

| 论文 | 代码参数（默认值） | 说明 |
|---|---|---|
| $S^{\text{m2g}}_j=\dfrac{\sum q_i\eta_{ij}\frac1{|E_{ij}|}\sum_{p\in E_{ij}}M_i(p)}{\sum q_i\eta_{ij}+\varepsilon}$ | `--m2m3_go_mask_weight 3.0` | 掩膜→模型支持分 |
| $H_i(p)=M_i(p)(1-A_i(p))$ | 渲染 alpha 直接参与 | 前景覆盖赤字 |
| $S^{\text{g2m}}_j$ | `--m2m3_go_opacity_weight 1.0` | 模型→掩膜方向 |
| $B^\partial_j$，$\sigma_b$ | `--m2m3_go_boundary_sigma 6.0` | 边界支撑项 |
| $U_j=\sigma(w_1S^{\text{m2g}}+w_2S^{\text{g2m}}+w_3B^\partial+w_4\log(\alpha+\varepsilon)+w_5\log(1+|\mathcal V_j|)+w_6R_j+b)$ | `--m2m3_go_w1 1.0 --m2m3_go_w2 0.5 --m2m3_go_w3 0.5 --m2m3_go_w4 0.5 --m2m3_go_w5 0.25 --m2m3_go_w6 0.5 --m2m3_go_bias 0.0` | 六项权重与论文一一对应 |
| $\operatorname{Prune}(g_j)=\mathbf 1[U_j<\tau_p]\mathbf 1[N_{rm}/N_{all}<r_{max}]$ | `--m2m3_go_max_remove_ratio 0.03` | 剪枝上限 |
| $\operatorname{Densify}(g_j)=\mathbf 1[S^{\text{m2g}}>\tau_m]\mathbf 1[S^{\text{g2m}}>\tau_h\lor R_j>\tau_r]$ | `--m2m3_go_tau_m 0.65 --m2m3_go_tau_h 0.35 --m2m3_go_tau_r 0.12` | 致密化三项阈值 |
| $\tau_p$ 的 fixed / quantile 双模式 | `--m2m3_go_tau_mode {fixed,quantile} --m2m3_go_tau_quantile 0.01 --m2m3_go_tau_p 0.15` | 论文 §2.7 明示固定 $\tau_p$ 会让控制器静默失效；**参数解析器默认 `fixed`，所有实验档位与 `run_downstream_2dgs.py` 都显式传 `quantile`** |
| Floor40：$N_k^{\min}=\rho_{floor}N_k^{ref}$ | `--m2m3_go_local_floor_ratio 0.40 --m2m3_go_local_blocks 6` | 局部容量下限 |
| $\operatorname{Prune}_{floor}$ 的门控 | 同上，逐块计数 | 与论文最后一段一致 |

**关于 $|\mathcal V_j|$ 项（必须如实说明）**：论文在 §2.7 自己承认 `--m2m3_go_tau_p 0.15` 这一固定值在 186 视角下"alone contributes more than 1.3 to the logit of almost every primitive, so a single fixed $\tau_p$ can silently disable the controller"。因此所有实验档位与 `run_downstream_2dgs.py` 都显式传 `--m2m3_go_tau_mode quantile`。这不是实现偷懒，是论文自己写明的设计。**注意**：`arguments/__init__.py` 的解析器默认仍是 `fixed`（见 `configs/m2m3_go_flags.json`），所以漏传这个参数就会静默退化成论文警告过的那种状态——`scripts/run_downstream_2dgs.py` 已把它写死在命令里，不依赖调用方记得。

---

## 2. 回归设计（本次实际采用的方案）

### 2.1 为什么不用 S21 的 `--group` 路径

S21 的 `scripts/run_m2m3_ablation.py` 设计成 `--group <G>`，其 `source_path` 与 `mask_dir` 都指向冻结的 E0 根
`/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1/<sample>/<group>/`。该根目录**只有 CaoMei2 / HongZhang / WanNianQing1**，
没有 KongQueZhuYu；而本次完整 RAP 掩膜只产出了 KQZ。若要在那里注册 `RAP_FULL`，就得往**冻结快照**里写新目录
（用户明令不得覆盖/污染冻结数据），或者复制一整套 COLMAP。两者都不合适。

因此本次改用**同一实现、同一配方、同一场景**的 2×2 设计，直接调用 `2d-gaussian-splatting-m2m3-floor40/train.py`，
由 `scripts/run_downstream_2dgs.py` 统一封装（该脚本已设置 `--m2m3_go_mode full --m2m3_go_tau_mode quantile
--m2m3_go_tau_quantile 0.01 --m2m3_go_save_report`，即 M4 等价配置）：

| # | tag | 掩膜来源 | M2M3-GO | 回答的问题 |
|---|---|---|---|---|
| 1 | `C_R1_baseline` | 冻结 C_R1（单提示词 P2） | off | 对照下限（已 COMPLETE） |
| 2 | `RAP_FULL` | 完整 RAP 210 帧 | off | **完整 RAP 掩膜进 3D 后是否更好** |
| 3 | `RAP_FULL_M2M3GO` | 完整 RAP 210 帧 | full | M2M3-GO 在完整 RAP 掩膜上的作用 |
| 4 | `C_R1_M2M3GO` | 冻结 C_R1 | full | 补齐 2×2，使"掩膜来源 × M2M3-GO"可解耦 |

这比 S21 的 M0–M4 阶梯少了两级（M1 旧剪枝代理、M2/M3 的 Floor40 拆分），**这一点必须如实声明**：
本次只做 M4 等价档（完整 M2M3-GO）与其 off 对照，用 2×2 换取"掩膜来源"这一维度的可辨识性。
S21 的 WanNianQing1 M0–M4 结果作为 M2M3-GO 内部阶梯的既有证据保留，**不被本次覆盖、也不被当作本次结果**。

---

## 3. 索引重映射与 Floor40 空间分块的既有保证

阶段二十一测试（`tests/test_phase21_m2m3_go.py`，14 项）覆盖：

- `--m2m3_go_tau_mode` 的 `fixed` / `quantile` 两种解析；
- densify 选择与 split 属性张量；
- prune/densify 同轮**索引重映射**（同轮删除后索引仍然正确）；
- Floor40 局部块统计与容量下限。

54/54 全量测试在每次改动后均通过（本恢复工作期间复跑确认）。

### 3.1 RAP_FULL_M2M3GO 的真实运行记录（`metrics/m2m3_go_RAP_FULL.csv`）

| iteration | $\tau_p$（来源） | 剪枝 | 致密化 | utility 均值 | Floor40 拦截 |
|---|---|---|---|---|---|
| 6000 | 0.3450 (quantile 0.01) | 5 331 | 199 | 0.70 | 28 |
| 7000 | 0.5133 | 5 156 | 241 | — | 21 |
| 8000 | 0.5135 | 5 814 | 117 | — | 26 |
| 9000 | 0.3460 | 6 379 | 87 | — | 25 |
| 10000 | 0.5148 | 6 118 | 79 | — | 26 |
| 11000 | 0.5152 | 6 641 | 25 | — | 15 |
| 12000 | 0.3465 | 7 028 | 46 | — | 12 |
| 13000 | 0.5174 | 6 338 | 46 | — | 166 |
| 14000 | 0.5166 | 6 640 | 24 | 0.704 | 232 |
| 15000 | 0.4929 | 6 718 | 99 | — | 400 |
| 16000 | 0.4491 | 6 359 | 21 | — | 693 |
| 17000 | 0.4254 | 6 005 | 22 | — | 983 |
| 18000 | 0.4062 | 5 625 | 27 | — | 1 293 |
| **合计** | — | **80 152** | **1 033** | — | — |

⚠️ 该表是 13:40 抓的快照（写到 14k 迭代）。最终 run 结束时有 **19 次更新、累计剪枝 107 955、致密化 1 094、Floor40 最大拦截 2 618**，$\tau_p$ 范围 0.338–0.517。对照组 `C_R1_M2M3GO` 为 19 次更新、剪枝 115 825、致密化 1 105、最大拦截 874。

两点必须写进论文：

1. **quantile 模式确实生效**：每轮 $\tau_p$ 由 1% 分位数解析（0.338–0.517），不是固定的 0.15——这正是论文 §2.7 警告"固定 $\tau_p$ 会静默失效"的那一项。
2. **Floor40 的拦截数随训练单调上升**（28 → 1 293，最终 2 618），说明局部容量下限在后期成为**主要约束**，而不是空转。这与"Floor40 在 16 万高斯时不 binding"的早期观察不同——本场景 65 万高斯、6×6×6 分块下它会咬住。论文若声称 Floor40 只是防御性设计，需要给出这个反例。

⚠️ 但**这些机制执行得再忠实，也没有换来指标提升**：`RAP_FULL_M2M3GO` 的 PSNR_fg 24.610 < `RAP_FULL` 24.734，SSIM 0.8135 < 0.8244（见 `06` §6.1）。

---

## 4. 执行约束（来自阶段二十一 REPRODUCTION.md，仍有效）

| 约束 | 值 |
|---|---|
| 解释器 | `2d-gaussian-splatting-m2m3-floor40/venv/bin/python` |
| 磁盘 | `/data` 机械盘，曾被压到 ~1 MB/s；213 视角加载 8–70 min |
| 内存 | 125 GB 被他用占满时**并行两档会触发内核 OOM killer**，必须串行 |
| 端口 | 两档并行必须不同 `--port-base`（避免 network_gui 6009 冲突）；但内存不允许并行 |
| 迭代 | 块设备健康时 30k 迭代约 25 min/档位 |

---

## 5. 执行状态与命令

```bash
PY=/home/test/.venvs/sam3/bin/python
S=00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery

# 2×2 设计（串行执行：两档并行会触发内核 OOM killer）
# 1–4 ✅ 全部 COMPLETE
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir 00-论文优化重构/数据管理/03-分割Mask/04-fsam3_masks/KongQueZhuYu \
    --tag C_R1_baseline --iterations 30000 --resolution 2
# 2–4（RAP 掩膜 210/210 已就绪）
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir $S/runs/KongQueZhuYu_RAP_FULL_210 \
    --tag RAP_FULL --iterations 30000 --resolution 2
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir $S/runs/KongQueZhuYu_RAP_FULL_210 \
    --tag RAP_FULL_M2M3GO --iterations 30000 --resolution 2 --m2m3-go --port 6050
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir 00-论文优化重构/数据管理/03-分割Mask/04-fsam3_masks/KongQueZhuYu \
    --tag C_R1_M2M3GO --iterations 30000 --resolution 2 --m2m3-go --port 6050
```

⚠️ `--m2m3-go` 必须显式传 `--m2m3_go_tau_mode quantile`：参数解析器的**默认是 `fixed`**，而论文 §2.7 与
S21 的实验档位都要求 quantile（固定 $\tau_p=0.15$ 会让控制器静默失效）。`run_downstream_2dgs.py` 已写死该参数，
不依赖调用方记得。

### 5.1 S21 M0–M4 阶梯（历史证据，保留不复用）

```bash
cd 00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十一_原稿方法全量实现与端到端验证
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_m2m3_ablation.py \
  --sample WanNianQing1 --group C_R1 --variants M0,M1,M2,M3,M4 --run-name m2m3_ablation
```

该路径只覆盖 `DATA/<sample>/<group>` 下已存在的冻结数据集（CaoMei2/HongZhang/WanNianQing1），**无法指向本次的 RAP 输出**，故本次不走它。KQZ 未在冻结根中，也没有必要为了它去写冻结快照。

---

## 6. 网格提取与表型测量（论文 §2.8）

### 6.1 已执行：四档 TSDF 网格

用 `scripts/run_mesh_phenotype.py` 对 2×2 的四个 30k 模型逐一执行 `render.py --skip_train --skip_test`，
网格协议冻结为 S19 的 standard TSDF（`--voxel_size 0.02 --depth_trunc 6.0 --sdf_trunc 0.08
--num_cluster 20 --mesh_res 256`）。产物归档在 `runs/mesh/<tag>/`（`mesh_raw.ply`、`mesh_post.ply`、
`mesh_metrics.json`、`mesh_record.json`、stdout/stderr 日志），**不覆盖模型目录里的中间文件**。

| tag | post 顶点 | post 三角 | 连通域 | 最大域占比 | 小碎块(≤1%) | 网格耗时 |
|---|---|---|---|---|---|---|
| `C_R1_baseline` | 170 040 | 330 076 | 14 | 0.9443 | 12 | 106.1 s |
| `C_R1_M2M3GO` | 168 920 | 327 964 | **5** | **0.9931** | **4** | 73.9 s |
| `RAP_FULL` | 175 350 | 337 921 | 12 | 0.9171 | 8 | 81.5 s |
| `RAP_FULL_M2M3GO` | 181 622 | 349 416 | 8 | 0.9284 | 5 | 78.4 s |

### 6.2 探索性几何量（`metrics/mesh_phenotype.csv`）

沿**相机姿态估计的重力上轴**（逐相机 up 的均值）与两个水平正交轴测量最大连通域：

| tag | 高度 (u) | 冠幅 a (u) | 冠幅 b (u) | 最大冠幅 (u) | PCA 三轴 (u) | 相对基线 Δ高度 / Δ冠幅 |
|---|---|---|---|---|---|---|
| `C_R1_baseline` | 3.5420 | 3.0664 | 3.1256 | 3.1256 | 3.772 / 3.475 / 3.042 | — |
| `C_R1_M2M3GO` | 3.5411 | 3.0660 | 3.1218 | 3.1218 | 3.742 / 3.503 / 3.181 | −0.03% / −0.1% |
| `RAP_FULL` | 3.5598 | 3.0696 | 3.1418 | 3.1418 | 3.773 / 3.469 / 3.101 | +0.50% / +0.5% |
| `RAP_FULL_M2M3GO` | 3.5611 | 3.0708 | 3.1393 | 3.1393 | 3.708 / 3.291 / 3.161 | +0.54% / +0.4% |

图：`figures/fig6_mesh_phenotype_geometry.png`。

三条结论：

1. **粗几何对掩膜来源不敏感**：四档高度差 ≤0.02 u（≤0.54%）、冠幅差 ≤0.02 u。掩膜来源造成的是
   **局部泄漏与纹理/σ 层面的差异**（PSNR_fg −0.285 dB、leakage 10×），而不是整体植株尺寸的漂移。
   因此"完整 RAP 掩膜让 3D 变差"不能被解释成尺度错乱。
2. **M2M3-GO 的可见效果是结构清理而不是精度提升**：在 C_R1 掩膜上它把连通域 14→5、最大域占比
   0.9443→0.9931、碎块 12→4，同时顶点只少 0.7%。这与 §3.1 的"剪枝 115 825"一致，也解释了为什么
   PSNR_fg 只有 +0.011 dB：它剪掉的是游离碎片，不是影响重建质量的主体。**这是本次对 M2M3-GO 唯一的
   正面证据，但它是几何结构指标，不是重建精度指标。**
3. 在 RAP 掩膜上，M2M3-GO 同样减少连通域（12→8）但顶点增加 3.6%，且 PSNR_fg 下降——它在坏掩膜上
   把预算花在了错误的前景上。

### 6.3 物理表型：`PHYSICAL_SCALE_NOT_AVAILABLE`（结果为空，不是失败）

沿用阶段二十F 的证据门（`阶段二十F_.../scripts/phenotype_core.py`）：**KongQueZhuYu 没有一段经过核验的
真实世界距离**把重建坐标系绑定到厘米，因此：

- `metrics/mesh_phenotype.csv` 的 `physical_scale_cm_per_unit` 为空、
  `physical_status = PHYSICAL_SCALE_NOT_AVAILABLE`、`physical_trait_claim = False`；
- 上表的"高度/冠幅"单位是 **COLMAP 场景单位，禁止写成 cm**；
- 网格是**含花盆的整体对象**（前景掩膜把盆一起包含），所以即使将来有尺度，这里的"高度"也只能叫
  "盆栽整体高度"，不能直接当论文的 `plant_height`；
- 叶片级性状（`leaf_length`、`leaf_width`）需要器官级身份与地标，当前链路不提供，状态
  `LEAF_ID_NOT_AVAILABLE`。

这不是本环节的遗漏，而是与阶段二十F 完全相同、且必须保持一致的证据标准。论文 §2.8 若声称表型精度，
必须同时给出尺度来源；当前**没有**，所以本次只能交付探索性几何。

---

## 7. 完成判据

1. 四个档位全部 `status == "COMPLETE"`（`runs/downstream/<tag>/run_record.json`）；
2. 每档 `mask_provenance.json` 的 `camera_count == 183`、`missing_count == 0`，且 183 条 SHA256 与对应 `mask_provenance_expected.json` 逐条一致（RAP 两档必须来自 `runs/KongQueZhuYu_RAP_FULL_210/最终掩膜`）；
3. 四档可并列成"掩膜来源 × M2M3-GO"表：`python scripts/build_downstream_2x2.py` → `metrics/downstream_2x2.csv`；
4. 网格 + 表型指标各出一份，且 mask provenance 可追溯到对应掩膜来源；
5. 结果（含失败）写入 `08_END_TO_END_VALIDATION.md`。

**当前状态**：`metrics/downstream_2x2.csv` 已生成，四档全部 `COMPLETE`（见 §5）。

---

## 8. 完成判据复核

| # | 判据 | 状态 |
|---|---|---|
| 1 | 四个档位全部 `status == "COMPLETE"` | ✅ 四档 `train_returncode=0`，2201/2153/2132/2166 s |
| 2 | 每档 `camera_count == 183`、`missing_count == 0`，183 条 SHA256 逐条一致 | ✅ 四档均一致 |
| 3 | 可并列成 2×2 | ✅ `metrics/downstream_2x2.csv` + `figures/fig5_downstream_2x2.png` |
| 4 | 网格 + 表型各一份，且 provenance 可追溯 | ✅ `runs/mesh/<tag>/` ×4 + `metrics/mesh_phenotype.csv`（每行带 `mask_manifest_sha256`）；物理表型按证据门留空 |
| 5 | 结果（含失败）写入 `08_END_TO_END_VALIDATION.md` | ✅ `08` §2.3 / §4 |

**结论：M2M3-GO 接入与回归已完成（含负面结论），下游闭环已贯通到 mesh。**
