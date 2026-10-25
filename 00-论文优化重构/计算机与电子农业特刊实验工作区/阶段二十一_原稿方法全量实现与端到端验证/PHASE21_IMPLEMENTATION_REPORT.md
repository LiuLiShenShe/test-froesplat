# Phase 21 — 原稿方法全量实现与端到端验证：实现报告（进行中）

本文件记录 Phase 21 的真实代码改动、验证证据与当前状态。状态口径：

- **VERIFIED**：有真实数据跑出的结果文件（日志/指标/哈希）支撑，可用命令复现。
- **IMPLEMENTED**：代码已实现并通过单元/契约测试，但尚未在真实数据上产生结果。
- **PARTIAL**：部分路径已实现或只在受限数据上验证。
- **BLOCKED**：因数据/依赖缺失无法执行，已记录原因，且**没有**用基线替代。

---

## 1. 代码改动清单

### 1.1 RAP-FSAM3（`数据管理/07-运行脚本与超参/S20-.../脚本/生成RAP-FSAM3掩膜.py`）

| 机制 | 改动 | 证据 |
|---|---|---|
| Eq.(1) FFT 帧筛选 | `02-FFT/blur_filter.py` 新增 `radial_high_pass_weight()` / `fft_blur_score_paper()`（径向高通 W_H、log(1+|F|)、L1 归一化）；`--fft_formula {legacy,paper,both}`；保留 legacy 路径 | `metrics/fft_formula_kqz.csv`（KongQueZhuYu 210 帧，legacy/paper 逐帧对照，33 帧判定不同） |
| SG-MPS | `semantic_gate_scores()` 输出 φ=[q_plant,q_area,q_comp,q_edge,q_contrast,q_temp]、ψ=[q_table,q_bg,q_neighbour,q_outlier] 与 `c=σ(w_φᵀφ−w_ψᵀψ−b)`；`apply_semantic_gate_softmax()` 计算 `a=softmax(c/τ_s)`；`semantic_gate_prompt()` 实现 `k*=argmax_k c`；gated 模式下 target/pot/side 参考区域改为图像内生（ExG 绿色分量 + 版面分带），不再用候选并集自证 | `metrics/sg_mps_ablation_summary.csv`、`metrics/sg_mps_ablation_per_frame.csv` |
| CDR/R1 | `score_select`、`weighted_fusion`、A1/A6/A7 变体选择、`per_instance` 默认路径全部改走 `select_best_policy/primary_score`；5 处 `getattr(..., "ranking_policy", "r1")` 口径一致 | 历史回放逐位一致（见 §2.1） |
| SB-FBPR | 沿用既有实现（正样本距离变换 TopK + 负样本环/桌面/背景/邻株 NMS），本轮未改动 | 回放运行覆盖 |
| GCFC | 新增 `points_to_support_field()`、`boundary_band()`；后端 `colmap_tracks_gaussian` 实现 `H_i`、`G_i=<H_i,M_i>/(‖H_i‖₁+ε)`、边界带 B_i 内正修正、`G_i<τ` 支撑门、负删除强制关闭；报告新增 `支撑场比例G` | `tests/test_phase21_gcfc.py`（7 项） |
| 选择/评分可追溯 | `语义门控评分.csv` 新增 `门控置信度`、`门控softmax`、`phi/psi 描述子`；每帧日志新增 `gate=`、`argmax_prompt=`、`a_max=` | 运行日志 |

**重要修正**：`semantic_gate_bonus` 这一“加性安全分”已删除。原稿定义的是
`k*=argmax_k a_i^(k)`，而 softmax 单调，等价于 `argmax_k c_i^(k)`；加性 bonus
既非原稿机制、又在实测量级下不可能改变选择。现在门控只负责**选提示词**，
同提示词内的实例仍由 R1 决策，二者不互相覆盖，也不重复计入 `q_contrast`。

### 1.2 Foreground-constrained 2DGS（`2d-gaussian-splatting-m2m3-floor40`）

| 机制 | 实现 | 证据 |
|---|---|---|
| 前景轨迹初始化 | `Scene` 前景投票 + `--init_pcd_min_observations`/`--init_pcd_foreground_threshold` | smoke 运行：23259 点保留 |
| 前景 RGB 监督 | `rgb_reconstruction_loss()` 仅在前景 stable 区域加权，边界忽略 2px | smoke：`Loss=0.114`、`mask=0.01395`、`bg=0.00430` |
| Alpha L1+Dice / 背景透明度抑制 | `mask_loss_terms()`（`l1_dice`）| 同上 |
| 视角质量软加权 | `compute_view_quality_weights()`：掩膜覆盖率 + Sobel 边界锐度 + 前景/背景 RGB 对比度 → z-score → `q_i=q_min+(q_max-q_min)σ((s_i−μ)/σ)` | smoke 日志：186 视角，range 0.643–0.995 |
| 掩膜同一性闭环 | `write_mask_provenance()` / `verify_mask_provenance()`；`--mask_provenance_required` 缺掩膜即拒绝训练 | smoke：`mask_provenance.json` 186 cams / 0 missing |

### 1.3 M2M3-GO（`utils/m2m3_go.py` + `train.py`）

实现 `S_m2g`、`H_i=M_i(1−A_i)`、`S_g2m`、`B^∂`、`U_j=σ(…)`、阈值化
keep/densify/prune（含 `N_removed/N_all<r_max`）、局部容量块
`N_k^{min}=ρ_floor·N_k^{ref}`、缺陷反投影补点。入口 `run_m2m3_go_update()`，
模式 `off|prune|full`。单测见 `tests/test_phase21_m2m3_go.py`（11 项）。

真实数据上补齐的三个实现缺陷（均为“跑通才发现”，已修复并加回归断言）：

1. **densify 属性形状（f_rest）**：本 fork 的 `GaussianModel` 存储
   `_features_rest = features[:,:,1:].transpose(1,2)` → `(N,(d+1)²−1,3)`，与上游
   3DGS 的 `(N,1,3·((d+1)²−1))` 不同。旧的补点张量导致
   `cat_tensors_to_optimizer` 报 `Expected size 15 but got size 1`。
2. **densify 属性形状（scaling）**：2DGS 面元只有两个面内尺度轴，`_scaling`
   是 `(N,2)`；旧实现给 `(N,3)`，报 `expected (2,) but got (3,)`。
3. **`sh_degree` 来源**：补点张量的 SH 宽度必须与模型自身一致，改用
   `gaussians.max_sh_degree`（原先误引用不在作用域内的 `dataset`）。
   现在 `run_m2m3_go_update()` 在拼接前逐项断言 attr 形状与模型参数一致，
   形状不符立即抛错，不会静默产生错误容量。

**阈值标定（重要，非调参掩盖）**：`U_j` 是线性得分的 sigmoid，其绝对尺度依赖
视角数与不透明度分布——186 视角时 `w5·log(1+|V|)` 一项就给几乎每个 primitive
加上 >1.3 的 logit。用原稿默认的固定 `τ_p=0.15` 在真实数据上实测每轮只有
0–3 个候选（30k 训练共剪 14 个 / 163k），模块等于空转。因此新增
`--m2m3_go_tau_mode {fixed,quantile}` + `--m2m3_go_tau_quantile`：quantile 模式把
`τ_p` 取为当轮 utility 分布的下分位点，并在日志/报告中逐轮记录**解析出的**
`tau_p` 与来源。真实数据 400 iter 冒烟（`logs/smoke_quantile_prune.txt`）：

```
[ITER 100] pruned 213/23259 (candidates=233), floor_blocked=20,  tau_p=0.3439(quantile(0.0100))
[ITER 200] pruned 120/23046 (candidates=230), floor_blocked=110, tau_p=0.2946(quantile(0.0100))
[ITER 300] pruned 113/22926 (candidates=229), floor_blocked=116, tau_p=0.2875(quantile(0.0100))
[ITER 400] pruned 111/22813 (candidates=228), floor_blocked=117, tau_p=0.2893(quantile(0.0100))
```

`floor_blocked` 约为候选数的一半，说明 Floor40 确实在拦截局部稀疏区域的删除，
不是装饰性开关。固定 `τ_p=0.15` 的那次 30k 结果已归档为
`runs/m2m3_ablation/archive_WanNianQing1_C_R1_M2_fixed_threshold`，作为“阈值退化”
的证据保留，不计入消融表。

4. **densify 规则按原稿落为 primitive 级**：原稿写的是
   `Densify(g_j)=1[S_m2g>τ_m]·1[S_g2m>τ_h or R_j>τ_r]`，是**对已有 primitive 的判定**，
   而最初实现只做“欠覆盖像素反投影补点”。现在默认
   `--m2m3_go_densify_source primitive`：按上式选出 primitive（按 `S_g2m+R`
   排序、受 `m2m3_go_densify_budget` 限额），再用
   `split_attribute_tensors()` 在其自身平面内抖动 `0.5·scale` 生成子 primitive
   （保持父 primitive，不动其颜色/不透明度）；像素反投影路径保留为
   `--m2m3_go_densify_source pixel` 备选。单测：
   `test_t10_select_densify_primitives`、`test_t11_split_attribute_tensors`
   （后者用 `quat_to_rotation_matrix()`，因为 `build_rotation()` 硬编码 cuda，
   在 CPU 单测里不可用）。

5. **densify 与 prune 同轮的索引错位（M4 跑之前的代码审查发现，已修复）**：
   `select_densify_primitives()` 返回的是**剪枝前**的 primitive 下标，而同一轮里
   `prune_points()` 已经把模型压紧；旧代码直接用这些旧下标去索引压紧后的
   `_xyz`，会把父 primitive 取错（并在 `max(index) >= N_after` 时抛
   `IndexError`）。新增纯函数 `remap_surviving_indices()`：丢掉本轮被剪掉的父点，
   其余下标减去其前方的删除数，与 `values[~remove_mask]` 的压紧结果逐项一致
   （`test_t12_remap_surviving_indices`）。该缺陷只在 `mode=full`（M4）触发，
   M2/M3 走 `mode=prune`，不受影响，已有结果无需重跑。

### 1.4 运行期导入修正（不改变任何数值语义）

`utils/image_utils.py` 与 `utils/point_utils.py` 在模块顶层 `import matplotlib.pyplot`，
而训练路径只用到 `image_utils.psnr` / `point_utils.depth_to_normal`，`plt` 在两个模块里
根本没有被调用。matplotlib 的导入会从磁盘读入数百个小文件；在本机磁盘被其它租户占满
时（见 §3 运行环境说明），这一步单次就要几十分钟。现改为：`colormap()` 内部惰性导入，
`point_utils` 直接删除无用导入。渲染与指标数值不受影响（`tests/` 全绿）。

### 1.5 系统级

- `scripts/run_pipeline.py`：`fft → rap → train → eval` 统一入口；manifest 记录
  git commit/branch/dirty、配置 SHA256、解释器、逐阶段命令、耗时、日志、返回码、
  消耗掩膜的 SHA256；上游失败/缺字段 → 下游 `BLOCKED`（绝不静默换基线）。
- `scripts/evaluate_foreground.py`：渲染验证相机并计算 `PSNR_fg / SSIM_fg /
  LPIPS_fg / outside_mask_non_black_ratio / leakage_energy_ratio`，逐视角记录
  `mask_sha256`；不再用整图 PSNR 评价黑底输出。
- `utils/foreground_metrics.py`：上述指标的纯函数实现（边界忽略带可配置）。
- **LPIPS 后端回退**：仓库自带的 `lpipsPyTorch/modules/lpips.py` 是**损坏文件**
  （提交进 git 的内容是 2026-02-25 的 TURN 日志乱码，1151 B，`import` 直接
  `SyntaxError`），因此此前所有 LPIPS 都是 `null`。现改为先试 `lpipsPyTorch`，
  失败回退到 venv 内已安装的 `lpips`（pip 0.1.4），并把实际使用的后端写进评价
  JSON 的 `lpips_backend` 字段。M0 实测：LPIPS_fg 0.1396，后端
  `lpips-pip(vgg)`。没有伪造、也没有跳过该指标。
- `scripts/summarize_m2m3_ablation.py`：只读汇总 M0–M4（指标、Gaussian 数、
  逐轮 prune/densify/floor_blocked、git commit、掩膜 SHA256）；缺结果标
  `MISSING_EVAL`。

---

## 2. 真实验证结果

### 2.1 历史回放一致性（VERIFIED）

`ranking_policy=base` 下重跑 Phase20C Easy21 V0（21 帧）：

```
identical 21   different 0
```

即 Phase 21 的所有改动在 legacy 语义下逐位一致，`gated_softmax` 与 M2M3-GO 均为
显式开关，不改变历史结果。

### 2.2 SG-MPS / R1 消融（KongQueZhuYu GT5，5 帧，冻结候选缓存）

| 配置 | 排名策略 | 门控 | F1 | mIoU |
|---|---|---|---|---|
| S0 | base | off | 0.9731 | 0.9478 |
| S1 | r1 | off | 0.9731 | 0.9478 |
| S2 | r1 | heuristic | 0.9731 | 0.9478 |
| S3 | r1 | gated τ=0.25 | 0.9731 | 0.9478 |
| S4 | r1 | gated τ=0.10 | 0.9731 | 0.9478 |
| S5 | r1 | gated τ=0.25 + contrast | 0.9731 | 0.9478 |

证据：`metrics/sg_mps_ablation_summary.csv`、`..._per_frame.csv`、
`metrics/sg_mps_ablation_rescore.json`（6 个配置 × 5 帧，逐帧记录预测/GT 的
SHA256），`logs/sg_mps_*.txt`。

**逐帧哈希核对**：同一帧在 6 个配置下的 `pred_sha256` 完全一致（每个帧
1 个不同哈希 / 6 个配置），即掩膜**逐位相同**，F1 相同不是抄写错误。
汇总表由 `scripts/rescore_sg_mps_ablation.py` 从已落盘的 6 组
`runs/sg_mps_ablation/*/最终掩膜/` 重新计算（此前一次单配置运行覆盖了
聚合 CSV，已补全；重算只读既有掩膜，不重跑 RAP、不补数）。

**结论（如实记录）**：在这 5 帧上所有配置选出同一候选（P2），F1 相同。门控
`c` 在 0.498–0.531 之间，argmax 与几何评分一致，因此未改变选择。这不是“门控无效”
的证明，而是“该数据上门控与几何排序一致、无冲突可消解”。门控能改变选择的情形由
`test_phase21_sgmps_r1.py::test_t6b` 覆盖（构造语义冲突候选时 `k*` 翻转）。
R1 在这 5 帧上没有额外的对比度误导可纠正（与 Phase20C 的 Easy21 结论一致）。

### 2.3 FFT 公式对照（VERIFIED，真实序列）

KongQueZhuYu 210 帧，`blur_percentile=25`：

| 公式 | 检出模糊 | 保留 | 强制保留 |
|---|---|---|---|
| legacy | 53 | 177 | 20 |
| paper (Eq.1) | 53 | 176 | 19 |

逐帧判定 33/210 不同（15.7%），说明原稿公式对筛选结果有实质影响，不是等价改写。
明细：`metrics/fft_formula_kqz.csv`。

### 2.4 端到端链路打通（VERIFIED，60 iter 冒烟）

`runs/smoke/m0_res8`：训练 → 掩膜 provenance（186/0 missing）→ 前景指标：

```
psnr_fg mean 17.69 (16.06–20.12)   ssim_fg 0.492
outside_mask_non_black_ratio 0.935  leakage_energy_ratio 0.119
27 个验证视角，mask_sha256 已记录
```

60 次迭代的绝对数值不具方法学意义，只证明“训练-记录-评价”链路可用；
30k 迭代的 M0–M4 对照在 `runs/m2m3_ablation`（见 §3）。

---

## 3. M0–M4 受控消融

配置：`scripts/run_m2m3_ablation.py`（冻结 E0 输入 `WanNianQing1/C_R1`，seed 0，
30k 迭代，resolution 2，`--port-base` 按驱动分配）。五档共享同一前景物体核心
（§2.4–§2.5 全部组件：前景初始化、前景 RGB 监督、alpha L1+Dice、背景透明度抑制、
视角质量软加权），只差 primitive 层：

| 档位 | 定义 |
|---|---|
| M0 | 前景物体 2DGS（§2.4–§2.5 全部组件），无 primitive 精化 |
| M1 | M0 + 掩膜引导剪枝（legacy topology + m2m3 容量控制） |
| M2 | M0 + M2M3-GO 双向评分 → utility 剪枝（无 Floor40、无补点；`τ_p` quantile 0.01） |
| M3 | M2 + Floor40 局部容量下限（ρ=0.40，6×6×6 块） |
| M4 | M3 + 缺陷驱动 densify（完整 M2M3-GO，budget 512） |

汇总脚本：`scripts/summarize_m2m3_ablation.py`（只读已落盘产物，缺失的档位报告
`MISSING_EVAL`，不补数）。结果见 `metrics/m2m3_ablation_summary.csv` / `.md`：
六档（C0、M0–M4）的 `masks_sha256` 全部为
`61be8b848d93273fd95e0929b12ae9933a826394c38c6991…`（同一套 213 个掩膜），
逐 job manifest 见 `runs/pipeline/<job>/run_manifest.json`（含 git commit
`d5db8371`、完整命令、耗时、返回码、消耗掩膜的组合 SHA256）。

**M0（已完成，VERIFIED）**：`runs/m2m3_ablation/WanNianQing1_C_R1_M0`

| 指标 | 值 |
|---|---|
| 训练 | returncode 0，4805.8 s，初始 23259 点 → 最终 164040 点 |
| 前景指标（27 个测试视角） | PSNR_fg 27.206 dB、SSIM_fg 0.8699、LPIPS_fg 0.1396（后端 `lpips-pip(vgg)`） |
| outside_mask_non_black_ratio | 0.0360（同一评价器下的无约束全场景对照**本阶段未重测**，不引用旧数字） |
| leakage_energy_ratio | 0.0263（同上） |
| 掩膜同一性 | 213 个掩膜，合并 SHA256 `61be8b84…4c9` |

**M1（已完成，VERIFIED）**：`runs/m2m3_ablation/WanNianQing1_C_R1_M1`
（legacy topology 剪枝 + m2m3 容量控制）

| 指标 | M0 | M1 | Δ |
|---|---|---|---|
| PSNR_fg | 27.206 | 27.098 | −0.108 |
| SSIM_fg | 0.8699 | 0.8685 | −0.0014 |
| LPIPS_fg | 0.1396 | 0.1402 | +0.0006（变差） |
| outside ratio | 0.0360 | 0.0360 | 0 |
| leakage | 0.0263 | 0.0267 | +0.0004（变差） |
| Gaussian 数 | 164,040 | 152,597 | −7.0% |

（容量以落盘 PLY 的 `element vertex` 为准。训练日志末行 `Points=152915` 是最后一次
剪枝**之前**的计数——最后一轮 `pruning removed 318/152915` 之后才写盘，
所以最终容量是 152,597。此前草稿里的 152,915 已按此更正。）

M1 用“事后式”容量剪枝换来了 7.0% 的 primitive 缩减，但四项质量指标全部轻微变差，
与原稿“post hoc / 容量式剪枝不能替代训练期前景物体优化”的判断一致；这也说明
M0→M1 不是有效改进，不能拿它当方法贡献。

**M3（已完成，VERIFIED）**：`runs/m2m3_ablation/WanNianQing1_C_R1_M3`
（M2M3-GO utility 剪枝 + Floor40 ρ=0.40、6³ 块、quantile τ_p）

| 指标 | M0 | M3 | Δ（配对，27 视角） |
|---|---|---|---|
| PSNR_fg | 27.2063 | 27.2023 | −0.0039 dB（sd 0.141，14/27 视角更好） |
| SSIM_fg | 0.86994 | 0.86992 | −0.000016（11/27） |
| LPIPS_fg | 0.13965 | 0.14010 | +0.00045（变差，7/27） |
| outside ratio | 0.03603 | 0.03592 | −0.00011（更好，15/27） |
| leakage | 0.02628 | 0.02651 | +0.00023（变差，12/27） |
| Gaussian 数 | 164,040 | 147,416 | **−10.1%** |

10 轮（iter 15000–24000，每 1000）共剪掉 15,589 个 primitive，解析出的
`τ_p` 在 0.4717–0.5109 之间（每轮 1% 分位），每轮候选全部被采纳（`max_remove_ratio
=0.03·N` 未触发）。27 个测试视角的 `mask_sha256` 与 M0 完全一致，配对差值可信。

**结论（如实记录）**：在**质量与 M0 统计上不可区分**的前提下（Δ PSNR_fg 的标准差
0.141 dB 远大于其均值 0.004 dB），M2M3-GO 把容量降低了 10.1%。这与原稿“primitive
级容量精化、不是决定性的前景分离手段”的定位一致；但它**没有**复现出正的
PSNR_fg 增益，论文里不能写成“提升前景重建质量”。

**Floor40 在本场景几乎不触发（如实记录）**：10 轮共 15,589 个候选中只有 **3** 个
被局部下限拦下（0.02%）。原因是 `τ_p` 取 1% 分位、`max_remove_ratio=0.03`，每轮
删除在空间上均匀摊开，单个 6³ 块每轮只掉约 1%，远达不到 `ρ=0.40` 的下限；
只有当某个局部块里有超过 60% 的 primitive 同时是候选时下限才会介入。
400 iter 冒烟（23k primitive 的早期模型）里 `floor_blocked` 曾达到候选数的一半，
说明代码路径是通的、只是在本场景的容量/分布下不活跃。因此本场景 M2 与 M3
预期给出同一组结果，这本身是“局部容量下限在均匀剪枝下不介入”的证据，
不能反过来当作“Floor40 有正向收益”。

**M4（已完成，VERIFIED）**：`runs/m2m3_ablation/WanNianQing1_C_R1_M4`
（M3 + primitive 级 densify，budget 512；**首个走 `mode=full` 的运行**）

| 指标 | M3 | M4 | Δ（配对，27 视角） |
|---|---|---|---|
| PSNR_fg | 27.2023 | 27.1830 | −0.0193 dB（sd 0.201，13/27） |
| SSIM_fg | 0.86992 | 0.86943 | −0.00049（7/27） |
| LPIPS_fg | 0.14010 | 0.14027 | +0.00017（变差，6/27） |
| outside ratio | 0.03592 | 0.03596 | +0.00004（15/27 更好） |
| leakage | 0.02651 | 0.02640 | −0.00011（更好，11/27） |
| Gaussian 数 | 147,416 | 147,125 | −0.2% |

10 轮共剪 15,553、增密 283。增密数逐轮衰减：`167 → 39 → 20 → … → 5 → 9`，与
“`R_j>τ_r` / `S_g2m>τ_h` 的欠覆盖随训练收敛而减少”的预期一致。M4 与 M3 的差异
同样落在噪声内（0.019 dB vs sd 0.201）。M4 是第一个真正执行
`select_densify_primitives → split_attribute_tensors → _append_gaussians` 的运行，
也是 §1.3 第 5 条索引错位修复后的验证：训练全程无 `IndexError`。容量账目自洽——
iter 15000 时 162,395 个 primitive，10 轮净 −15,553 + 283 = −15,270，
最终 162,395 − 15,270 = 147,125，与落盘 PLY 顶点数一致。

**M2（已完成，VERIFIED）**：`runs/m2m3_ablation/WanNianQing1_C_R1_M2`
（M3 去掉 Floor40：`--m2m3_go_local_floor_ratio 0`）

| 指标 | M0 | M2 | Δ（配对，27 视角） | M3（对照） |
|---|---|---|---|---|
| PSNR_fg | 27.2063 | 27.1872 | −0.0191 dB（sd 0.129，11/27） | 27.2023 |
| SSIM_fg | 0.86994 | 0.86930 | −0.00067（8/27） | 0.86992 |
| LPIPS_fg | 0.13965 | 0.14013 | +0.00048（变差，6/27） | 0.14010 |
| outside ratio | 0.03603 | 0.03587 | −0.00016（更好，18/27） | 0.03592 |
| leakage | 0.02628 | 0.02644 | +0.00016（14/27 更好） | 0.02651 |
| Gaussian 数 | 164,040 | 148,020 | −9.8% | 147,416 |

`M3 − M2 = +0.0152 dB`（sd 0.190，12/27 视角更好），与“Floor40 只拦截了 3 个候选”
一致：**两个档位的差异落在噪声内，Floor40 在本场景没有改变结果**。M2 每轮 τ_p 与
M3 几乎相同（0.4732/0.4809 vs 0.4717/0.4825），也印证两档走的是同一条剪枝轨迹。

**C0（已完成，VERIFIED）：无前景物体注入的对照**
`runs/m2m3_ablation/WanNianQing1_C_R1_C0`。同一输入、同一 213 视角/27 测试视角、
同一 30k 迭代与同一评价器，但**不加**前景初始化、前景 RGB 监督、掩膜/背景不透明度
损失与视角质量加权，即普通全场景 2DGS。

| 指标 | C0（无注入） | M0（注入） | Δ（配对，27 视角） |
|---|---|---|---|
| PSNR_fg | 25.1879 | 27.2063 | **+2.018 dB**（sd 0.862，26/27 视角更好） |
| SSIM_fg | 0.8574 | 0.8699 | +0.0125（27/27） |
| LPIPS_fg | 0.1501 | 0.1396 | **−0.0104**（27/27 更好） |
| outside ratio | 0.0114 | 0.0360 | +0.0246（0/27 更好，即**变差**） |
| leakage | 0.0051 | 0.0263 | +0.0211（0/27 更好，即**变差**） |
| Gaussian 数 | 132,514 | 164,040 | +23.7% |

**这里必须如实记录的转折**：冻结 E0 输入本身就是**黑底 RGB 预合成**（`images_rgba`，
Phase20E1 记录为 “Input treatment is explicit black-RGB precomposition only”），
因此“无注入”的 C0 渲染**本来就不会把背景重建出来**（outside 0.0114）。
本阶段**无法**用这份冻结输入复现“无约束全场景 → outside≈0.99”的对照：数据里根本没有
带背景的 RGB。据此：

- 注入带来的收益是**前景保真度**：PSNR_fg +2.02 dB、SSIM +0.0125、LPIPS −0.010，
  27/27 视角方向一致，不是噪声；
- 注入**没有**降低 outside/leakage，反而 27/27 视角都略微升高
  （+0.0246 / +0.0211，绝对量都很小）。原因是掩膜驱动损失把能量推向掩膜边界带
  （`mask_ignore_boundary_px=2` 内不约束），且容量多 23.7%。
  因此论文里不能写“注入降低了背景泄漏”，只能写“注入提升了前景保真度，
  代价是更多 primitive 与略高的掩膜外能量”。

**M3–M2 = +0.015 dB**（sd 0.190，12/27），说明 Floor40 在本场景没有改变结果。

旧 M2 的失败产物（10-24 23:33 起、00:45 被 OOM killer 杀掉、无 `point_cloud`）
已原样移到 `runs/m2m3_ablation/archive_WanNianQing1_C_R1_M2_oom_20261024`、
`logs/pipeline/archive_WanNianQing1_C_R1_M2_oom_20261024`（含旧 manifest 与驱动
拒绝覆盖的日志），**没有删除、也没有沿用半成品**；新 M2 于 04:59 串行重跑完成。



逐视角配对核对（27 个测试视角，逐视角 `mask_sha256` 完全一致，排除掩膜不同的干扰）：
`Δ PSNR_fg = M1 − M0` 均值 −0.108 dB、标准差 0.198，M1 只在 7/27 个视角更好。

**进度**：M0、M1、M2、M3、M4 全部完成并验证；另有 C0（无前景物体注入的全场景
2DGS 对照，同一输入/分辨率/迭代数/评价器）在跑，用于给论文 Table 6 提供可追溯的
“不做注入”基线。汇总见 `metrics/m2m3_ablation_summary.csv`。

**运行环境说明**：`/data` 是 3.6 TB 机械盘（`/dev/sda1`），本阶段期间被其它租户
持续读写压到近乎瘫痪。实测（`2026-10-25 03:30–03:45`）：

| 探针 | 结果 |
|---|---|
| `/proc/pressure/io` | `some avg300=96.9`、`full avg300=91.3`，已持续 3 h 以上 |
| `sda` 整盘吞吐 | 持续 ~23–25 MB/s（由不可见的其它用户进程占用） |
| 本进程 4 KB 随机读 | 4.97 s（NVMe 对照 0.05 ms） |
| 顺序读 7 MB 图 | 807 kB/s（buffered）、300 kB/s（O_DIRECT） |
| M3 主线程状态 | 长时间 `D`/`blk_mq_get_tag`，单轮 60 s 内 `read_bytes` 增量为 0 |

具体后果：M3（01:59 启动）**仅模块导入就用掉 1 h 48 min**（03:47 才打印
`Optimizing ...`），随后 213 视角加载约 9 s/视角（≈32 min）；而在磁盘正常时
（M0，10-24 20:32）导入 + 加载合计约 40 min。为减少重复读盘，已把训练路径上
无用的 `matplotlib` 顶层导入改为惰性（§1.4）。若被中断，manifest 保留 `FAILED`
状态与日志，不会产生伪造指标。


**一次真实的运行事故（如实记录）**：23:36 起的 M2 与 23:55 起的 M4 并行训练时，
主机内存耗尽（125 GB 中 108 GB 被其它会话占用，swap 8 GB 用满），内核 OOM
killer 在 00:45 杀掉进程。M2 的 manifest 记录 `train FAILED returncode=-9`、
`eval BLOCKED`，已达 30000 iter 但未落盘；M4 掉到 9.6 s/it（swap 抖动）。
处置：全部改为**串行单任务**运行，并给 `run_pipeline.py` 的 train/eval 子进程加
`-u`（无缓冲），这样即使进程被外部杀掉，已打印的逐轮记录也不会因 stdout 缓冲而丢失。
被中断的档位一律重跑，不沿用半成品目录。

---

## 4. 阻塞与限制（不掩盖）

1. **GCFC 真实数据执行**：需要与掩膜帧匹配的 COLMAP 观测。带候选缓存的
   KongQueZhuYu GT5 没有对应注册束（与 Phase20C 的 GT10 V2/V3 同一阻塞），
   因此 GCFC 目前为 IMPLEMENTED（单测覆盖语义），未在真实帧上产出指标。
2. **TEST32 未标注帧**：没有可追溯标注，不作为性能结论；仅用于流程回归。
3. **SG-MPS 在现有 GT 集上不改变选择**：见 §2.2，需要语义冲突样本才能体现差异，
   当前仅单元测试构造覆盖。
4. **M2M3-GO 真实消融**：M0/M1/M3/M4 已完成并验证，M2 正在重跑；Floor40 在本场景
   15,589 个候选中只拦截 3 个（均匀剪枝下局部下限不介入），因此“本地容量保护能带来
   正向结构收益”在本数据上**没有被证明**，只能报告为“未触发”。要检验该机制需要
   更稀疏的局部区域或更激进的剪枝率，属于后续实验，不能写成本文的已验证结论。
5. **整图 PSNR 不可用于黑底输出**：Phase20E1 的 B/C 组 PSNR≈5 属评价口径问题，
   Phase 21 起改用前景指标（§1.5）。

### 4.6 历史阶段测试的当前状态（如实记录，未改动这些测试）

Phase 21 结束后把历史测试全部重跑了一遍。以下失败与本轮代码改动的关系已逐条定位，
**没有为了让测试变绿而删改历史断言**：

| 阶段 | 结果 | 根因（已核实） |
|---|---|---|
| 阶段十二 / 十八 | 22 / 8 全过 | — |
| 阶段二十C_FIX | 33 全过 | — |
| 阶段二十C_R1 | 6 全过 | — |
| 阶段二十D_独立GT | 34 全过 | — |
| 阶段二十E1 / E2 / F | 13 / 16 / 12 全过 | — |
| 阶段二十B | 7 过 1 失败 | `test_no_production_files_changed` 断言“除本阶段目录外 git 工作区全净”。当前 40 项改动包含 Phase 21 的合法改动（RAP 脚本、trainer、阶段二十一目录）以及**本轮开始前就存在的** uv 迁移文件（`pyproject.toml`/`uv.lock`/`.python-version`、`UV_MIGRATION.md` 等）。该断言是跨阶段不变量，Phase 21 起必然不成立；属结构性过期，不是回归 |
| 阶段二十D_固定轮廓 | 41 过 2 失败 | 只有 `hd95_px` 偏离，且偏差为 **1 个 float32 ULP**（如 1563.328857 vs 1563.328735）。该指标走 `阶段二十B/scripts/segmentation.py` → `03-扩展实验后续执行/.../hd95()`，这两个文件 git 干净、且**不经过**本轮修改的 RAP 脚本；漂移来自 numpy 2.2.6 / scipy 1.17.1 的 `np.percentile` 实现变化（记录 CSV 由旧版本生成）。测试容差 `abs=1e-12` 无法跨库版本成立 |
| 阶段二十E0 | 收集报错 | 默认解释器缺 `pycolmap`。用 `instant-ngp/venv` 的 site-packages 作 `PYTHONPATH` 后 34 项可正常收集（收集本身 124 s，机械盘）。完整套件在 600 s 内未跑完，未据此下任何结论 |
| 阶段二十E3 | 4 过 3 失败 | 2 项 `ModuleNotFoundError: open3d`（默认解释器无此包；`2d-gaussian-splatting-main/venv` 有 0.19.0，但在机械盘上 import 超时）；1 项 `test_disabled_training_ast_identical` 断言“分支 trainer 与基线 trainer 除 capacity 块外 AST 相同”——Phase 21 在 `training()` 中加入了显式开关控制的前景/M2M3 代码，该不变量按定义已过期。原意（关闭新机制后训练回路等价）由新增的开关测试覆盖：`variables` 全为 `off/none` 时不进入任何新分支，`test_t6b_tau_p_resolution_modes` 覆盖阈值退化为固定值的路径 |
| 阶段二十G0 | 0 项 | 该阶段无测试文件 |

Phase 21 自身：`tests/` 54 项全过（见 §5）。

---

## 5. 复现命令

见 `REPRODUCTION.md`。

---

## 6. 论文正文同步（Table 6 / §3.4 / 摘要 / 结论）

工作稿：`计算机与电子农业特刊实验工作区/ForeSplat__.../ForeSplat.md`。

| 位置 | 改动 |
|---|---|
| 摘要 | 删掉 “outside 0.9908→0.0294、leakage 1.2201→0.0190” 的不可追溯表述，改为本轮实测：注入把前景 PSNR 25.19→27.21 dB、SSIM 0.857→0.870、LPIPS 0.150→0.140；primitive 控制器在质量不变的前提下减 7–10% 容量 |
| §3.4 正文 | 整段重写为 C0/M0–M4 的受控消融叙述；明确冻结输入是黑底 RGB 预合成、因此本表比较的是前景保真度与容量，不是背景泄漏；如实写出注入使 outside/leakage 各升高 0.025/0.021 |
| Table 6 | 9 行旧表整体替换为 C0 + M0–M4 六行实测（PSNR_fg/SSIM_fg/LPIPS_fg/outside/leakage/Gaussian 数 + 控制器定义） |
| Table 6 注 | 新增机制说明与 Floor40 只拦截 3/15,589 的事实；另附一段 **[Phase21 注记 · 投稿前删除]**，逐条列出被删除的旧行及其不可追溯的判定依据 |
| §3.4 第二段 | 改为“移除 M2M3-GO 只回收容量、质量在噪声内；M1 事后剪枝质量代价略大（−0.108 dB）”；删掉 “modest improvement” |
| Fig. 7 说明 | 不再声称“注入比输入掩膜/事后剪枝更能抑制背景”，改为注入 vs 无注入的前景保真度差异 |
| §4 讨论 | 删掉 “opacity-only 约束失败” 与 “post hoc 无法替代训练期优化” 的无产物支撑表述，改为本轮可验证的对比（C0 vs M0 = 2.02 dB；M1–M4 的变化 ≤0.11 dB） |
| 结论 | “注入是关键机制” 由 “抑制背景学习” 改为 “前景保真度（+2.02 dB）”；M2M3-GO 定位为容量控制器 |

**未改动、需要用户决策的部分**：

1. `论文2026-05-28/ForeSplat_en.md`、`ForeSplat_zh.md`（2026-08-27 快照）仍含旧 Table 6
   数字，本阶段**没有**动它们（历史快照，避免静默覆盖）。若它们才是投稿副本，
   需要把上面的改动同步过去。
2. Table 5（RAP-FSAM3 的 F1 0.9706 / mIoU 0.9641 / HD95 183.42 / Boundary-F1 0.4643 /
   leakage 0.001739）与 Table 7（20 序列均值）**本阶段未重跑**，不作为本阶段的验证结论；
   §2.2 的 SG-MPS 消融是在 5 帧 GT 上做的，不能替代它们。
3. §3.4 标题 “VFM-Prior Injection … Is Necessary” 现由 C0→M0 的 +2.02 dB 支撑，
   但“必要性”只在这条冻结输入上成立；若审稿要求“带背景输入”的对照，
   需要另做一份保留背景的输入（当前冻结数据不支持），已写进 Table 6 注记。

