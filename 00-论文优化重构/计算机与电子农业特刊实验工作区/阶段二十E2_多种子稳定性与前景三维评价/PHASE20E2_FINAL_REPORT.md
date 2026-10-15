# Phase20E-2 最终报告

```text
PHASE20E2 STATUS: PASS

ADDITIONAL_TRAININGS_COMPLETED: 8/8

SEEDS:
0, 1, 2

WAN_NIAN_QING_R1_3D:
INCONCLUSIVE

CAO_MEI_R1_3D:
MIXED

FOREGROUND_PROXY_EVALUATION:
COMPLETE

R1_3D_BENEFIT:
MIXED

R1_FINAL_ROLE:
ROBUST_SELECTION_ONLY

M2M3_NEXT_ACTION:
REWRITE_TO_IMPLEMENTATION

NEXT_PHASE:
PHASE20E3
```

本阶段真实完成 8 组新增训练，每组达到 30,000 iterations，并通过 checkpoint/PLY 重载。继承 8 个历史实体模型，HongZhang B/C 共用其中一个；最终表含 17 个 group/seed 行、368 个逐视图行、341 个独立模型-相机浮点渲染。两个变化场景各三种子，验证视角不是训练重复。

## A. Full raw RGB appearance fidelity

所有组使用同一 canonical undistorted raw RGB、冻结 heldout 相机、原色彩范围和浮点归一化、SSIM kernel 与 LPIPS 版本/权重、尺寸和黑背景。下表仅比较 B/C 的配对最终模型；A 组只有原 Seed 0，不对 A→B 的随机性作结论。

| Scene | Seed | B PSNR | C PSNR | ΔPSNR | ΔSSIM | ΔLPIPS |
| --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 0.000000 | 5.212587 | 5.214271 | 0.001684 | 0.000790 | -0.000728 |
| WanNianQing1 | 1.000000 | 5.213663 | 5.124970 | -0.088693 | -0.020463 | 0.011026 |
| WanNianQing1 | 2.000000 | 5.212740 | 5.213390 | 0.000650 | 0.000126 | -7.054099330194852e-05 |
| CaoMei2 | 0.000000 | 4.955827 | 4.930537 | -0.025290 | -0.004376 | 0.002335 |
| CaoMei2 | 1.000000 | 4.944296 | 4.919199 | -0.025097 | -0.003253 | 0.003308 |
| CaoMei2 | 2.000000 | 4.962600 | 4.948894 | -0.013706 | -0.002487 | 0.001770 |

## B. Fixed proxy foreground appearance fidelity

每视角统一使用原 R0_proxy mask；MSE 分母为 3×ROI 像素数。Composited SSIM/LPIPS 含共同黑背景及边界卷积影响，均为代理；R0 来源可能偏向 B。没有新增人工 GT。

| Scene | Seed | B ROI PSNR | C ROI PSNR | ΔROI PSNR | Δcomp SSIM | Δcomp LPIPS | Δabs color |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 0.000000 | 23.082850 | 23.410994 | 0.328145 | 0.000963 | -0.000931 | -0.001332 |
| WanNianQing1 | 1.000000 | 23.285643 | 22.200409 | -1.085234 | -0.021694 | 0.013286 | 0.027505 |
| WanNianQing1 | 2.000000 | 23.150321 | 23.171722 | 0.021401 | 0.000097 | -0.000155 | 0.000098 |
| CaoMei2 | 0.000000 | 19.999550 | 18.898690 | -1.100860 | -0.004531 | 0.002659 | 0.017231 |
| CaoMei2 | 1.000000 | 19.005154 | 19.694239 | 0.689084 | -0.003432 | 0.003829 | 0.012014 |
| CaoMei2 | 2.000000 | 20.093747 | 19.979360 | -0.114387 | -0.002690 | 0.002217 | 0.008807 |

## C. Background suppression

Outside-ROI RGB energy/alpha 与 missing-alpha 同时报告；外部 alpha 连通区域只是外观代理，不能全部叫 floating Gaussian。

| Scene | Seed | Δenergy | Δalpha | Δmissing alpha | Δcomponents |
| --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 0.000000 | 0.000030 | 0.015676 | -0.000169 | -1.518519 |
| WanNianQing1 | 1.000000 | -0.000133 | -0.022597 | -0.002341 | -0.888889 |
| WanNianQing1 | 2.000000 | 0.000018 | 0.025274 | -1.387756869748822e-05 | -0.074074 |
| CaoMei2 | 0.000000 | -7.224563887575641e-06 | 0.013911 | -7.037731376970312e-06 | 0.000000 |
| CaoMei2 | 1.000000 | -2.811408072440311e-06 | -0.006304 | 0.000050 | 0.928571 |
| CaoMei2 | 2.000000 | -6.511579028613447e-06 | -0.030016 | -0.000129 | 0.071429 |

## D. Descriptive geometric behavior

Gaussian count、scale/opacity 分布、alpha 支持的 depth、渲染与深度法向内部一致性均为描述。深度单位为 inherited COLMAP 任意单位，无独立几何 GT，不计算真值 Chamfer/F-score。A/B/C/D 不合并成排名。

| Scene | Seed | B count | C count | Δcount | Δdepth median | Δnormal proxy |
| --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 0.000000 | 128447.000000 | 133425.000000 | 4978.000000 | 0.000820 | 0.000934 |
| WanNianQing1 | 1.000000 | 129634.000000 | 133557.000000 | 3923.000000 | -0.178347 | -0.024174 |
| WanNianQing1 | 2.000000 | 131775.000000 | 131522.000000 | -253.000000 | 0.000043 | -0.000393 |
| CaoMei2 | 0.000000 | 130164.000000 | 143927.000000 | 13763.000000 | -0.212730 | -0.025326 |
| CaoMei2 | 1.000000 | 135054.000000 | 125603.000000 | -9451.000000 | -0.105746 | 0.003893 |
| CaoMei2 | 2.000000 | 134055.000000 | 129922.000000 | -4133.000000 | -0.010325 | -0.015803 |

## 多种子波动与证据范围

配对单位是每个 scene 内的一次 seed B/C 训练对。样本 SD 使用 ddof=1；无 p 值，无将视角当独立实验的显著性声明。所有 seed、负值及描述性离群标记保留。

| Scene | Endpoint | Paired mean | Paired SD | Min | Max | Directions | B SD | C SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | PSNR | -0.028786 | 0.051883 | -0.088693 | 0.001684 | +,-,+ | 0.000582 | 0.051306 |
| WanNianQing1 | foreground_PSNR | -0.245229 | 0.743457 | -1.085234 | 0.328145 | +,-,+ | 0.103271 | 0.641121 |
| CaoMei2 | PSNR | -0.021364 | 0.006633 | -0.025290 | -0.013706 | -,-,- | 0.009255 | 0.014985 |
| CaoMei2 | foreground_PSNR | -0.175387 | 0.896530 | -1.100860 | 0.689084 | -,+,- | 0.603149 | 0.560066 |

Seed 0 的现场初始 RNG 未保存。源码、环境、原 safe_state 路径与当前重放支持路径等价，但不能伪造历史 hash；严格统一包装三种子声明暂停。新增 Seed 1/2 的真实初始化与 30,000 次相机选择逐对相同，四组配对均通过现场证据。两新增 seed 的独立描述如下，不将 n=2 夸大为显著性。

| Scene | Endpoint | ΔS1 | ΔS2 | Mean Δ | SD Δ | B SD | C SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | PSNR | -0.088693 | 0.000650 | -0.044021 | 0.063175 | 0.000652 | 0.062523 |
| WanNianQing1 | foreground_PSNR | -1.085234 | 0.021401 | -0.531916 | 0.782509 | 0.095687 | 0.686822 |
| CaoMei2 | PSNR | -0.025097 | -0.013706 | -0.019401 | 0.008054 | 0.012943 | 0.020997 |
| CaoMei2 | foreground_PSNR | 0.689084 | -0.114387 | 0.287349 | 0.568140 | 0.769751 | 0.201612 |

## 运行与失败

新增训练总 wall-clock（包含各运行场景加载与原生训练记录边界）为 15792.6 秒。物理 GPU UUID 均为 `GPU-6836f890-98b2-5891-07ef-675c25199c65`，与历史一致。运行命令、环境、初始状态、checkpoint/PLY SHA256 和逐迭代曲线见 manifest 与 logs。原生 30,000 iterations 对应 29,999 次 optimizer.step，忠实继承原实现。

WanNianQing1/B_R0/Seed 1 的前两次阶段记录启动在 iteration=0、optimizer_steps=0 时因 NumPy 标量 JSON 序列化错误失败。失败脚本、日志和目录保留；事先计划未覆盖此种记录错误，查看新增优化结果前已另存 PREOPTIMIZATION_REMEDIATION.json 公开补充。该偏离不能隐藏成原计划内成功。修复只转录元数据类型，不改训练数学、seed、输入或 GPU；没有质量驱动重跑。

四个变化训练视图的原始候选与冻结 B/C RGBA 已只读核对：内部 RGB 与 canonical RGB 精确相同，掩膜外 RGB 精确为零，alpha 与冻结 Mask 一致。CaoMei2 raw0139 的注册图像实际为 crop_0124.png。输入可视化与直接查看记录见 CHANGED_TRAINING_INPUT_AUDIT.csv 和 INPUT_VISUAL_QA.md；不把训练视图选择差异当作 held-out 三维收益。

## 可视化与真实重算

figures 包含事先按相机顺序冻结的 first/middle/last 视角，以及明确标为探索性的旧 Seed 0 ROI 最好/最差/近零案例。每个视角展示 canonical raw RGB、B/C 三种子、真实 RGB 差值、alpha、depth 和 native normal。差值与 depth 共用显示范围，显示 PNG 不用于正式评价；索引保存真实 float 来源和 SHA。最终图像目视 QA 与刷新命令记录见 reports/FIGURE_VISUAL_QA.md 和 reports/FIGURE_REFRESH_PROGRESS.json。

全部 341 个实际浮点文件已由 recompute_metrics.py 重算，参考图与 ROI 原始/resize hash 逐视角一致。Seed 0 正式 RGB/alpha 直接继承原 float，depth/normal 为同冻结 PLY 新目录 replay；重放差异单列，未改旧指标。详见 reports/FLOAT_RENDER_METRIC_RECOMPUTATION.json。

## 判定与下一阶段

总体 `MIXED`；R1 当前定位 `ROBUST_SELECTION_ONLY`，不能写成额外三维增益已证明。逐项十二问、端点差异、波动比及结构代理解读见 R1_3D_ROBUSTNESS_DECISION.md。

M2M3 下一动作 `REWRITE_TO_IMPLEMENTATION`：按真实加权剪枝/全局 floor 描述，未实现的双向匹配另立任务。Mesh 工程参考已预先固定旧 B_R0/Seed 0，物理尺度和表型资料仍缺。本阶段没有修改/执行 M2M3、mesh 或论文正文。详见 M2M3_METHOD_CORRECTION_HANDOFF.md 和 NEXT_PHASE_EXECUTION_PLAN.md。

## 验收与恢复

| Gate | Requirement | Result |
| --- | --- | --- |
| G1 | Original Seed 0 and E0/E1 files unchanged | PASS |
| G2 | Eight actual additional trainings, with all failures disclosed | PASS |
| G3 | Seeds 1/2 change all actual RNG streams | PASS |
| G4 | Same-seed B/C initialization, full config, physical GPU comparable | PASS |
| G5 | All required models complete 30000 iterations | PASS |
| G6 | Checkpoints and Gaussian PLY reload | PASS |
| G7 | Identical frozen heldout cameras | PASS |
| G8 | Canonical full raw RGB common reference | PASS |
| G9 | One unchanged R0 proxy ROI for every group | PASS |
| G10 | All proxies labelled non-independent GT | PASS |
| G11 | Scene/training seed pair is the unit, not heldout views | PASS |
| G12 | No seed/checkpoint/figure cherry-picking | PASS |
| G13 | No extra A/HongZhang/unrelated training | PASS |
| G14 | R1/2DGS/M2M3 production code unchanged | PASS |
| G15 | Manual GT and paper unchanged | PASS |
| G16 | Metrics recompute from actual float renders | PASS |
| G17 | All negative results and failures retained | PASS |
| G18 | Concrete next M2M3/mesh execution decision | PASS |

首次自动终结在 `recompute_metrics.py` 因 HongZhang Seed 0 共享别名断言失败；原失败进度及日志保持原样，其 SHA256 为 `ff0b45399b59fe11341e938fa673d2c969e6f7843b1714d82642b701fb3e4ec6`。修复仅严格校验同一 checkpoint/render 的 B/C 别名，并对全部唯一浮点渲染重算；随后图表与 G1-G18 验收均通过。恢复命令、日志哈希和原始证据哈希见 `reports/FINALIZATION_RECOVERY.json`，未重新训练或渲染。

完整恢复路径、命令与大文件索引见 RECOVERY_AND_REPRODUCTION.md、reports/LARGE_ARTIFACT_MANIFEST.csv。大模型保留独立管理目录，不上传 Git、不覆盖 E1。所有生产源和历史文件已复核 SHA256；本阶段之外 tracked diff 必须为空。

来源：本仓库冻结输入、实际 GPU 训练/渲染、只读源码与 SHA256、float 重算报告。Context7 当前 PyTorch 文档已存档；博查检索 HTTP403 额度不足，外部搜索不可用，没有用训练数据填补实验结果。
