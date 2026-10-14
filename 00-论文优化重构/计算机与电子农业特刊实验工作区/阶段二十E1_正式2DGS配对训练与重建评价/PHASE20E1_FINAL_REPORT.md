# ForeSplat Phase20E-1 正式配对训练与共同参考评价

Baseline/source/config commit: `3e909f93`. 日期: 2026-10-14T10:51:30.216165+08:00.
GPU1 physical UUID `GPU-6836f890-98b2-5891-07ef-675c25199c65`.
初始 main 工作区干净；实际资产提交 SHA 见 DELIVERY_RECEIPT.md（提交后生成）。

## 真实结果

| sample | group | PSNR | SSIM | LPIPS | Gaussians | Time_seconds | VRAM_allocated_MiB | VRAM_reserved_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | A_raw_RGB_control_v1 | 26.121066 | 0.915390 | 0.349732 | 445086 | 3115.678788 | 1950.012207 | 25208.000000 |
| WanNianQing1 | B_R0 | 5.212587 | 0.260462 | 0.606881 | 128447 | 2486.149806 | 1187.294434 | 2518.000000 |
| WanNianQing1 | C_R1 | 5.214271 | 0.261252 | 0.606153 | 133425 | 2826.202454 | 1192.833008 | 3052.000000 |
| CaoMei2 | A_raw_RGB_control_v1 | 25.124810 | 0.893551 | 0.389774 | 321451 | 3546.770499 | 1728.974609 | 31876.000000 |
| CaoMei2 | B_R0 | 4.955827 | 0.108415 | 0.718155 | 130164 | 2250.226632 | 1173.285156 | 2054.000000 |
| CaoMei2 | C_R1 | 4.930537 | 0.104039 | 0.720490 | 143927 | 2192.681303 | 1337.769531 | 3678.000000 |
| HongZhang | A_raw_RGB_control_v1 | 24.555175 | 0.890713 | 0.354719 | 538440 | 3254.061911 | 2073.063965 | 19660.000000 |
| HongZhang | B_R0 | 5.376382 | 0.292100 | 0.588685 | 162458 | 2175.729039 | 1242.666016 | 4366.000000 |
| HongZhang | C_R1 | 5.376382 | 0.292100 | 0.588685 | 162458 | 2175.729039 | 1242.666016 | 4366.000000 |

| sample | comparison | PSNR | SSIM | LPIPS | Gaussians | primary_direction |
| --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | A_raw_RGB_control_v1->B_R0 | -20.908479 | -0.654928 | 0.257149 | -316639 | REGRESSED |
| WanNianQing1 | B_R0->C_R1 | 0.001684 | 0.000790 | -0.000728 | 4978 | IMPROVED |
| CaoMei2 | A_raw_RGB_control_v1->B_R0 | -20.168983 | -0.785136 | 0.328381 | -191287 | REGRESSED |
| CaoMei2 | B_R0->C_R1 | -0.025290 | -0.004376 | 0.002335 | 13763 | REGRESSED |
| HongZhang | A_raw_RGB_control_v1->B_R0 | -19.178793 | -0.598614 | 0.233966 | -375982 | REGRESSED |
| HongZhang | B_R0->C_R1 | 0.000000 | 0.000000 | 0.000000 | 0 | NO_CHANGE |

```text
PHASE20E1 STATUS: PASS
TRAININGS_COMPLETED: 8/8
CHECKPOINTS_VALID: 8/8
HELDOUT_EVALUATION: PASS
A_VS_B: REGRESSED
B_VS_C: MIXED
R1_3D_BENEFIT: MIXED
HONGZHANG_BC_EQUIVALENCE: VERIFIED
NEXT_PHASE: PHASE20E2
```

## 证据与解释

8 个必要模型独立训练；HongZhang B/C 明确共用 C 模型，不计作第九次训练。
本表为共同原始 RGB 参考，不使用各组不同的 masked RGB 评价充当公平比较。
表格指标来自逐视角 HELDOUT_PER_VIEW_METRICS.csv；真实过程、checkpoint/PLY
路径、大小、SHA256 与恢复加载结果均保留。时间为实际训练 wall-clock，含
冻结加载器和原实现定期验证，不含提交/后续评价。CUDA allocated/reserved
为进程真实峰值；nvidia-smi 2秒采样为设备占用峰值下界，并非精确进程峰值。

Three scenes and one fixed seed form a feasibility pilot, not broad generalization.
Held-out views are correlated and are not independent experimental repetitions.
No independent 3D GT exists: no truth Chamfer/F-score, physical dimensions or mesh.
All 68 frozen validation views lack matching existing manual fixed contours.
Frozen R0 proxy ROIs are predicted labels, not GT; fixed central rectangles are
method-independent descriptive spatial ROIs, not guaranteed plant-only regions.
CaoMei2 uses historically foreground-conditioned COLMAP cameras, shared by A/B/C.
CaoMei2 TEST32 raw0146 and WanNianQing1 raw0227 are not registered improvement cases.
Actual treatment switches: WanNianQing1 raw0037/raw0038/raw0165; CaoMei2 raw0139.
HongZhang B/C uses the same checkpoint; it cannot establish an R1-versus-R0 benefit.
Input treatment is explicit black-RGB precomposition only; no alpha-weighted RGB
loss or mask-guided sparse initialization is enabled. A is not historical ForeSplat.
CUDA rasterization/densification may not be bitwise deterministic across repeated
runs; one-seed differences cannot estimate this variability.

## 科研结论边界

A→B: REGRESSED; B→C: MIXED; R1 pilot benefit: MIXED.
WanNianQing1 三个训练输入变化和 CaoMei2 一个变化对应场景层面的实测效应，
但不能分别归因于单一视图、证明普遍泛化或宣称统计显著性。
背景污染/alpha覆盖 proxy 与原始全图指标应一起读；外观相似不等于几何精度。
固定视图图件及直接识图记录见 figures/ 与 reports/VISUAL_QA.md。
下一阶段建议增加独立场景和预声明多种子，不默认全量重训；未启动 E2。

## 真实性、来源与交付

E0 6831受管资产哈希及57生产源文件检查记录在PRETRAINING_INPUT_RECHECK.json。
最终再次核验输入哈希/代码与验收门，结果见FINAL_INPUT_RECHECK.json和FINAL_ACCEPTANCE.json。
G1–G20逐项见reports/ACCEPTANCE_GATES.csv。只提交本阶段小型脚本/报告/指标，
外部大型模型和原尺寸渲染不推送。未修改 R1、2DGS、M2M3-GO、Mask、相机或论文。

来源：本机冻结实验资产、真实训练日志、重载模型和逐帧评价，不基于训练记忆伪造结果。
PyTorch文档通过Context7抓取；博查检索HTTP403失败，未虚构搜索结果。

## 2026-10-14 收尾复核

本次继续工作未重新训练。177 份真实浮点渲染的 531 项主指标全部复算通过，
最大绝对差为 0；逐视角指标与场景汇总一致。6831 条 E0 受管资产及
57 个生产源文件再次核验无变化；冻结训练环境回归测试 13 项通过。
详细收尾记录见 `reports/FINAL_DELIVERY_AUDIT.md`，最终验收以
`reports/FINAL_ACCEPTANCE.json` 为准，提交与远端状态见 `DELIVERY_RECEIPT.md`。

PASS 表示本阶段训练、评价和证据链执行通过，不表示 R1 已证明具有普遍
三维增益。R1 的单种子结果仍为 MIXED；未启动 E2、M2M3-GO 或 Mesh，
未修改论文。

## Git 发布记录

实验资产提交：`addc10a840a8676efb19b5901c58f0e8a153192c`，正常推送至 `origin/main`。
远端分支 SHA 已于 `2026-10-14T11:46:40+08:00` 通过 `git ls-remote` 核验与本地一致。
该实验提交完成后工作区为 clean，`git diff --check` 通过。
本发布记录和 `DELIVERY_RECEIPT.md` 由后续独立文档提交补齐，
收据提交 SHA、最终远端一致性与最终工作区状态见本次对话交付。
该文档提交不改变训练、评价或验收结果；不执行下一阶段。
