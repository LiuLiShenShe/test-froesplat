# RAP-FSAM3-v2 验证清单

## 用途

在正式执行特刊补充实验前，用本清单确认 RAP-FSAM3-v2 的代码、数据、输出和日志是否完整。新版重点验收 A1s 语义门控选择和 A5c 重建一致性修正；旧 A3/A4/A5 作为阶段八保护性链路和诊断日志保留。

## 代码与数据位置

| 项目 | 路径或取值 | 状态 |
| --- | --- | --- |
| 基线 FSAM3/SAM3 脚本 | `/data/fj/F2DMAS/03-SAM/segment_v2.py` | 已定位，作为历史基线参考；新版入口已单独创建 |
| SAM3 默认源码目录 | `/data/fj/F2DMAS/第三方源码/SAM3-latest` | 当前提交 `8e451d5`；`git ls-files '*.py'` 为 170 个 Python 文件，空字节检查为 0；已通过导入、1 帧真实冒烟和小批量验证 |
| SAM3 历史源码目录 | `/data/fj/F2DMAS/sam3-main` | 已定位但不再作为默认入口；`decoder.py` 等文件含空字节，真实导入会失败 |
| SAM3 权重与配置 | `/data/fj/F2DMAS/sam3/sam3.pt`，`/data/fj/F2DMAS/sam3/` | 已定位，已与最新源码完成 1 帧真实冒烟 |
| SAM3 Python 环境 | `/home/test/biosoft/enter/envs/sam3/bin/python` | 已验证；base 环境存在依赖或 torchvision 兼容问题，不建议用于正式运行 |
| 原始图像序列根目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames` | 已定位 |
| FFT 筛选图像根目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/02-fft_frames` | 已定位，建议 RAP-FSAM3 优先使用 |
| 现有 SAM3/SAM 掩膜目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/02-sam_masks/<样本名>` | 已定位，软链接到旧 `03-SAM/<样本名>` |
| 现有 FSAM3 掩膜目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/04-fsam3_masks` | 已定位，当前为空 |
| 新 RAP-FSAM3 输出目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜` | 已创建，已有 A0/A1/A5 1 帧冒烟输出、`KongQueZhuYu` A5 10 帧小批量输出和 `XianKeLai1` A5 细结构冒烟输出 |
| 阶段八验证产物目录 | `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段八验证产物` | 已生成阶段八验证总表、失败帧记录、P1-P5 候选对比图和选择细化对比图 |
| 人工标注掩膜目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/01-gt_masks` | 已定位；当前 JSON 为近似闭合 `linestrip` 标注，需确认是否按闭合多边形转换为二值 GT 后再计算正式 IoU/Dice |
| SEEM 对比掩膜目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/03-seem_masks` | 已定位 |
| COLMAP 稀疏重建目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked/<样本名>` | 已定位，正式实验优先使用 |
| A6 清洗位姿目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/02-位姿COLMAP/04-sanitized_for_A6/<样本名>` | 已定位，部分代表样本可用 |
| 2DGS 代码目录 | `/data/fj/F2DMAS/2d-gaussian-splatting-main` | 已定位 |
| 2DGS Python 环境 | `/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python` | 已定位 |
| 2DGS 实验输出目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/06-实验输出` | 已定位 |
| RAP-FSAM3 运行脚本与超参目录 | `/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证` | 已创建，已写入参数化生成脚本和命令示例 |

## 已确认样本与资源

| 类型 | 已确认内容 | 备注 |
| --- | --- | --- |
| SAM3/SAM 掩膜覆盖 | 19 个样本有统一入口；`KongQueZhuYu` 实测有 210 张 `mask_*.png` 和 210 张 `crop_*.png` | `BaiZhang` 暂未在 `02-sam_masks` 中看到入口，需后续补查或重跑。 |
| SEEM 掩膜覆盖 | 20 个样本有统一入口 | 可用于实验一横向对比。 |
| 人工标注样本 | `CaoMei1`、`ChangShouHua2`、`DouBanLv1`、`KongQueZhuYu`、`XianKeLai1` | 适合实验二和实验三的分割指标。 |
| COLMAP final_locked | 20 个样本均有入口 | 正式训练优先读取。 |
| 已有 2DGS 输出 | `KongQueZhuYu`、`CaoMei2`、`DouBanLv3`、`XianKeLai1` 等 | 可作为实验四和代表样本选择参考。 |

## 输出命名规范

建议输出结构：

```text
数据管理/03-分割Mask/05-RAP-FSAM3掩膜/
  样本名/
    图像索引.csv
    候选掩膜/
      P1_绿色植物/
      P2_整株去花盆/
      P3_叶和茎/
      P4_作物幼苗/
      P5_去背景植物体/
    提示词评分.csv
    语义门控评分.csv
    语义门控框/
    选择后掩膜/
    正负提示细化掩膜/
    残差修复掩膜/
    重提示帧标记.csv
    几何反馈.csv
    几何修正提示.csv
    几何修正掩膜/
    corrective_geometry_delta.csv
    最终掩膜/
    前景图/
    透明图/
```

## 冒烟验证样本

| 样本 | 选择原因 | 状态 |
| --- | --- | --- |
| `KongQueZhuYu` | 有 SAM 掩膜、GT、COLMAP、A6 和多组 2DGS 消融结果，适合作为主冒烟样本 | A0、A1、A5 1 帧真实冒烟已通过；A5 10 帧小批量流程通过，但存在绿色花盆和邻近绿色植物泄漏 |
| `XianKeLai1` | 有 GT、COLMAP、A6 代表样本结果，适合检查细结构和边界 | A5 1 帧细结构真实冒烟通过；几何反馈 `crop_0000.png -> mask_stem=0000` 对齐成功 |

## RAP-FSAM3 命令行开关验收

- [x] 新脚本支持 `--input_dir`、`--output_dir`、`--sam3_repo`、`--sam3_checkpoint`。
- [x] 新脚本支持 `--prompt_list`、`--default_prompt_id`、`--use_prompt_ensemble`、`--prompt_selection_mode`。
- [x] 新脚本支持 `--use_spnp_refinement`、`--spnp_backend`。
- [x] 新脚本支持 `--use_residual_repair`、`--closing_kernel`、`--opening_kernel`、`--thin_repair_mode`。
- [x] 新脚本支持 `--use_reprompt_detection`、`--reprompt_detection_mode`。
- [x] 新脚本支持 `--use_geometry_feedback`、`--geometry_feedback_mode`、`--colmap_dir`。
- [x] 不传任何新增模块开关时，输出等价于单提示词基线；A0 1 帧冒烟中 `SPNP后端=disabled`。
- [x] A0、A1、A5 可通过不同 `--xxx` 参数组合运行；A2-A4 由同一入口的中间开关组合派生。
- [x] 新脚本支持 `--use_semantic_gate`、`--semantic_gate_backend`、`--semantic_box_json` 和语义门控权重参数。
- [x] 新脚本支持 `--use_corrective_geometry`、`--corrective_geometry_backend` 和几何修正阈值参数。
- [x] 新版 A0、A1、A1s、A2、A5c 可通过不同 `--xxx` 参数组合运行，且不覆盖阶段八旧输出。

## 检查项

### 文件对齐

- [x] 每个输出掩膜与原图同名。
- [x] 掩膜尺寸与原图一致。
- [x] 没有静默跳帧。
- [x] 输出顺序与原始帧顺序一致。

### 候选掩膜生成

- [x] P1-P5 每个提示词都能生成候选掩膜。
- [x] 不同提示词的掩膜分目录保存。
- [x] 空掩膜和过大掩膜被记录。
- [x] 运行时间被记录。
- [x] 失败次数被单独汇总记录；每次运行输出 `失败汇总.json`，已补写既有 `KongQueZhuYu` 10 帧和 `XianKeLai1` 1 帧结果。

### 可靠性评分

- [x] 计算面积合理性。
- [x] 计算连通域得分。
- [x] 计算边界得分。
- [x] 计算时序稳定得分。
- [x] 计算前背景对比得分。
- [x] 记录下方泄漏得分和下方区域占比；默认不参与评分，显式设置 `score_weights` 中的 `leak` 后启用。
- [x] 每帧每提示词都有总分。
- [x] 每帧都有最终选择的提示词编号。
- [x] 候选清理默认执行最大连通域；残差修复后已补充最终最大连通域保底，防止细修复阶段重新引入碎片。

### A1s 语义门控选择

- [x] 生成或读取植物主体框。
- [x] 生成或读取花盆/土壤/桌面干扰框。
- [x] 生成或读取侧边邻近植物/背景干扰框。
- [x] 计算 `target_box_score`。
- [x] 计算 `pot_overlap_penalty`。
- [x] 计算 `side_distractor_penalty`。
- [x] 计算 `center_prior_score`。
- [x] 写出 `语义门控评分.csv`。
- [x] 保存语义门控可视化调试图。
- [x] `XianKeLai1` GT1 上 A1s 已避免旧 A1 选错 P3，并修正为 P2。

### 结构化正负提示

- [x] 正点来自植物掩膜内部主体区域。
- [x] 负点来自掩膜外扩环带。
- [x] 如果使用下方花盆区域负点，需明确记录。
- [x] 保存细化前后掩膜。
- [x] 当前默认 A5 使用 SAM3 正负 box 几何提示；`postprocess_only` 作为备用后端保留。

### 残差修复

- [x] 保存闭运算/开运算后的残差修复掩膜。
- [x] 保存被删除并恢复的细结构残差区域。
- [x] 只恢复满足细结构条件的区域。
- [x] 不恢复大块背景。

### 重提示检测

- [x] 计算相邻帧掩膜 IoU。
- [x] 计算面积突变。
- [x] 计算图像结构变化或等价指标。
- [x] 计算边界不稳定指标。
- [x] 标记提示词敏感帧。
- [x] 如有 COLMAP 结果，计算几何一致性，并写入最终重提示标记。
- [x] 支持 COLMAP 图像名 `crop_0000.png` 与输入帧 `0000.jpg` 的几何反馈对齐，并在 `几何反馈.csv` 中记录 `mask_stem`。

### A5c 重建一致性修正

- [x] 从 COLMAP track projection 或 2DGS 一致性证据中生成 mask 外前景点。
- [x] 从几何无支撑区域中生成负提示或删除候选区域；负向删除默认关闭，需显式 `--geometry_enable_negative_correction` 才启用。
- [x] 输出 `几何修正提示.csv`。
- [x] 输出 `几何修正掩膜/`。
- [x] 输出 `corrective_geometry_delta.csv`。
- [x] 记录修正前后 IoU、delta 像素比例、接受/拒绝原因。
- [x] A5c 已在 `XianKeLai1` GT1、`KongQueZhuYu` GT5 候选复用冒烟和 `KongQueZhuYu` GT5 完整 `sam3_if_supported` 后端重跑中产生非零且可解释的正向几何 delta。
- [x] 修正过强时能按 `geometry_correct_min_iou` 或 `geometry_correct_max_delta_ratio` 回退到 A2。

### 视觉检查

- [x] 生成 P1-P5 候选掩膜对比图；见 `阶段八验证产物/*_P1-P5候选对比.png`。
- [x] 生成选择前后对比图；见 `阶段八验证产物/*_选择细化对比.png`。
- [x] 检查失败帧；见 `阶段八验证产物/失败帧与视觉检查记录.md`。
- [x] 检查细结构样本。
- [x] 检查花盆和背景泄漏样本。

## 正式实验准入条件

只有满足以下条件，才进入实验二、三、四：

- 文件对齐通过。
- 候选提示词掩膜可稳定生成。
- 每帧可靠性评分可追踪。
- A1s 已生成语义门控评分和调试可视化，且能解释最终候选选择。
- A2 在 `KongQueZhuYu` GT5 上保持已观察到的改进趋势。
- A5c 已产生可量化几何修正 delta，不能只输出 low/ok 日志。
- 最终掩膜视觉上合理；`KongQueZhuYu` GT5 已完成完整 A2/SAM3 后端新版实验三主表重跑，候选复用冒烟仅作为机制诊断保留。
- 不覆盖基线 FSAM3 结果。
- 已记录已知限制。

## 验证记录

| 日期 | 样本 | 结果 | 后续动作 |
| --- | --- | --- | --- |
| 2026-06-05 | `KongQueZhuYu`，`0000.jpg` | A0 单提示词真实冒烟通过；默认关闭新增模块，`SPNP后端=disabled`，最终掩膜尺寸为 2160×3840 | 作为实验三 A0 基线入口 |
| 2026-06-05 | `KongQueZhuYu`，`0000.jpg` | A1 多提示词选择真实冒烟通过；P1-P5 候选、评分表、选择表、最终掩膜、前景图、透明图和叠加图均生成；自动选择 P2，前景面积比例约 0.2248 | 可扩大到 5-10 帧，作为实验二和实验三 A1 小批量验证 |
| 2026-06-05 | `KongQueZhuYu`，`0000.jpg` | A5 完整开关真实冒烟通过；`sam3_if_supported` 细化被接受，细化 IoU 约 0.985，几何分数约 0.601 并标记为 `low`，最终重提示标记为 1 | 下一步扩大到 5-10 帧，并补测 `XianKeLai1` |
| 2026-06-05 | `KongQueZhuYu`，`0000.jpg`-`0009.jpg` | A5 完整开关 10 帧小批量通过；最终掩膜 10 张、候选掩膜 50 张、SPNP/残差/叠加图各 10 张；SPNP 10/10 接受；自动选择 P2 8 帧、P3 2 帧；平均前景面积比例约 0.2382；几何标记 10/10 为 `low` | 流程可进入消融批处理准备，但正式指标前需处理绿色花盆和邻近植物泄漏 |
| 2026-06-05 | `XianKeLai1`，`0000.jpg` | A5 细结构真实冒烟通过；自动选择 P3，前景面积比例约 0.1464；SPNP 接受，细化 IoU 约 0.989；几何分数约 0.975，标记为 `ok`；`crop_0000.png` 已通过 `mask_stem=0000` 对齐 | 可作为细结构样本继续扩到 5-10 帧；GT `linestrip` 到二值掩膜转换方式需先确认 |
| 2026-06-05 | `KongQueZhuYu`，`0000.jpg`-`0009.jpg` | 候选复用泄漏惩罚重选通过；复用 A5 的 50 张候选掩膜，不重跑 SAM3；`score_weights` 加入 `leak=2` 后，平均前景面积比例从约 0.2382 降至约 0.1859，P2 花盆候选被压低，主要改选 P3/P1 | 该小修能缓解绿色花盆泄漏，但 `0007` 等帧仍有左侧邻近绿色植物粘连，需继续处理侧边/目标中心约束 |
| 2026-06-05 | `KongQueZhuYu`，`0000.jpg`-`0009.jpg` | 最大连通域专项复查完成；原 A5 候选掩膜基本均已是单连通域，最终掩膜 9/10 帧为单连通域；`0007` 虽有 3 个连通域，但最大连通域占 99.999915%，强制最大连通域仅移除 2 个像素 | 最大连通域可作为碎片保底，但当前花盆/邻近植物泄漏已与主体粘成同一大连通域，不能单独解决该失败模式 |
| 2026-06-05 | 阶段八汇总 | 已生成阶段八验证总表；`KongQueZhuYu` A5 10 帧：10 图、50 候选、14 空候选、0 最终空掩膜、SPNP 10/10 接受、几何 low 10/10；`XianKeLai1` A5 1 帧：5 候选、1 空候选、0 最终空掩膜、几何 ok | 阶段八算法实现与小样本冒烟验证可收口；正式实验前仍需处理 `KongQueZhuYu` 的粘连泄漏质量门槛 |
| 2026-06-05 | `XianKeLai1` GT1 | A1s + A5c 候选复用冒烟通过；A1s 将旧 A1 的 P3 错选修正为 P2，F1 从 0.860 回到 0.963；A5c 产生正向几何 delta，修正像素比例约 0.000604，修正前后 IoU 约 0.9966 | A1s 可作为正式实验三新增主模块；A5c 作为正向几何补充模块进入正式重跑 |
| 2026-06-05 | `KongQueZhuYu` GT5 | A1s + A5c 候选复用冒烟通过；5/5 帧生成语义门控评分和调试图，5/5 帧生成 A5c 正向几何 delta，单帧修正比例约 0.000001-0.000441；负向几何删除默认关闭 | 该候选复用结果只验证接口和机制；正式指标已由完整 A2 的 `sam3_if_supported` 后端重跑替代 |
| 2026-06-05 | `KongQueZhuYu` GT5 | 新版实验三主表已用完整 `sam3_if_supported` A2 后端重跑 A0/A1/A1s/A2/A5c；A2 相对 A0 将 F1 从 0.9723 提升到 0.9820、mIoU 从 0.9626 提升到 0.9756、泄漏能量从 0.005283 降到 0.001372；A5c 5/5 接受几何修正，平均修正像素比例约 0.000166，几何内点覆盖率约 0.9883 | 分割主表可填写；结论写作需区分 A2 主增益、A1s 机制成功和 A5c 几何一致性补充 |
