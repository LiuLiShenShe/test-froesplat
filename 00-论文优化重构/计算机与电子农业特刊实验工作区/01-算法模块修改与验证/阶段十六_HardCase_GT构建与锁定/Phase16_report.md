# Phase 16 — Hard-case GT Challenge Set Construction and Split Freeze

**日期**: 2026-09-07 → 2026-09-09 更新（用户手动选帧 + 标注 10 帧；TEST 32 帧脚本选出）
**Scope**: 构建真正硬案例 GT 挑战集：确定性 sample-level DEV/TEST 切分、帧选择清单冻结、
标注协议与标注包、GT QA 门控、清单 SHA256 锁定。**本阶段不进行任何 A6/A7 评估。**
**Starting SHA**: `ad1c7395`（Phase 15 commit，工作树干净）

---

## §0. Starting-State Audit

Phase 15 封口后状态：

| 项 | 值 |
|---|---|
| 评估帧 | 21 帧 easy set（F1≥0.976，天花板效应确认） |
| GT | GT_potted_clean（21 帧，Phase 12 冻结） |
| 挑战集候选 | 60 帧难度排名（difficulty_ranking.csv，纯原始图属性） |
| non-GT samples | 15 个，~3,752 原始帧，COLMAP sparse/0 全部存在 |
| 上阶段代码 SHA | `ad1c7395`（Phase 15，未改动） |
| 本阶段新增 | 全部在 `阶段十六_HardCase_GT构建与锁定/` 内，不触碰任何历史产物 |

详细审计见 `Phase16_input_audit.md`（git HEAD/log/status、数据集索引、COLMAP 完备性、
帧可读性抽查、候选路径存在性）。

## §1. Scope and Deliverables

本阶段只做 **DATA**，不做 **MODEL**：

| 交付物 | 内容 | 状态 |
|---|---|---|
| `Phase16_input_audit.md` | 数据池完整审计 | ✅ 自动生成 |
| `Phase16_selection_manifest_preGT.csv` | 58 目标帧清单（预 GT） | ✅ |
| `Phase16_DEV_manifest.csv` | 26 帧 / 2 sample（BaiZhang, DouBanLv2） | ✅ |
| `Phase16_TEST_manifest.csv` | 32 帧 / 8 sample | ✅ |
| `Phase16_manifest.json` | SHA256 冻结（清单 + 58/58 逐帧图像校验和） | ✅ |
| `Phase16_GT_annotation_protocol.md` | 标注协议（身份规则、包含/排除、QA 门） | ✅ |
| `annotation_package/` | 58 个标注目录（target + 上下文 + 说明卡 + 联系表） | ✅ |
| `脚本/process_gt_challenge.py` | 升级版 GT 处理（Q1-Q8 QA 门控） | ✅ |
| `tests/test_phase16.py` | T1-T12 测试（16 项断言） | ✅ 16/16 通过 |
| `Phase16_report.md` | 本报告 | ✅ |

**本阶段不含**：任何 V00/V10/V01/V11 推理、任何参数调整、任何历史产物改动。

## §2. Data Pool Audit

| 资源 | 数量 | GT? | 用途 |
|---|---|---|---|
| 21 帧 easy set（Phase 12） | 21 | Yes | 历史基准（冻结，不碰） |
| 15 non-GT samples | ~3,752 帧 | No | 挑战集数据池 |
| 60 难度候选（Phase 15） | 60 | No | 帧选择候选池 |
| **Phase 16 目标帧** | **58** | **10 已标注（人类 GT）/ 48 待标注** | 挑战 GT |
| COLMAP sparse/0 | 15/15 | — | A6/A7 运行时依赖（Phase 17） |

COLMAP 注意：WangWenCao1 注册率仅 70.6%，已放入 DEV（不入锁定 TEST）。

## §3. Difficulty Ranking Source (Model-Independence)

帧难度排名完全来自 **原始图像属性**（`scout_difficulty.py`，5 特征）：

| 属性 | 方向 | 含义 |
|---|---|---|
| Laplacian variance | 低=难 | 清晰度 |
| Pixel entropy | 高=难 | 纹理复杂度 |
| ExG separation | 低=难 | 植物-背景色彩分离 |
| Green ratio | 极端=难 | 绿色占比 |
| Dynamic range | 低=难 | 对比度展宽 |

Composite = 0.2×(ls+es+xs+grc+ds)，min-max 归一化。**不含任何模型输出列**
（input audit 已验证 ranking CSV 无 V00/V10/V01/V11/F1 等列）。DouBanLv3 在
Phase 15 无候选（0/60），本阶段用同一公式在线重算 28 个新候选难度（30 均匀采样帧，
2 帧不可读被跳过）。

## §4. Sample-Level Split (Not Frame-Level)

**原则**：切分单位是 sample（序列），不是帧。同一序列的所有帧只属于一个 split。
这是 A7 视频记忆隔离的硬性要求（temporal memory 不能跨 DEV/TEST 泄漏）。

15 non-GT samples → **7 DEV / 8 TEST**，确定性分配（无随机）：

| Split | Samples |
|---|---|
| DEV (7) | BaiZhang, ChangShouHua1, DouBanLv2, WangWenCao1, WanNianQing2, XiangPiShu1, XianKeLai2 |
| TEST (8) | CaoMei2, ChangShouHua3, DouBanLv3, HongZhang, WangWenCao2, WanNianQing1, XiangPiShu2, XianKeLai3 |

**分配理由**：
1. **COLMAP 质量**：WangWenCao1（70.6% 注册率）→ DEV，永不入锁定 TEST。
2. **物种配对 1-1 切分**：ChangShouHua1/3、DouBanLv2/3、WangWenCao1/2、
   WanNianQing1/2、XiangPiShu1/2、XianKeLai2/3 每对一只进 DEV 一只进 TEST，
   无物种只在单侧出现（物种平衡见 §6 表格）。
3. **单例**：BaiZhang → DEV（凑 7/8）；CaoMei2、HongZhang → TEST。
4. **难度均衡**：DEV mean=0.622，TEST mean=0.637（原始图 composite，见 §5）。

不变式：DEV∩TEST=∅（sample 级 + sequence 级），15 个 sample 全部覆盖。
T2/T3/T11 测试锁定该不变式。

## §5. Frame Selection (Temporal Coverage + Difficulty)

每个 sample 至多 4 帧目标，选择策略（确定性，无随机）：

1. 候选池 = Phase 15 排名（或 DouBanLv3 在线重算）。
2. 按帧索引升序，把候选切成至多 4 个时间四分位。
3. 每个四分位取难度最高（composite max）的一帧。
4. 不足则用剩余最难帧补齐（`select_targets` 的 fill-remainder 分支）。

**结果**（用户手动选帧覆盖自动选择；TEST 由脚本按同一密度准则选出）：

| Split | 帧数 | 样本 | 帧索引范围 |
|---|---|---|---|
| DEV | 26 | BaiZhang(13), DouBanLv2(13) | BaiZhang 0033–0045, DouBanLv2 0029–0041 |
| TEST | 32 | 8 个 TEST 样本各 4 帧 | 见下 |
| **已标注** | **10** | BaiZhang(5) + DouBanLv2(5) | BaiZhang 0033–0037, DouBanLv2 0037–0041 |
| **待标注** | **48** | 剩余 DEV 16 + TEST 32 | — |

DEV 26 帧由用户从 ffmpeg 目录手动挑选：要求画面背景可见其他植株/盆栽（多株共存场景），
已通过 multi-plant density metric（green% × 横向峰值）客观验证：
BaiZhang 0033–0045（绿 25–34%，峰 8–15），DouBanLv2 0029–0041（绿 14–23%，峰 6–12）。

TEST 32 帧由脚本按同一密度准则（green%×peaks 前 4）从 8 个 TEST 样本选出：
CaoMei2(0007-0009,0146), ChangShouHua3(0189-0191,0200), DouBanLv3(0211,0220-0222),
HongZhang(0134,0200,0202,0229), WangWenCao2(0014-0017), WanNianQing1(0223-0225,0227),
XiangPiShu2(0133-0136), XianKeLai3(0117,0133,0141,0204)。

## §6. Difficulty Taxonomy (Heuristic, Model-Independent)

选择时无法目视 54 帧，故用**原始图属性启发式**映射难度类别（D1-D14 子集），
标注时由标注员人工确认：

| 难度标签 | DEV | TEST | 合计 | 启发式依据 |
|---|---|---|---|---|
| D11_cluttered_background | 17 | 19 | 36 | 熵高/绿分离弱 |
| D1_neighbor_plant_adhesion | 8 | 8 | 16 | 绿占比极高（密植） |
| D6_small_target | 1 | 1 | 2 | 绿占比极低 |
| （secondary）D3_fg_bg_confusion | — | — | 36 | 熵高+绿分离弱的伴生标签 |
| （secondary）D12_low_contrast | — | — | 2 | D6 伴生 |

**物种分布（DEV/TEST 每物种 1-1 平衡）**：

| 物种 | DEV 帧 | TEST 帧 |
|---|---|---|
| ChangShouHua | 4 | 4 |
| DouBanLv | 3 | 4 |
| WanNianQing | 4 | 4 |
| WangWenCao | 4 | 4 |
| XianKeLai | 3 | 4 |
| XiangPiShu | 4 | 4 |
| BaiZhang（单例） | 4 | 0 |
| CaoMei（单例） | 0 | 3 |
| HongZhang（单例） | 0 | 1 |

**重要说明**：heuristic 标签仅用于标注引导，**不参与任何评分或过滤**。标注员确认后的
难度类别将记录在最终 GT 审计中。D1-D14 中本阶段样本只触及 D1/D3/D6/D11/D12 五类。

## §7. Temporal Context Integrity

**无 contact graph / 邻接文件**（Phase 15 审计确认）。时间邻接 = 文件名顺序
（0000.jpg → 0001.jpg → …）。每目标帧记录上下文窗口：

- `context_start = max(0, frame-5)`
- `context_end = min(total-1, frame+5)`

58 帧（26 用户手动 DEV + 32 脚本密度 TEST）→ 全部通过 T9 校验：上下文帧真实存在、窗口包含目标帧。标注包内置 `context_prev.jpg` / `context_next.jpg` 供标注员确认目标个体身份。
`context_prev.jpg` / `context_next.jpg` 供标注员确认目标个体身份。

**数据质量发现**：WangWenCao1_0220.jpg 损坏不可读（cv2.imread 返回 None）。
标注包对上下文帧做了**最近可用帧回退**（`nearest_readable`，窗口内向外扫描），
确保每个标注目录都有可用的上下文。目标帧本身全部可读（54/54）。

## §8. Annotation Protocol (Phase16_GT_annotation_protocol.md)

完整协议见独立文档，核心要点：

- **GT 定义**：`potted_clean = potted_plant AND NOT blue_cube`（与 Phase 12 完全一致）。
- **Label schema**：`potted_plant`（必须）、`plant`（建议）、`pot`（建议）、
  `blue_cube`（如存在）。
- **目标身份规则**：目标是该 sample 序列中的**同一株个体**；先用 ±5 上下文帧确认身份；
  只标目标个体，排除邻居植株/背景植被/支撑结构/阴影反射/温室结构。
- **反偏差规则**：标注包内**不含任何模型输出**（无 V00/V10/V01/V11 掩膜/提示词），
  身份判断只依据 RGB 上下文帧。
- **QA 门 G-A1~G-A6**：非空、非全图、面积范围 500px~60%、与 cube 重叠 ≤10px² 等。
- **图像尺寸**：3840×2160（imageHeight=3840, imageWidth=2160，纵向），linestrip 闭合多边形。

## §9. Annotation Package

`annotation_package/`：58 个目录（26 DEV 用户选帧 + 32 TEST 脚本选帧）+ 联系表 + 无模型输出。

```
annotation_package/
├── HARD-BaiZhang-0033/
│   ├── target.jpg           # 3840×2160 原始帧副本
│   ├── context_prev.jpg     # 上下文前帧（±5 窗口内最近可用）
│   ├── context_next.jpg     # 上下文后帧
│   └── instruction_card.md  # 难度标签 + 身份规则 + 原始路径
├── ...（58 目录）
└── contact_sheets/
    ├── BaiZhang_contact.jpg # 该 sample 全部目标帧网格
    ├── DouBanLv2_contact.jpg
    └── ...（8 张 TEST sample 联系表）
```

验证：58/58 target 可读且形状正确、所有上下文文件存在、无 V10/V11/mask 等模型产物渗漏。

## §10. Manifest Freeze (SHA256)

`挑战集列表/Phase16_manifest.json`：

| 哈希 | 值 |
|---|---|
| sha256_full_manifest | `792a6724a2d7deaabaa562243dbcbd9f959e65690ee3ec6faa1ab1767a55f807` |
| sha256_dev_manifest | `063b33b9d7559eb0400528978993a63cbe0146e44e40f4c5d38fc0dfa7ba3cc6` |
| sha256_test_manifest | `7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157` |

- 58/58 目标帧逐帧 SHA256 图像校验和（基于**原始帧**，非重编码副本）。
- 确定性：相同输入行 + 相同顺序 → 相同哈希（T8 验证：重算与 manifest.json 完全一致）。
- **TEST manifest 对 Phase 17 不可变**（T12 锁定）。

## §11. GT Processing QA (process_gt_challenge.py)

从 Phase 15 版本升级，新增 8 道门控：

| Gate | 检查 | 违规处置 |
|---|---|---|
| Q1 | 帧必须在冻结 manifest 中（is_gt_target=yes） | 跳过并记录 |
| Q2 | 掩膜仅含 {0,255}（二进制） | 标记失败 |
| Q3 | 掩膜形状 == JSON 尺寸 == 原始图尺寸 | 标记失败 |
| Q4 | potted_clean 非空 | 标记失败 |
| Q5 | 非全图（前景 ≤60%） | 标记失败 |
| Q6 | 跨帧 SHA256 去重（防复制粘贴） | 报告可疑组 |
| Q7 | 每个 .json 必须有配对 .jpg | 跳过并记录 |
| Q8 | 前景占比 sanity 列（fg_ratio） | 审计输出 |

输出：`GT_potted_clean_challenge/`、`GT_QA_challenge/gt_audit_challenge.csv`、
`GT_QA_challenge/challenge_QA_visual/`。Phase 12 冻结 GT 路径完全隔离（T10 锁定）。

## §12. Tests (T1–T12)

`tests/test_phase16.py`，16 项断言全部通过：

| 测试 | 验证内容 |
|---|---|
| T1 | 切分确定性（SPLIT_ASSIGNMENT 覆盖 15 sample） |
| T2 | DEV∩TEST sample = ∅，manifest split 与 SPLIT_ASSIGNMENT 一致 |
| T3 | 无序列跨 split 泄漏 |
| T4 | GT 二进制 + 形状一致（GT 存在时） |
| T5 | shape mismatch 门控列存在 |
| T6 | empty mask 门控 + 无空掩膜 |
| T7 | 无重复 (sample, frame) 目标、challenge_id 唯一 |
| T8 | manifest.json SHA256 与 CSV 重算一致（full/DEV/TEST） |
| T9 | 上下文帧存在、可读、形状正确、包含目标帧 |
| T10 | 挑战 GT 输出路径与 Phase 12 冻结 GT 隔离；21 帧冻结 GT 未变 |
| T11 | DEV/TEST manifest 行级交集为空，26/32 数量 |
| T12 | TEST manifest 哈希冻结；58 帧原始图校验和与磁盘一致 |

**全量回归**：阶段十一/十二/十三/十四点一 + 阶段十六 = **100 passed**（84 历史 + 16 新增）。

## §13. Determinism & Reproducibility

| 元素 | 机制 |
|---|---|
| Split 分配 | 硬编码 dict（SPLIT_ASSIGNMENT），无随机 |
| 帧选择 | 时间四分位 + 难度排序，纯排序无随机 |
| 难度计算 | scout_difficulty.py（固定 5 特征 + 固定公式） |
| 哈希 | SHA256，规范化换行（\r\n→\n）+ 去 BOM |
| 测试 | pytest，纯数据断言，无模型 |

重跑 `phase16_split.py` → 相同 manifest；`phase16_freeze.py` → 相同哈希（T8 已证）。

## §14. No Model Inference

- 本阶段零 GPU 推理、零 SAM3 调用。
- 未读取任何 V00/V10/V01/V11 选择后掩膜、提示词选择.csv、共识投票.csv。
- 标注包无模型输出（§9 已验证）。

## §15. No Historical Artifact Modification

- `git status`：仅新增 `阶段十六_HardCase_GT构建与锁定/`（未跟踪）。
- Phase 12 GT_potted_clean 21 帧：未变（T10 断言 21 张）。
- Phase 14/14.1/15 全部产物：未触碰（SHA `ad1c7395` 未动）。

## §16. Difficulties Encountered

| 问题 | 处置 |
|---|---|
| `scout_difficulty` 不在 Phase 16 脚本路径 | sys.path 注入 Phase 15 脚本目录 |
| ranking CSV frame 带 `.jpg` 后缀 | 读取时 `Path(r["frame"]).stem` 规整 |
| DouBanLv3 Phase 15 无候选（0/60） | 在线重算 28 候选，选 4 帧（rank=999 标注） |
| WangWenCao1_0220 损坏不可读 | 上下文帧最近可用回退；目标帧不受影响 |
| 图像尺寸方向易混 | 确认 3840×2160（imageHeight=3840, imageWidth=2160）并统一写入协议/说明卡 |
| 哈希规范化不一致 | freeze 与 test 统一 `\r\n→\n` + 去 BOM 后再哈希 |
| T12 校验和基线 | 改为哈希**原始帧**（manifest 记录对象），而非重编码副本 |
| 难度特征选帧不符合科研目标 | 用户指出自动选帧（拉普拉斯/熵/ExG）单株为主 → 改为**用户手动选帧**（要求背景可见其他植株），并加 verifiable multi-plant density 列佐证 |
| 自动密度选帧仍被拒 | 用户指出自动选帧多为近距离/纯绿画面 → 完全由用户从 ffmpeg 目录选定 26 帧，脚本仅客观度量（绿%×峰）供审计 |
| 标注标签空格差异 | 用户 labelme 用 `"potted plant"`，规范标签 `"potted_plant"` → `.replace(" ", "_")` 归一化 |
| Q1 门全拒 | `待标注/` 子目录被当作 sample 目录扫描 → 删除后重跑 |

## §17. Acceptance Gates

| Gate | 判据 | 状态 |
|---|---|---|
| G1 | 15 non-GT samples 全部审计 | ✅ PASS（input audit） |
| G2 | 候选选择协议冻结 | ✅ PASS（用户手动选帧 DEV + 脚本密度选帧 TEST + manifest 确定性） |
| G3 | 切分为 sample/sequence 级 | ✅ PASS（T3） |
| G4 | DEV∩TEST sample = ∅ | ✅ PASS（T2） |
| G5 | DEV∩TEST sequence = ∅ | ✅ PASS（T3） |
| G6 | 48–60 GT 目标 | ✅ PASS（58：26 DEV + 32 TEST） |
| G7 | DEV GT 全部标注 | ⚠️ 10/26 已标注（16 待标注） |
| G8 | GT QA 通过 | ✅ PASS（10/10 P6 valid，0 QA fail，0 重复） |
| G9 | 目标身份协议文档化 | ✅ PASS（Phase16_GT_annotation_protocol.md §3） |
| G10 | 最终 manifest SHA256 冻结 | ✅ PASS（Phase16_manifest.json） |
| G11 | 无模型推理 | ✅ PASS（零计算） |
| G12 | 无参数调整 | ✅ PASS |
| G13 | V00 基线表征 | ⛔ **BLOCKED — 需 GT** |
| G14 | 历史测试通过 | ✅ PASS（100/100） |
| G15 | 历史证据未变 | ✅ PASS（git 仅新增） |

## §18. Status and Next Steps

**本阶段执行结果：PARTIAL — 帧选择完成（58），人类 GT 标注 10/58。**

当前状态：
- ✅ DEV 26 帧（用户手动选帧：BaiZhang 0033–0045，DouBanLv2 0029–0041）。
- ✅ TEST 32 帧（脚本按同一密度准则选出，8 样本各 4 帧）。
- ✅ 用户已标注 10 帧（各 5 帧），QA 全部通过（Q1–Q8，10/10 P6 valid）。
- ⏳ 剩余 48 帧待标注：DEV 16 + TEST 32。

下一步：
1. 继续标注剩余帧（待标注帧已备好在 annotation_package/HARD-*）
2. 标注输出到 `/data/fj/F2DMAS/03-GT-区分_challenge/<sample>/<frame>.json`
3. 运行 `脚本/process_gt_challenge.py` → 更新 GT_potted_clean_challenge + QA
4. 目视复核 QA 可视化
5. TEST GT 就绪后进入 Phase 17（锁定硬案例析因评估）

## §19. Output Directory

```
阶段十六_HardCase_GT构建与锁定/
├── Phase16_input_audit.md          # 输入审计
├── Phase16_GT_annotation_protocol.md  # 标注协议
├── Phase16_report.md               # 本报告
├── 挑战集列表/
│   ├── Phase16_selection_manifest_preGT.csv  # 58 帧选择清单（gt_status 10 已标注/48 待标注）
│   ├── Phase16_DEV_manifest.csv    # 26 帧 / 2 sample
│   ├── Phase16_TEST_manifest.csv   # 32 帧 / 8 sample
│   └── Phase16_manifest.json       # SHA256 冻结（58/58 校验和）
├── annotation_package/             # 58 标注目录 + 联系表
├── GT_potted_clean_challenge/      # 10 张挑战 GT 掩膜（BaiZhang 0033-0037, DouBanLv2 0037-0041）
├── GT_QA_challenge/                # gt_audit_challenge.csv + 可视化复核
├── 脚本/
│   ├── phase16_input_audit.py
│   ├── phase16_split.py
│   ├── phase16_annotation_package.py
│   ├── phase16_freeze.py
│   └── process_gt_challenge.py     # 升级版（Q1-Q8）
└── tests/
    └── test_phase16.py             # T1-T12（16 断言）
```

## §20. Reproducibility

```
Python / env：  /home/test/biosoft/enter/envs/sam3/bin/python
GPU：            未使用（零计算路径）
输入：          Phase 15 difficulty_ranking.csv + dataset_index.json + raw_frames
帧选择：        DEV 26 帧用户从 ffmpeg 手动选择；TEST 32 帧脚本按 multi-plant density（绿%×峰）选出
GT：            /data/fj/F2DMAS/03-GT-区分_challenge/（10 已有标注 / 48 待标注）
脚本：          阶段十六_HardCase_GT构建与锁定/脚本/ 下全部 .py
测试：          tests/test_phase16.py（100/100 全量回归）
```
