# TEST32 GT 审计 — 2026-10-11

HUMAN GT: USER_ANNOTATION_REQUIRED
TEST32 EVALUATION: BLOCKED

## 输入与标注状态

| sequence ID | 原始 RGB | 原始 P6 候选 | 人工 JSON | 合格双人复核 |
| --- | --- | --- | --- | --- |
| CaoMei2 | 4 | 15 | 0 | 0 |
| ChangShouHua3 | 4 | 4 | 0 | 0 |
| DouBanLv3 | 4 | 4 | 0 | 0 |
| HongZhang | 4 | 4 | 0 | 0 |
| WanNianQing1 | 4 | 6 | 0 | 0 |
| WangWenCao2 | 4 | 6 | 0 | 0 |
| XianKeLai3 | 4 | 7 | 0 | 0 |
| XiangPiShu2 | 4 | 25 | 0 | 0 |

32/32 原始 RGB 与 Phase16 哈希、Phase17A 原始生成输入、Phase20C-FIX 原图链接一致。
71/71 候选通过显式 ID、历史 score-list、box、阈值、mask hash、原始分辨率及二值像素
核验；strict metadata loader 接受。候选缺失时脚本停止受影响帧，禁止替代/合成。
32 帧均为宽 2160、高 3840 的 RGB；这里的“8 个”是冻结序列身份数，
并不是已经由人独立确认的 8 株物理个体。

Phase16 DEV26（BaiZhang、DouBanLv2）与 TEST 无 sample/sequence ID 交集；
Phase20C Easy21/GT10 的 31 个 DEV RGB 同样无序列交集。
检查文件 SHA256 与解码后 RGB SHA256：TEST 内及 TEST/上述 DEV 的完全重复匹配为 0。
邻近帧存在相关性；未人工验证近重复/跨序列物理身份，不能声称“完全排除所有泄漏”。
明细见 `DUPLICATE_FRAME_AUDIT.csv`。

`ANNOTATION_LOCATION_AUDIT.csv` 对每帧检查本阶段与 Phase20C-FIX `annotations/`、
03-GT/03-GT-区分/03-GT-区分_challenge、Phase16 target.json/annotations 共 7 个路径。
找到 TEST32 人工 JSON 0/32；原始及既有复核表均无合格双人审批，人工视觉证据 0。
预检 `logs/annotation_gate.json` 明确拒绝评价；无预测/准确率模块被读取或运行。

## 历史标注包差异

Phase16 `target.jpg` 副本有 32/32 与原始 RGB 文件 hash 不同，
因此不作为冻结生成输入/新标注原图使用；记录其真实 hash，但不改写历史文件。
本阶段标注包直接链接经验证的原始 RGB。
旧 instruction card 的可选 plant/pot、linestrip、“身份不明删除帧”与新规范不一致；
新说明要求全部三个 label、polygon、保持 32 帧不变，并阻断不明确的身份。

## 人工待办

全部 32 帧列在 `TEST32_PENDING_ANNOTATION.csv`；每帧人工画 potted_plant/plant/pot、
存在时 blue_cube，记录目标身份与实际 D1/D3/D6/D11/D12（可 none）。
另一位真实独立审核者复核、签署并绑定最终 JSON SHA256。
主口径固定 `potted_clean = potted_plant AND NOT blue_cube`。
不允许 SAM3/R1/Agnes/AI 产生 GT 或替代第二审核人。

来源：本目录真实审计 CSV/JSON/日志；Phase16 冻结 manifest、Phase17A U00_Control、Phase17C 历史行为表及 Phase20C-FIX 决策/生产源。
