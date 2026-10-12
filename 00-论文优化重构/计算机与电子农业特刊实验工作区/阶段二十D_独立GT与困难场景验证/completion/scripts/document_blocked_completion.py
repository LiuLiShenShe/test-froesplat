from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE / "scripts"))

from phase20d_common import DIFFICULTIES, METRICS, read_csv, sha256, unique_index, write_csv, write_json, write_text


def document(stage, completion):
    gate_path = completion / "logs/annotation_gate_completion.json"
    gate = json.loads(gate_path.read_text())
    if gate.get("evaluation_allowed") is not False or gate.get("annotation_files_found") != 32:
        raise ValueError("This documenter applies only to the observed 32-file blocked gate, not a PASS or missing-GT run")
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    geometry = read_csv(completion / "GT32_GEOMETRY_REPORT.csv")
    difficulties = read_csv(stage / "DIFFICULTY_HUMAN_VERIFICATION.csv")
    if len(frames) != 32 or {row["frame_id"] for row in frames} != {row["frame_id"] for row in geometry}:
        raise ValueError("Full frozen coverage required")
    original = unique_index(read_csv(stage / "TEST32_HISTORICAL_SELECTION_CHANGE.csv"), "frame_id")
    write_json(completion / "GT_GATE_RESULT.json", gate)
    write_json(completion / "COMPLETION_STATUS.json", {
        "phase20d_completion": "PARTIAL", "human_gt": "USER_INPUT_REQUIRED", "human_review": "INCOMPLETE",
        "test32_r0_f1": "NOT VERIFIED", "test32_r1_f1": "NOT VERIFIED", "test32_r1_minus_r0": "NOT VERIFIED",
        "switch_cases": "NOT VERIFIED", "difficulty_labels": "PARTIAL", "foreground_method": "PROVISIONAL",
        "ready_for_phase20e": "NO", "evaluation_allowed": False, "gate_sha256": sha256(gate_path),
        "frozen_rgb_candidates_verified": True, "gt_json_found": 32,
        "valid_geometry": sum(row["geometry_status"] == "VALID" for row in geometry),
        "predictions_executed": False, "accuracy_metrics_computed": False, "three_dimensional_training_executed": False})
    per_frame = []
    for frame in frames:
        row = {"challenge_id": frame["challenge_id"], "frame_id": frame["frame_id"], "sequence": frame["sequence"],
               "evaluation_status": "BLOCKED_NOT_EVALUATED", "evaluation_executed": "NO",
               "historical_r0_id_not_new_run": original[frame["frame_id"]]["historical_r0_selected"],
               "historical_r1_id_not_new_run": original[frame["frame_id"]]["historical_r1_selected"],
               "r0_selected_id": "", "r1_selected_id": "", "oracle_id": "", "selection_changed": ""}
        for metric in METRICS:
            row.update({f"r0_{metric}": "", f"r1_{metric}": "", f"paired_delta_{metric}": ""})
        per_frame.append(row)
    write_csv(completion / "TEST32_PER_FRAME_EVALUATED.csv", per_frame)
    per_sequence = []
    for sequence in sorted({frame["sequence"] for frame in frames}):
        row = {"sequence": sequence, "expected_frames": sum(frame["sequence"] == sequence for frame in frames),
               "evaluation_status": "BLOCKED_NOT_EVALUATED", "evaluation_executed": "NO", "evaluated_frames": ""}
        row.update({f"{group}_{metric}": "" for group in ("r0", "r1", "paired_delta") for metric in METRICS})
        per_sequence.append(row)
    write_csv(completion / "TEST32_PER_SEQUENCE_EVALUATED.csv", per_sequence)
    summary = []
    for aggregation in ("frame_equal_mean", "sample_equal_mean", "frame_median", "frame_sd_ddof1",
                        "sequence_mean_median", "sequence_mean_sd_ddof1"):
        row = {"aggregation": aggregation, "expected_frames": 32, "expected_sequences": 8,
               "evaluation_status": "BLOCKED_NOT_EVALUATED", "evaluation_executed": "NO"}
        row.update({f"{group}_{metric}": "" for group in ("r0", "r1", "paired_delta") for metric in METRICS})
        summary.append(row)
    write_csv(completion / "TEST32_SUMMARY_EVALUATED.csv", summary)
    frame_map = unique_index(frames, "frame_id")
    strata = []
    for code, name in DIFFICULTIES.items():
        selected = [row for row in difficulties if row[code] == "YES"]
        strata.append({"difficulty": code, "description": name, "record_status": "RECORDED_DECISIONS_ONLY_NOT_VERIFIED",
                       "provisional_frame_count": len(selected),
                       "provisional_unique_sequence_count": len({frame_map[row["frame_id"]]["sequence"] for row in selected}),
                       "verified_evidence_frame_count": sum(row["status"] == "APPROVED" for row in selected),
                       "r0_f1": "", "r1_f1": "", "paired_delta": "", "r0_failure_count": "", "r1_failure_count": "",
                       "evaluation_status": "BLOCKED_NOT_EVALUATED"})
    write_csv(completion / "DIFFICULTY_STRATIFIED_RESULTS.csv", strata)
    issues = {row["frame_id"]: json.loads(row["issues_json"]) for row in geometry}
    missing = [(frame, ";".join(issue["label"] for issue in values if issue["code"] == "MISSING_REQUIRED_LABEL"))
               for frame, values in issues.items() if any(issue["code"] == "MISSING_REQUIRED_LABEL" for issue in values)]
    missing_table = "\n".join(f"| {frame} | {labels} |" for frame, labels in missing)
    write_text(completion / "HUMAN_REVIEW_ARCHIVE_REPORT.md", f"""# 已有人工工作归档 — 2026-10-12

用户报告已完成人工标注和真人复核。本阶段不否定其已做的工作，不要求重做 32 张。
实际找到 32 份原始 JSON，位于 `annotation_package/rgb/`（682234 字节合计），
不是原 Gate 指定的 `annotations/`。现已建立 32 个本地只读链接；逐文件原字节
SHA256 前后相同，见 `GT32_SOURCE_ARCHIVE.csv`。JSON、RGB、候选均未重写。

## 已可靠归档

- 两表恰好覆盖冻结 32 frame_id；与 challenge_id/RGB 哈希逐行匹配。
- 从现有 JSON 填入当前 `annotation_sha256`，追加图像尺寸和客观几何状态。
  此哈希仅绑定当前导出，不表示审批通过或轮廓已最终冻结。
- 保留 `22073ca5` 的全部 160 个 D 列 YES/NO 单元格；仅从同一帧五列派生标签。
  D1=12、D3=16、D6=15、D11=16、D12=12；16 帧为 `none`。类型允许重叠。
  这只是已有人工判定的转录，不是已经获得文字证据和双人审批的困难分类。
- 不填姓名、审核者代号、人工确认、来源声明、时间、目标描述、视觉证据、审批。
  现有 PENDING 状态不变；JSON 的 `checked=false` 不被解释为人类审核结论。

## 具体导出兼容性问题

32/32 文件的全部 75 个 shape 都是 `linestrip`，原冻结 Gate 只接受 `polygon`。
没有将开放折线填充成多边形、闭合端点、修改 shape_type 或补画缺失区域。
现有点无非有限、越界或退化坐标错误；这不意味着多边形 GT 已有效。
没有合法 raster GT，故 plant/pot 包含性与 potted_clean 非空检查均未通过，
不是误报“空掩膜”。全部文件均未含 blue_cube；实际是否不存在仍需真人确认。

15 帧另缺必需 label（11 帧缺 plant，15 帧缺 pot）：

| frame_id | missing labels |
| --- | --- |
{missing_table}

先寻找/提供已有的最终兼容导出及组件标注，不要求整集重新描绘。若只有当前导出，
需实际标注者确认折线的预期闭合语义及缺失组件；未经授权和真人确认，AI 不转换。
任何真实人员补充/格式修正都要保留旧版本、记录新 SHA256，并绑定最终审批。

## 只补录真实审核记录

`HUMAN_REVIEW_BATCH_REGISTRATION.csv`：实际标注者、首次审核者、独立第二审核者的
代号、实际完成时间、覆盖范围、身份映射的私有保存位置及公开代号确认。
首次审核者可为标注者，第二审核者必须与两者不同。不能用文件 mtime、Git 日期、
当前日期或用户总体完成声明推测审核日期/人员。

`HUMAN_REVIEW_FRAME_SUPPLEMENT.csv`：预填逐帧原始哈希、已有 D 判断、派生标签、
具体几何问题和缺失字段清单；人工区域为空。填真实目标身份、cube 判定、确认、
各真实困难标签对应的图像特定证据；`none` 也需要真人检查依据。不以通用文本
或重复序列标签代替逐帧记录。仅把实际记录合并回两份正式 CSV，不自动批量批准。

两个表中当前每帧都缺：annotation_origin、annotator、reviewer、second_reviewer、
三个人类声明、manual_annotation_confirmed、target_identity_description、
target_identity_confirmed、plant_confirmed、pot_confirmed、blue_cube_present、
difficulty_manually_confirmed、difficulty_evidence_json、reviewed_at、second_reviewed_at；
approval 仍 PENDING，difficulty status 仍 PENDING_HUMAN_VISUAL_REVIEW。

真实姓名与代号对应表保存在 completion/private/ 或库外私有位置，不能上传。
用户未提供匿名代号的公开确认，故未自动创建身份代号。当前公开材料不含身份。

## Gate

原 `scripts/validate_annotations.py` 未修改；32 JSON 存在，0/32 通过，退出码 2，
evaluation_allowed=false。权威日志为 `logs/annotation_gate_completion.json`；
`GT_GATE_RESULT.json` 是内容相同的交付副本。停止所有依赖 GT 的后续运行。

来源：本地真实 LabelMe JSON、冻结 manifest、22073ca5 人工五列和未修改的原 Gate。
""")
    write_text(completion / "TEST32_SWITCH_CASE_AUDIT.md", """# TEST32 两次历史候选切换

状态：新 CPU 运行、GT 指标、Oracle 和真人目标身份结果均 **NOT VERIFIED**。

| frame_id | historical R0 ID | historical R1 ID | R0 F1/IoU | R1 F1/IoU | Oracle | identity | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CaoMei2_0146 | CaoMei2_0146_04 | CaoMei2_0146_03 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| WanNianQing1_0227 | WanNianQing1_0227_00 | WanNianQing1_0227_01 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |

来源为原 TEST32_HISTORICAL_SELECTION_CHANGE.csv / Phase17C_test32_safety.csv，
不是新跑分。历史 2/32=6.25% 改变，30 帧未改变；所有 32 行均保留在带 BLOCKED
状态的输出中。没有排除退化帧，也不能把未切换、非空或原始 mask 间 IoU 当作正确。
两次切换均需批准的 potted_clean 和独立预测目标身份记录才能判断改善/退化/持平。
""")
    write_text(completion / "R0_R1_INDEPENDENT_COMPARISON.md", """# 冻结 R0/R1 独立比较 — BLOCKED

正式评价未执行，R0 F1、R1 F1、配对差、其他准确率指标和新 selection-change count
均 NOT VERIFIED；所有 *_EVALUATED.csv 是显式 BLOCKED 的覆盖/字段清单，
数值留空，不是结果表。历史候选 ID 单独命名，不伪装成新运行的 selected ID。

32 RGB / 8 序列 / 71 候选已通过原严格输入验证；不等同于 GT 或算法效果通过。
R0 只把 V1 ranking_policy 从 r1 改为 base，其余参数、score、候选与评价口径相同。
P6/per_instance/single，ensemble/SPNP/Residual/GCFC/A6/A7 全 OFF，权重和风险规则
完全来自 Phase20C-FIX。六位 confidence 为真实归档精度，不补猜原张量尾数。

Gate 通过后必须先跑 CaoMei2_0146 与 WanNianQing1_0227 两政策 CPU smoke，再全32。
复用生产算法入口和严格候选 metadata loader；不复制另一套 ranking。固定 GT 为
potted_plant AND NOT blue_cube。需报告 selected 与 final cleanup 层级以免混淆。

正式结果应重用 Phase20B/20C 的定义：F1、IoU_fg、背景/前景均值 mIoU、3px
Boundary-F1、双向边界距离 HD95、Leakage=FP/图像像素。Oracle 用同帧相同候选
对 GT 的最佳 F1，说明是原始候选层级；不要与清理后指标相混。Severe F1<0.5
仅为客观 proxy，真人目标身份/严重错误独立记录，不能由 F1 自动代签。

全32逐帧、8序列、frame-equal/sample-equal、median、SD(ddof=1)、配对差和
改善/退化均保留。若 bootstrap，单位为8个序列，说明小样本/宽区间，不把32
相关帧作为独立样本。无GT图像曾参与行为检查，不声称完全未见过的外部数据集。
现有 FIX/scripts/evaluate_independent_test.py 确实循环冻结32帧，但以 FIX 的
WORK/旧 review queue 寻址，缺少本次完整 Oracle/离散度/困难统计入口；不可把
其开发集表或只运行旧 CLI 当成 completion 正式全量结果。本次未声称实现了该
完整运行器；Gate 后仍需完成入口并按本规范实际执行。严禁 TEST32 结果驱动调参。
""")
    write_text(completion / "PHASE20D_COMPLETION_FINAL_REPORT.md", """# Phase20D-Completion — 2026-10-12

```text
PHASE20D COMPLETION: PARTIAL
HUMAN GT: USER_INPUT_REQUIRED
HUMAN REVIEW: INCOMPLETE
TEST32 R0 F1: NOT VERIFIED
TEST32 R1 F1: NOT VERIFIED
TEST32 R1-R0: NOT VERIFIED
SWITCH CASES: NOT VERIFIED
DIFFICULTY LABELS: PARTIAL
FOREGROUND METHOD: PROVISIONAL
READY FOR PHASE20E: NO
```

从干净 main 的 22073ca5 开始；所有新增代码只在 completion/，仅两个正式 CSV
填空的客观信息和由已存在五列一致派生的标签。原生产/排序/Gate、GT 点、RGB、
候选、2DGS/M2M3-GO 与论文正文均不变。没有重新 SAM3 推理或重新调参。

## 已完成可靠工作

找到了实际已做的 32 份 JSON（原在 RGB 同目录），建立本地原字节归档链接，
记录 32 个原始 SHA256、尺寸、shape/point/缺 label 诊断，绑定两份 CSV 的当前
JSON 版本。保留 160 个人工 D 单元格，补录表按帧预填事实但不填写人类证据。
原严格加载器验证 32 RGB、71 候选、box/ID/score/顺序/哈希与全部冻结来源。

## 实际阻断项，不把文件存在冒充合格 GT

- 32 文件全部75个 shape 为 linestrip，原 Gate 明确只接受 polygon；未偷偷转换。
- 15 帧缺必需组件 label：11 缺 plant、15 缺 pot，详见归档报告及逐形状 CSV。
- 点坐标和2160×3840尺寸合法，但不能据此通过 potted_clean/包含性/身份审批。
- 32 帧真实审核人、审核日期、目标身份、cube 判断、确认与图像特定困难证据
  尚未归档；用户声明已审核不提供未知字段的实际值。0/32 已批准审核记录。
- 原 Gate 真实退出2，evaluation_allowed=false。依用户 Step3，立即阻止预测/准确率。

请提供已完成的最终兼容导出/现成审核记录，或让实际人员只补录缺项；不要求重做
整套标注，也不自动填闭合区域、缺失标签、签名或时间。补录格式和隐私安排见
HUMAN_REVIEW_ARCHIVE_REPORT.md、两份 HUMAN_REVIEW_* 表。所有现有轮廓都保留。

## 未执行的研究步骤

两帧 CPU smoke、全32冻结 R0/R1、指标重算、Oracle、预测身份确认、困难定量、
真实比较图和最终方法锁定均 BLOCKED。没有填零或假统计；*_EVALUATED.csv
仅为显式 BLOCKED 的覆盖清单。历史 2/32 候选变化未解释为改善，30 不变未宣称
正确。独立算法结论 NOT VERIFIED，V1 仅保留开发集推荐的 PROVISIONAL。

## 验证和交接

具体测试、来源保护、覆盖/哈希、保留人工字段和空指标检查见 completion/logs/。
CPU 独立评价测试与指标重算一致性测试为 BLOCKED（不是 PASS），合成预检测试
不计入 TEST32 评价。完整参数仍在父目录 configs，未将 DEV 路径冒充 TEST 命令。
最终锁定说明和仅准备的三维计划分别见 FINAL_FOREGROUND_METHOD_LOCK.md、
PHASE20E_HANDOFF.md。停止在 Phase20D，不训练3D、不修改正文。

来源：真实本地文件、22073ca5 人工 D 列、原冻结校验逻辑和本阶段实际日志。
Bocha 搜索 HTTP403，未获得在线实验资料；结果只依据本地证据，不用训练知识填充。
""")


def main():
    parser = argparse.ArgumentParser(description="Document a real blocked gate; never computes or fabricates metrics")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    args = parser.parse_args()
    stage = args.stage_dir.resolve()
    document(stage, stage / "completion")


if __name__ == "__main__":
    main()
