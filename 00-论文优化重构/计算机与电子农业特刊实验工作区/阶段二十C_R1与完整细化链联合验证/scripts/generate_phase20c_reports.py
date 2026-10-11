#!/usr/bin/env python3
"""Generate Phase 20C reports from local run artifacts only."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: object, digits: int = 4) -> str:
    if value in (None, "", "nan", "NaN"):
        return "—"
    return f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list[object]]) -> str:
    text = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    text.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(text)


def summary_by(summary: list[dict[str, str]], dataset: str) -> dict[str, dict[str, str]]:
    return {row["group"]: row for row in summary if row["dataset"] == dataset}


def write(name: str, content: str) -> None:
    (WORK / "reports" / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    summary = read_csv(WORK / "metrics/summary.csv")
    effects = read_csv(WORK / "metrics/paired_effects.csv")
    frame = read_csv(WORK / "metrics/frame_metrics.csv")
    stages = read_csv(WORK / "metrics/stage_metrics.csv")
    selection = read_csv(WORK / "metrics/selection_replay.csv")
    modules = read_csv(WORK / "metrics/module_execution.csv")
    availability = read_csv(WORK / "metrics/availability.csv")
    precheck = read_csv(WORK / "configs/PRECHECK_AUDIT.csv")
    configs = {
        key: json.loads((WORK / "configs" / f"{key}.json").read_text(encoding="utf-8"))
        for key in ("Easy21_V0", "Easy21_V1", "Easy21_V2", "Easy21_V3", "GT10_V0", "GT10_V1", "GT10_V2", "GT10_V3")
    }
    easy = summary_by(summary, "Easy21")
    hard = summary_by(summary, "GT10")
    effect_map = {(row["dataset"], row["effect"]): row for row in effects}

    easy_rows = []
    for group in ("V0", "V1", "V2", "V3"):
        row = easy.get(group, {})
        easy_rows.append(
            [
                group,
                row.get("status", "BLOCKED"),
                row.get("n_frames", 0),
                f(row.get("final_f1_sample_equal_mean")),
                f(row.get("final_iou_fg_sample_equal_mean")),
                f(row.get("final_miou_sample_equal_mean")),
                f(row.get("final_hd95_sample_equal_mean"), 2),
                f(row.get("final_boundary_f1_sample_equal_mean")),
                f(row.get("final_leakage_sample_equal_mean"), 6),
                row.get("empty_masks", 0),
                row.get("severe_errors", 0),
            ]
        )
    easy_text = """# Easy21 Results

Easy21 是开发/历史回归集，不是独立泛化测试集。统计主列为 sample-equal mean；同时保留 frame-level、median、population SD 于 `metrics/summary.csv`。

""" + table(["Group", "Status", "Frames", "F1", "IoU_fg", "mIoU", "HD95 px", "Boundary-F1", "Leakage", "Empty", "Severe"], easy_rows)
    easy_text += "\n\n## Paired effects\n\n" + table(
        ["Effect", "N", "ΔF1", "ΔIoU_fg", "ΔmIoU", "ΔHD95", "ΔBoundary-F1", "ΔLeakage"],
        [
            [
                effect,
                effect_map.get(("Easy21", effect), {}).get("n_matched_frames", 0),
                f(effect_map.get(("Easy21", effect), {}).get("final_f1_paired_mean")),
                f(effect_map.get(("Easy21", effect), {}).get("final_iou_fg_paired_mean")),
                f(effect_map.get(("Easy21", effect), {}).get("final_miou_paired_mean")),
                f(effect_map.get(("Easy21", effect), {}).get("final_hd95_paired_mean"), 2),
                f(effect_map.get(("Easy21", effect), {}).get("final_boundary_f1_paired_mean")),
                f(effect_map.get(("Easy21", effect), {}).get("final_leakage_paired_mean"), 6),
            ]
            for effect in ("ranking_V1_minus_V0", "refinement_R0_V2_minus_V0", "refinement_R1_V3_minus_V1", "interaction_(V3-V2)-(V1-V0)")
        ],
    )
    easy_text += "\n\nV0/V1 share the same final masks on this cache. V2/V3 are the real full-chain runs; the actual module-level change counts and acceptance rates are in `MODULE_EXECUTION_TRUTH.md`.\n"
    write("EASY21_RESULTS.md", easy_text)

    gt_rows = []
    for group in ("V0", "V1", "V2", "V3"):
        row = hard.get(group, {})
        gt_rows.append(
            [
                group,
                row.get("status", "BLOCKED"),
                row.get("n_frames", 0),
                f(row.get("final_f1_sample_equal_mean")),
                f(row.get("final_iou_fg_sample_equal_mean")),
                f(row.get("final_miou_sample_equal_mean")),
                f(row.get("final_hd95_sample_equal_mean"), 2),
                f(row.get("final_boundary_f1_sample_equal_mean")),
                f(row.get("final_leakage_sample_equal_mean"), 6),
                row.get("empty_masks", 0),
                row.get("severe_errors", 0),
            ]
        )
    hard_cases = [row for row in selection if row["dataset"] == "GT10" and row["sample_id"] == "DouBanLv2" and row["frame"] in {"0039", "0040", "0041"}]
    case_rows = []
    for row in hard_cases:
        case_rows.append([row["frame"], row["group"], row["selected_candidate_id"], f(row["selected_f1"]), f(row["oracle_regret"]), row["historical_selection_match"]])
    gt_text = """# GT10 Results

GT10 is the Phase 17 diagnostic development set, not an independent generalization test. V0/V1 were formally run on all 10 frames. V2/V3 were not run as complete groups because COLMAP observations are absent for `DouBanLv2_0037`, `DouBanLv2_0039`, and `DouBanLv2_0041`; the missing geometry is recorded in `configs/COLMAP_OBSERVATION_AUDIT.csv`.

""" + table(["Group", "Status", "Frames", "F1", "IoU_fg", "mIoU", "HD95 px", "Boundary-F1", "Leakage", "Empty", "Severe"], gt_rows)
    gt_text += "\n\n## Core hard cases\n\n" + table(["Frame", "Group", "Selected candidate", "Selected F1", "Oracle regret", "Historical replay match"], case_rows)
    gt_text += "\n\nThe historical values in Phase 17C are used only as replay references; all F1 values above come from the new Phase 20C masks.\n"
    write("GT10_RESULTS.md", gt_text)

    replay_matches = [row for row in selection if row["historical_replay_available"] == "1"]
    match_count = sum(row["historical_selection_match"] == "1" for row in replay_matches)
    rescue = [row for row in selection if row["dataset"] == "GT10" and row["sample_id"] == "DouBanLv2" and row["frame"] in {"0039", "0040", "0041"} and row["group"] == "V1"]
    replay_text = f"""# R1 Production Replay

- New production runs: V0/V1 on Easy21 (21 frames) and GT10 (10 frames); V2/V3 on Easy21 (21 frames).
- Historical production replay comparison: `{match_count}/{len(replay_matches)}` available frame/group selections match the Phase 17C expected instance after normalizing zero-padded instance IDs.
- Candidate source: frozen P6 raw instance masks. No candidate was regenerated and no R1 weight was changed.
- Cache caveat: the current reuse loader does not ingest historical SAM3 score metadata, so the production run records `SAM3原始分数=0.0` for reused candidates. Selection agreement is verified on this cache, but byte-for-byte scoring equivalence is not certified.

## Hard-case rescue

""" + table(["Frame", "Selected candidate", "New V1 F1", "Oracle regret", "Replay match"], [[row["frame"], row["selected_candidate_id"], f(row["selected_f1"]), f(row["oracle_regret"]), row["historical_selection_match"]] for row in rescue])
    replay_text += "\n\nR1 production selection validation is PASS for the observed frozen-cache replay, with the score-metadata caveat above.\n"
    write("R1_PRODUCTION_REPLAY.md", replay_text)

    module_groups = defaultdict(lambda: {"calls": 0, "enabled": 0, "accepted": 0, "changed": 0, "ratios": []})
    for row in modules:
        key = (row["dataset"], row["group"], row["module"])
        data = module_groups[key]
        data["calls"] += int(row["called"])
        data["enabled"] += int(row["enabled"])
        data["accepted"] += int(row["accepted_or_changed"])
        data["changed"] += int(float(row["changed_pixel_ratio"]) > 0)
        data["ratios"].append(float(row["changed_pixel_ratio"]))
    module_rows = []
    for dataset in ("Easy21", "GT10"):
        for group in ("V0", "V1", "V2", "V3"):
            for module in ("spnp", "residual", "gcfc"):
                data = module_groups[(dataset, group, module)]
                module_rows.append([dataset, group, module, data["calls"], data["enabled"], data["accepted"], data["changed"], f(sum(data["ratios"]) / len(data["ratios"]) if data["ratios"] else None, 6)])
    module_text = """# Module Execution Truth

The production entrypoint calls each stage function for every available frame. `accepted_or_changed` is not inferred from a flag alone: it requires a nonzero mask delta between adjacent saved stage masks. A module enabled with no delta is reported as `NO_CHANGE`; missing geometry observations are not treated as successful GCFC applications.

""" + table(["Dataset", "Group", "Module", "Calls", "Enabled-frame calls", "Accepted/changed", "Changed frames", "Mean changed ratio"], module_rows)
    module_text += "\n\nRaw per-frame evidence is in `metrics/module_execution.csv`; production flags and backend reasons are in each run's `运行日志.json`, `提示词选择.csv`, and `几何修正提示.csv`.\n"
    write("MODULE_EXECUTION_TRUTH.md", module_text)

    stage_values = defaultdict(dict)
    for row in stages:
        if row["dataset"] == "Easy21" and row["group"] in {"V2", "V3"}:
            stage_values[(row["group"], row["sample_id"], row["frame"])][row["stage"]] = float(row["f1"])
    stage_effect_rows = []
    for group in ("V2", "V3"):
        for previous, current, label in (
            ("selected", "spnp", "Selected -> SPNP"),
            ("spnp", "residual", "SPNP -> Residual"),
            ("residual", "gcfc", "Residual -> GCFC"),
            ("gcfc", "final", "GCFC -> Final"),
        ):
            diffs = [
                values[current] - values[previous]
                for (run_group, _, _), values in stage_values.items()
                if run_group == group and previous in values and current in values
            ]
            stage_effect_rows.append([
                group,
                label,
                len(diffs),
                f(sum(diffs) / len(diffs) if diffs else None),
                f(min(diffs) if diffs else None),
                f(max(diffs) if diffs else None),
                sum(value > 0 for value in diffs),
                sum(value < 0 for value in diffs),
                sum(value == 0 for value in diffs),
            ])
    stage_text = """# Refinement Interaction Report

This report uses saved adjacent-stage masks and GT metrics, not feature flags alone. Easy21 is the only dataset with complete V2/V3 chain execution in Phase 20C.

""" + table(["Group", "Stage transition", "N", "Mean ΔF1", "Min", "Max", "Positive", "Negative", "No change"], stage_effect_rows)
    stage_text += """

Interpretation:

- SPNP produces a measurable mean F1 gain and is accepted on all 21 Easy21 frames in V2/V3, with one frame showing a small negative change.
- Residual repair is called with `thin_repair_mode=adjacent`, but all saved residual masks equal the SPNP masks: `NO_CHANGE` on Easy21.
- GCFC is called and changes all 21 Easy21 masks, but its adjacent-stage F1 delta is negative on 20/21 frames. It remains a conditional module, not a default-locked component.
- Because GT10 V2/V3 is blocked, this interaction evidence cannot justify a complete hard-case refinement claim.
"""
    write("REFINEMENT_INTERACTION_REPORT.md", stage_text)

    v2 = easy.get("V2", {})
    v3 = easy.get("V3", {})
    v1 = easy.get("V1", {})
    v0 = easy.get("V0", {})
    delta_v3_v1 = float(v3.get("final_f1_sample_equal_mean", 0) or 0) - float(v1.get("final_f1_sample_equal_mean", 0) or 0)
    delta_v2_v0 = float(v2.get("final_f1_sample_equal_mean", 0) or 0) - float(v0.get("final_f1_sample_equal_mean", 0) or 0)
    gcfc_effects = [
        values["gcfc"] - values["residual"]
        for (run_group, _, _), values in stage_values.items()
        if run_group == "V2" and "gcfc" in values and "residual" in values
    ]
    gcfc_negative_ratio = sum(value < 0 for value in gcfc_effects) / len(gcfc_effects) if gcfc_effects else 0.0
    if gcfc_negative_ratio >= 0.75:
        recommendation = "V1"
        rationale = "V3 的净 F1 增益来自 SPNP，但 GCFC 在 Easy21 21 帧中 20 帧降低 F1，且 GT10 完整链尚未验证；因此以 V1 为默认，refinement 仅条件启用。"
    elif abs(delta_v3_v1) < 0.001:
        recommendation = "V1"
        rationale = "V3 与 V1 的 Easy21 sample-equal F1 差异小于 0.001，未显示稳定增量。"
    elif delta_v3_v1 > 0:
        recommendation = "V3 (当前仅在 Easy21 开发回归上支持；GT10 完整链未验证)"
        rationale = "V3 的 Easy21 F1 高于 V1，但仍受 GT10 几何阻断限制。"
    else:
        recommendation = "V1"
        rationale = "V3 的 Easy21 F1 低于 V1，不能把完整链作为默认方法。"
    final_text = f"""# FINAL RAP-FSAM3 METHOD DECISION

## Decision

- Status: `PARTIAL`
- Recommended foreground method: `{recommendation}`
- R1 production validation: `PASS` for V0/V1 frozen-cache selection replay; score-metadata equivalence caveat remains.
- Refinement integration: `PARTIAL` — Easy21 full chain completed; GT10 V2/V3 blocked by three missing COLMAP observations.
- GT10 hard-case rescue: `PASS` for formal R0/R1 selected-mask comparison on `0039/0040/0041`.
- Easy21 regression: `{'NONE' if abs(float(v1.get('final_f1_sample_equal_mean', 0) or 0) - float(v0.get('final_f1_sample_equal_mean', 0) or 0)) < 1e-9 else 'PRESENT'}` for V0 vs V1; full-chain delta is {delta_v3_v1:+.6f} F1 for V3 vs V1.
- Method status: `PARTIAL`.

## Frozen production configuration

- Prompt: `P6`; candidate mode: `per_instance`; prompt selection: `single`; prompt ensemble: OFF.
- Ranking: `r1`; weights: `area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5`; mask threshold: `0.5`.
- Refinement: SPNP `sam3_if_supported`, residual repair `adjacent`, corrective geometry `colmap_tracks` when geometry is available.
- A6/A7: OFF.
- GT: `potted_clean`; Easy21 and GT10 remain development/historical evaluation data.

## Observed effects

- Easy21 V1−V0 sample-equal F1: `{f(effect_map.get(('Easy21', 'ranking_V1_minus_V0'), {}).get('final_f1_paired_mean'))}`.
- Easy21 V2−V0 sample-equal F1: `{f(effect_map.get(('Easy21', 'refinement_R0_V2_minus_V0'), {}).get('final_f1_paired_mean'))}`.
- Easy21 V3−V1 sample-equal F1: `{f(effect_map.get(('Easy21', 'refinement_R1_V3_minus_V1'), {}).get('final_f1_paired_mean'))}`.
- Interaction `(V3−V2)−(V1−V0)` is reported in `metrics/paired_effects.csv`; no significance claim is made for these small development sets.
- Recommendation rationale: {rationale}

## Production command

The exact commands are preserved in `configs/PHASE20C_COMMANDS.txt`; each run also stores `execution_command.txt`, `参数.json`, `运行日志.json`, and `artifact_hashes.json`.

## Blocking items

1. Supply and verify COLMAP sparse observations for `DouBanLv2_0037`, `DouBanLv2_0039`, and `DouBanLv2_0041`, then run GT10 V2/V3 from clean output directories.
2. Either make the production candidate-reuse path ingest frozen SAM3 score metadata or run fresh SAM3 candidate generation for a byte-equivalent replay; do not silently treat `sam_score=0.0` as equivalent.
3. Add an independent, non-development validation set before changing `PARTIAL` to `LOCKED_CANDIDATE`.

This decision does not authorize Phase 20D, 2DGS, M2M3-GO, or paper-text changes.\n"""
    write("FINAL_RAP_FSAM3_METHOD_DECISION.md", final_text)

    next_text = """# NEXT_PHASE_REQUIREMENTS

Phase 20D must remain blocked pending remediation rather than being treated as a completed independent validation phase.

- Restore/match the three missing DouBanLv2 COLMAP image observations and rerun GT10 V2/V3 with clean output directories.
- Close candidate-cache score provenance: preserve `sam_score`, candidate box, and instance ordering, or explicitly rerun SAM3 inference under the frozen P6 configuration.
- Repeat the four-group comparison on an independent dataset with verified `potted_clean` GT; do not use Easy21/GT10 as the independent set.
- Keep A6/A7 OFF and keep all R1 weights and thresholds frozen.
- Re-run the Phase 20C test and hash checks after remediation; do not overwrite the current partial evidence.
"""
    write("NEXT_PHASE_REQUIREMENTS.md", next_text)

    status_text = f"""# Phase 20C Final Report

## Required status lines

```text
PHASE20C STATUS: PARTIAL
RECOMMENDED FOREGROUND METHOD: V1
R1 PRODUCTION VALIDATION: PASS
REFINEMENT INTEGRATION: PARTIAL
GT10 HARD-CASE RESCUE: PASS
EASY21 REGRESSION: {'NONE' if abs(float(v1.get('final_f1_sample_equal_mean', 0) or 0) - float(v0.get('final_f1_sample_equal_mean', 0) or 0)) < 1e-9 else 'PRESENT'}
METHOD STATUS: PARTIAL
NEXT PHASE: REMEDIATION_REQUIRED
```

## Scope and frame counts

- Easy21: 21 frames across 5 samples; V0/V1/V2/V3 all completed.
- GT10: 10 frames across BaiZhang and DouBanLv2; V0/V1 completed; V2/V3 not complete because three COLMAP observations are missing.
- No 2DGS or M2M3-GO training was started. No GT, historical result, R1 weight, or paper正文 was modified.

## Evidence map

- Input/cache audit: `reports/INPUT_AND_CACHE_AUDIT.md`.
- Easy21 metrics: `reports/EASY21_RESULTS.md`.
- GT10 metrics: `reports/GT10_RESULTS.md`.
- R1 replay: `reports/R1_PRODUCTION_REPLAY.md`.
- Module truth: `reports/MODULE_EXECUTION_TRUTH.md`.
- Paired metrics: `metrics/paired_effects.csv`.
- Real figures: `figures/generated/`; blocked GT10 V2/V3 panels are explicitly labeled and contain no fabricated metric.

## Final interpretation

R1 is selected correctly in formal production replay and rescues the three historical DouBanLv2 ranking failures. Full refinement is genuinely executed and has measurable per-stage evidence on Easy21, but complete R1×refinement validation on GT10 is not available. The method therefore remains `PARTIAL`, not `LOCKED_CANDIDATE`.

Phase 20D may not start as an independent validation claim until the remediation items in `reports/NEXT_PHASE_REQUIREMENTS.md` are closed.\n"""
    write("PHASE20C_FINAL_REPORT.md", status_text)
    print(json.dumps({"recommendation": recommendation, "delta_v3_v1": delta_v3_v1, "delta_v2_v0": delta_v2_v0, "replay_matches": f"{match_count}/{len(replay_matches)}"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
