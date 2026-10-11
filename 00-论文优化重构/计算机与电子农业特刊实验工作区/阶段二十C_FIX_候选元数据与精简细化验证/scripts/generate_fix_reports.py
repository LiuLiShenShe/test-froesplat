from __future__ import annotations

import json

from fix_common import WORK, read_csv


def table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if value not in ("", None):
                try:
                    number = float(value)
                    value = f"{number:.6f}" if field not in ("frames", "empty_count", "severe_accuracy_failure_proxy_count", "spnp_accepted_count", "spnp_changed_frames") else str(int(number))
                except (TypeError, ValueError):
                    pass
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write(name, content):
    (WORK / name).write_text(content.strip() + "\n", encoding="utf-8")


def main():
    metrics = read_csv(WORK / "V4_V5_METRICS.csv")
    pairs = read_csv(WORK / "PAIRED_METRICS.csv")
    frames = read_csv(WORK / "V4_V5_PER_FRAME.csv")
    selections = read_csv(WORK / "R0_R1_SELECTION_REPLAY.csv")
    replay = json.loads((WORK / "SCORE_REPLAY_AUDIT.json").read_text())
    decision = json.loads((WORK / "DECISION_CHECK.json").read_text())
    runtime = read_csv(WORK / "RUN_TIMES.csv")
    final_test_line = (WORK / "TEST_RESULTS.txt").read_text().strip().splitlines()[-1]
    hd95_details = []
    for dataset in ("Easy21", "GT10"):
        effect_rows = [value for value in pairs if value["dataset"] == dataset and value["effect"] == "SPNP_R1"]
        frame_effect = next(value for value in effect_rows if value["aggregation"] == "frame_level_paired_difference")
        sample_effect = next(value for value in effect_rows if value["aggregation"] == "sample_equal_paired_difference")
        hd95_details.append(f"- {dataset}: frame mean ΔHD95 = {float(frame_effect['final_hd95']):+.6f}px; sample-equal ΔHD95 = {float(sample_effect['final_hd95']):+.6f}px.")
    fields = ["group", "frames", "selected_f1", "final_f1", "final_iou", "final_hd95", "final_boundary_f1", "final_leakage", "oracle_regret", "empty_count", "severe_accuracy_failure_proxy_count", "spnp_accepted_count"]
    sections = []
    spnp_negative = []
    for dataset in ("Easy21", "GT10"):
        summary = [value for value in metrics if value["dataset"] == dataset and value.get("aggregation") == "sample_equal_mean"]
        frame_pairs = [value for value in pairs if value["dataset"] == dataset and value["aggregation"] == "frame_level_paired_difference"]
        sample_pairs = [value for value in pairs if value["dataset"] == dataset and value["aggregation"] == "sample_equal_paired_difference"]
        sections.append(f"## {dataset}\n\n### sample_equal_mean — primary aggregate\n\n{table(summary, fields)}\n\n"
                        f"### frame-level paired differences — different weights\n\n{table(frame_pairs, ['effect', 'frames', 'final_f1', 'final_iou', 'final_hd95', 'final_boundary_f1', 'final_leakage', 'positive_f1_frames', 'negative_f1_frames'])}\n\n"
                        f"### sample-equal paired differences\n\n{table(sample_pairs, ['effect', 'frames', 'final_f1', 'final_iou', 'final_hd95', 'final_boundary_f1', 'final_leakage'])}\n")
        for row in frames:
            if row["dataset"] == dataset and row["group"] == "V4" and float(row["final_f1"]) < float(row["selected_f1"]):
                spnp_negative.append({"dataset": dataset, "sample": row["sample"], "frame": row["frame"],
                                      "selected_f1": row["selected_f1"], "final_f1": row["final_f1"],
                                      "delta_f1": float(row["final_f1"]) - float(row["selected_f1"]), "final_leakage": row["final_leakage"]})
    hard = [value for value in frames if value["sample"] == "DouBanLv2" and value["frame"] in ("0039", "0040", "0041")]
    hard_table = table(hard, ["frame", "group", "selected_candidate_id", "selected_f1", "final_f1", "oracle_regret", "spnp_accepted", "hard_case_target_identity_preserved"])
    acceptance_table = table([value for value in metrics if value.get("aggregation") == "frame_equal_mean" and value["group"] in ("V4", "V5")],
                             ["dataset", "group", "frames", "spnp_acceptance_rate", "spnp_changed_frames", "changed_pixel_ratio_per_image"])
    write("R0_R1_REVALIDATION.md", f"""# R0/R1 score revalidation

All 55 candidates restore independently matched original archived confidences and boxes.
Maximum historical total-score difference: {replay['maximum_historical_total_error']:.12g};
explicit archive tolerance: {replay['tolerance']}. All scores include real q_sam.
`SCORE_EQUIVALENT` is numerical archived-precision equivalence, not unrounded tensor bytes.
`ROUNDING_SELECTION_CERTIFICATE.csv` separately certifies 62/62 historical R0/R1
selections for every confidence consistent with six-decimal serialization; no confidence
value is inferred. No Phase20C selection changes after restoration.

Easy21: R1 leaves all 21 correct selections unchanged. GT10: R0 still has three
severe wrong-target selections; R1 selects original correct candidates 0039_01,
0040_00, 0041_03, with zero oracle regret. This is DEV/regression, not independent TEST.

{hard_table}

Selected candidate ID, primary, risk penalty, q_sam, F1/IoU, oracle regret and change
reason for historical R0, Phase17C offline R1, Phase20C zero-score R0/R1 and restored
R0/R1 are all in `R0_R1_SELECTION_REPLAY.csv`. Per-candidate counterparts and hashes
are in `CANDIDATE_SCORE_REPLAY.csv`. New V0/V1 production masks are validated against
the real raw masks, not just against labels in a CSV.

Sources: this delivery's replay CSVs and small archived production logs; frozen
Phase12/P17A original metadata and Phase17C benchmark. ranking_policy.py is unchanged.
""")
    write("SPNP_ONLY_ABLATION_REPORT.md", f"""# SPNP-only ablation report

V4 = P6/per_instance/single/R1/SPNP ON/Residual OFF/GCFC OFF.
V5 differs only in ranking_policy=base. Both use the original candidate masks,
real archived confidences, RGB/GT/checkpoint/weights/AMP/preprocessing, and independent
output directories. `FROZEN_PARAMETER_AUDIT.csv` verifies every non-factor setting.
Two-frame smoke tests run on CaoMei1_0000 and DouBanLv2_0039 for both groups before
the complete runs. V5 deliberately keeps R0's wrong instance; its failed target is
not hidden or treated as a technical smoke failure. The smoke checks execution,
identity consistency and nonempty same-resolution masks; V4 also retains its target.

{''.join(sections)}

## Hard cases and attribution

{hard_table}

## Every negative V4 F1 frame (not excluded)

{table(spnp_negative, ['dataset', 'sample', 'frame', 'selected_f1', 'final_f1', 'delta_f1', 'final_leakage'])}

V4-V1 isolates SPNP under R1; V5-V0 isolates SPNP under R0; V4-V5 isolates ranking
under fixed SPNP. Do not attribute all V4-V0 gain to SPNP. Selected F1 and oracle
regret refer to pre-refinement candidates; final F1 is a different measurement.
SPNP acceptance comes from the real production `SPNP接受` field, independently
of nonzero pixel change. Changed-pixel ratio uses full image area; a separate
selected-mask-normalized ratio is also saved. Severe F1<0.5 is an accuracy proxy,
not a new manual identity annotation. Hard-case target IDs are known original
candidate identities and validated against frozen GT before and after refinement.

## Acceptance / changed pixels (real production decisions)

{acceptance_table}

V5 GT10 accepts only 7/10 SPNP refinements; the three wrong-target R0 frames
remain failures rather than being rescued by SPNP. Guarded rejection is logged,
not counted as accepted merely because the module is enabled.

## HD95 aggregation is not interchangeable

Easy21 has sample sizes 5/5/5/5/1. sample_equal_mean weights every sample at 1/5,
while a frame-level mean gives the single-frame XianKeLai1 sample 1/21.
An improvement in frame-level mean ΔHD95 does not override sample-equal worsening.
GT10 has equal 5/5 frame counts, so these two means coincide. Both values and
per-frame deltas are retained; no favorable metric alone determines the method.

{chr(10).join(hd95_details)}

具体权重解释：XianKeLai1 只有 0000 一帧，其 ΔHD95 为 +39.049294px；
sample-equal 中权重 1/5，frame mean 中权重 1/21。这一距离指标损失在两种
统计权重下贡献不同，因此 Easy21 出现逐帧均值改善但 sample-equal 退化。
这不是通过更换统计口径证明方法变好；主决策保持预设 sample-equal 权重。

## Full-chain historical comparison

V3_HISTORY is the saved Phase20C Easy21 chain, not a new restored-score V3 run.
Its zero-score input ranking is not labelled SCORE_EQUIVALENT. Actual original
RGB, candidate identities, selected mask pixels and SPNP parameters match; exact
historical SPNP-mask equality is recorded per new frame. These allow a matched
saved-mask performance comparison, but not a claim that historical scoring or
GCFC coordinate mapping was valid. GT10 has no full V3 group; no fabricated
seven-frame or full-chain result is substituted. Geometry audit explains absence.
Historical residual masks were unchanged; GCFC had mostly negative adjacent-stage
F1 effects on Easy21. No stable cross-dataset GCFC benefit is established.

## Runtime

{table(runtime, ['dataset', 'group', 'wall_sec_including_load_and_io', 'pipeline_elapsed_sec', 'gpu_used'])}

GPU timings are real production wall/pipeline times including model startup, scoring,
mask I/O and shared-machine load, not isolated CUDA kernel times. V0/V1 are CPU replay;
their wall time is not directly comparable as GPU compute cost. SPNP adds a SAM3
checkpoint load and per-frame text/geometric prompt calls; V1 requires no SAM3 GPU
when verified candidates are already cached.

Sources: V4/V5 real run logs, `V4_V5_PER_FRAME.csv`, `V4_V5_METRICS.csv`,
`FRAME_PAIRED_DIFFERENCES.csv`, archived Phase20C masks and `RUN_TIMES.csv`.
""")
    method = decision["recommended_method"]
    if method != "V1":
        raise RuntimeError("Observed rule chose a different method; revise method-specific prose before publishing")
    write("FINAL_FOREGROUND_METHOD_DECISION.md", f"""# Final foreground method decision

RECOMMENDED METHOD: {method}

The conservative multi-metric decision rule was written in `IMPLEMENTATION_PLAN.md`
before new GPU results. It requires both datasets to preserve identities and avoid
sample-equal regressions in F1, IoU, Boundary-F1, HD95 and Leakage. Observed V4
failed conditions: {', '.join(decision['V4_failed_conditions']) or 'none'}.
No GT-dependent runtime switch, retuned weight or post-hoc threshold is introduced.

1. **Ranking:** R1 still rescues all three hard frames, Easy21 churn remains zero.
   Real archive-score replay plus rounding-bound selection certificate support this;
   unsaved confidence digits are not pretended to be available.
2. **SPNP:** real V4/V5 runs isolate its increment. It can improve overlap/boundary
   while worsening leakage/HD95 or some frames. These gains and losses are shown
   together in `SPNP_ONLY_ABLATION_REPORT.md`; a small F1 gain is not a universal win.
3. **GCFC:** missing GT10 registration plus historical coordinate mismatch and mostly
   negative Easy21 F1 delta do not establish stable benefit. It is OPTIONAL, not default.
4. **Residual:** no actual saved-mask change in historical Easy21; necessity unproven.
5. **Dependencies:** V1 needs no unverified geometric observations or auxiliary module.
   It fixes ranking at frozen R1 with SPNP/Residual/GCFC OFF. Existing modules stay
   available as optional/historical ablations; no CLI default is silently rewritten.
6. **Cost:** V4 needs loaded SAM3 and extra prompt inference. Simpler cached V1 avoids
   that dependency. This decision is DEV-supported, not independently generalized yet.

Recommended frozen foreground recipe: P6, per_instance, single, ranking_policy=r1,
fixed original score weights and threshold, SPNP=OFF, Residual=OFF, GCFC=OFF,
A6/A7=OFF, verified metadata or an explicitly new SAM3 candidate set. Independent
TEST32 must confirm the frozen method without tuning. Historical stages and paper
正文 are unchanged; no 2DGS/M2M3-GO training is launched.

Sources: predeclared rule, `DECISION_CHECK.json`, paired/sample-equal metrics,
geometry audit and archive-score/identity certificates in this directory.
""")
    write("PHASE20C_FIX_FINAL_REPORT.md", f"""# Phase20C-FIX final report

```text
PHASE20C_FIX STATUS: PARTIAL
CANDIDATE SCORE PROVENANCE: PARTIAL
R1 HARD-CASE RESCUE: PASS
SPNP-ONLY VALIDATION: PASS
RECOMMENDED METHOD: {method}
GCFC STATUS: OPTIONAL
READY FOR INDEPENDENT TEST: NO
NEXT PHASE: REMEDIATION_REQUIRED
```

PARTIAL is deliberate: real original confidences are recovered at archived six-decimal
precision for 55/55 candidates, but pre-serialization tensor digits were not preserved;
the three GT10 COLMAP frames are genuinely unregistered; TEST32 human GT is absent.
SPNP-ONLY PASS means both real groups completed and were evaluated, not that every
refinement metric improved. No missing geometry or GT is synthesized.

## Completed work

- Audited Phase20C commit 970082da, clean main start, production loader/call path,
  all frozen parameters, candidate format, metadata, original masks and source RGB.
- Minimal backward-compatible metadata fix with strict mode and identity/hash checks;
  frozen R1 source SHA256 c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905.
- 55 matched candidates / 31 frames; historical score error ≤ {replay['maximum_historical_total_error']:.12g}.
  All 62 R0/R1 selections certified even under any rounding-consistent confidence.
- New CPU V0/V1 production runs on 31 frames each; four GPU smoke frames plus
  V4/V5 on 31 frames each. All outputs isolated; no old artifact overwritten.
- Comprehensive overlap/boundary/distance/leakage, oracle, identity, acceptance,
  changed pixels, per-stage, paired and runtime evidence; no cherry-picked final metric.
- Verified missing registration in both original/distorted models; coordinate caveat
  for historical GCFC reported; no COLMAP reconstruction or core algorithm change.
- TEST32 package: 32 RGB / 8 disjoint samples / 71 complete original candidates,
  no verified GT; human labels and distinct second review remain user work.
- Tests: `{final_test_line}`; expected annotation gate blocks before test accuracy.

{''.join(sections)}

## Decision and remaining work

Recommend {method}; see `FINAL_FOREGROUND_METHOD_DECISION.md` for multi-metric reasons
and optional-module policy. The archive-precision limit is documented, not repaired
with guessed confidence or changed ranking. Independent validation can proceed only
after human target/plant/pot/cube annotations and manual difficulty/second review.
Those actions are in `PHASE20D_INDEPENDENT_VALIDATION_PLAN.md`. Do not retune on TEST32.
No claim of independent validation, full GT10 V3, universal refinement effectiveness,
paper rewrite, 2DGS or M2M3-GO experiment is made.

## Evidence and reproducibility

Requested deliverables are in this directory. `small_run_outputs/` archives true small
production parameters, candidate scores, selections, runtime and failure summaries.
`logs/` contains original stdout/stderr as tracked text; GPU masks/RGB remain local
with paths, hashes and configurations in `artifact_hashes/` and execution records.
`ARTIFACT_RECOVERY.md` explains ignored data recovery/reproduction; missing source
archives are explicitly required, not replaced with inferred masks.

The external Bocha search failed HTTP403 (quota exhausted), and the requested GitHub
fetch yielded no usable page. All experiment conclusions here come from actual local
frozen files and new runs, not training-data assertions. No secrets are in artifacts.
Final boundary: commit/push these verified small files and stop; do not enter Phase20D
or alter manuscript text. Git submission status is recorded separately.
""")


if __name__ == "__main__":
    main()
