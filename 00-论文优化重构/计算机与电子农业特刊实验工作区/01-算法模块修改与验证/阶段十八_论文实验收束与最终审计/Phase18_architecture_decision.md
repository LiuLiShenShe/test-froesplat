# Architecture Decision Record

> Phase 18 — paper-consolidation architecture decisions for the RAP-FSAM3
> plant segmentation pipeline. Each decision is grounded in a specific phase,
> commit, and quantitative evidence. Cross-reference:
> `Phase18_final_algorithm_spec.md` (production-code-accurate definition).

---

## Component: Candidate Generation (per_instance)

**Decision:** RETAIN

**Reason:** Phase17B showed the SAM3 per-instance candidate generator produces high-quality whole-plant masks even on the previously catastrophic hard frames (raw oracle F1 0.9867/0.9882/0.9865 on DouBanLv2_0039/0040/0041). Generation was exonerated as a failure source; the failure was entirely in candidate ranking (C3_RANKING_FAILURE). The `per_instance` mode with `sam3_mask_threshold=0.5` and `save_raw_instance_masks=True` is the frozen production default.

**Paper wording:** "SAM3-based per-instance candidate generation produces multiple alternative masks per image region."

---

## Component: P6 Prompt

**Decision:** RETAIN

**Reason:** The P6 semantic prompt ("whole potted plant including the pot") is part of the production prompt set and produced the highest-oracle candidates on the hard frames (Phase17B per-prompt oracle audit: all three DouBanLv2 failure frames had P6 raw oracle ≥ 0.9865). The prompt succeeds whenever the subsequent ranking selects the correct instance — the failure was ranking, not the prompt.

**Paper wording:** "The P6 prompt generates a reliable whole-potted-plant candidate; selection failures observed in Phase 17A/17B were traced to the candidate-ranking function, not to prompt-based generation."

---

## Component: R1 Robust Instance Ranking

**Decision:** RETAIN / DEFAULT

**Reason:** Phase17C compared four ranking policies (base/r1/r2/r3) by the fixed decision order (failure rescue → zero material regressions → oracle regret → churn → simplicity). R1 rescues 3/3 identified DEV failures (F1 ≥ 0.95), produces zero material regressions on Easy21, has oracle regret 0.0, and is the structurally simplest intervention (single term removed). `--ranking_policy r1` is the production default (`ranking_policy.py`, single source of truth).

**Paper wording:** "R1, which removes q_contrast from the primary ranking score, was selected as the production ranking policy."

---

## Component: q_contrast Primary Contribution

**Decision:** REMOVE FROM PRIMARY (retain as diagnostic)

**Reason:** Phase17C contrast-bias audit found Spearman ρ = −0.7187 (area_ratio, q_contrast) on the failure frames with p = 0.003781 — small distractor masks strongly saturate q_contrast and win the old base ranking. Removing q_contrast from the primary score rescues 3/3 dev failures with zero regressions. The feature is retained in `ScoreRecord` (fields `q_contrast`, `contrast`, `contrast_effective`) for failure-diagnosis and ablation evidence, but no longer influences selection.

**Paper wording:** "q_contrast is computed and reported for diagnostic analysis but excluded from the primary ranking score."

---

## Component: A6 Cross-View Consensus

**Decision:** DEMOTE / DEFAULT OFF

**Reason:** Phase14 factorial ablation on Easy21 measured E_A6 = −0.0028 (negligible negative); Phase17A on GT10 showed A6 acting on 4/4 BaiZhang frames and producing F1 regression in every one (mean ΔF1 = −0.007, 0 improvements). A6 provided no consistent measurable benefit under the evaluated settings, so `--use_cross_view_consensus` stays False by default and A6 is reported only as a negative/ablation result.

**Paper wording:** "A6 did not provide consistent measurable benefit under the evaluated settings. It is disabled by default."

---

## Component: A7 SAM3 Memory Propagation

**Decision:** DEMOTE / DEFAULT OFF

**Reason:** Phase14.1 (after lifecycle correction) measured E_A7 ≈ 0.0000 on Easy21 — neutral. Phase17A on GT10 found only 1 marginal improvement (BaiZhang, ΔF1 = +0.006) with 3 neutral and no regressions. Even with corrected FSA lifecycle, memory-propagation benefit remains near-neutral, so `--use_memory_propagation` stays False by default.

**Paper wording:** "A7 did not provide consistent measurable benefit under the evaluated settings. It is disabled by default."

---

## Component: A6 + A7 Interaction

**Decision:** DEMOTED

**Reason:** Phase14 showed the combined A6+A7 variant was near-neutral on Easy21 with a material regression (ChangShouHua2_0075, ΔF1 = −0.058), and Phase17A showed the combined configuration regressed on the same 4/4 frames as A6 alone. There is no evidence of interactive benefit, so the combined branch is not retained in the production flow.

**Paper wording:** "The combined A6+A7 configuration did not provide consistent measurable benefit and is not retained."

---

## Summary Table

| Component | Decision | Phase evidence |
|---|---|---|
| Candidate generation | RETAIN | Phase17B: oracle F1 ≈ 0.987 (0.9867/0.9882/0.9865) |
| P6 prompt | RETAIN | Phase17B: prompt succeeds when ranked correctly |
| R1 ranking | DEFAULT | Phase17C: 3/3 rescue, 0 regressions |
| q_contrast in primary | REMOVE | Phase17C: Spearman −0.7187 in failures (p=0.0038) |
| A6 | DEFAULT OFF | Phase14: E_A6 = −0.0028; Phase17A: 4/4 regressions |
| A7 | DEFAULT OFF | Phase14.1: E_A7 ≈ 0; Phase17A: 1 marginal improvement |
| A6+A7 | DEMOTED | Phase14: 1 material regression; Phase17A: 4/4 regress |