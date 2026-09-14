# Phase 19A — Method Component Audit: ForeSplat.tex vs Production (A6/A7/memory/ranking)

Audit date: 2026-09-14. Source: `ForeSplat.tex` (850 lines, CAS-SC), `ForeSplat-highlights.tex`.
Production ground truth: Phase 17A (A6/A7 disabled, commit `d0a9970b`), Phase 17C (R1 ranking winner, q_contrast excluded from primary score, `--ranking_policy r1`, commit `d704faa8`), Phase 12 (frozen P6 metrics, 21-frame annotated set), Phase 18 evidence registry.

## 0. Headline findings

1. **G11 gate satisfied (A6/A7):** A6 (cross-view consensus) and A7 (memory propagation / temporal memory) are **NOT described as core method stages anywhere in ForeSplat.tex**. The demotion happened before this manuscript was written; the text contains no cross-view-consensus stage and no temporal-memory stage in the Method chapter. The only related trace is `q_temp` (temporal consistency) inside the (OLD) semantic-gated descriptor in §2.3.3 and a *future-work* mention of "SAM2/3 temporal propagation" in §4.5 (Discussion, explicitly future). **Production status (A6 OFF, A7 OFF) and manuscript Method chapter are therefore CONSISTENT — no rewrite needed on this axis.**

2. **G11 gate VIOLATED (ranking):** §2.3.3 (Semantic-Gated Multi-Prompt Selection) describes the **OLD scoring policy**: the primary foreground-prior descriptor includes both `q_contrast` (foreground-background appearance separability) and `q_temp` (temporal consistency), and the gated confidence score `c_i^(k)` drives selection. Production R1 robust instance ranking **EXCLUDES q_contrast from the primary score** (Spearman contrast bias −0.2927 full / −0.7187 on failure frames; R1 rescues DouBanLv2 0039/0040/0041 from F1≈0 to ≥0.95). The manuscript's method description therefore does **not** match the production default. **This is the single required Method-chapter correction.** The word "robust" appears in "R1 robust instance ranking" as a policy NAME (acceptable), but the manuscript never mentions the R1 policy or its q_contrast exclusion.

3. **Per-instance candidate generation:** production generates candidates per plant instance; §2.3.2 describes 5 fixed prompts with K=5 candidates. Not a contradiction (the 5 prompts remain the default prompt list / P2 prompt_list), but the text should say "per instance" and align the prompt list with the production `prompt_list` (P2) default. Minor edit.

4. **A5c refinement:** the geometry-consistency foreground correction in §2.3.5 is the manuscript's match to the production A5c-family refinement behaviour. It is present and described accurately. No change needed beyond possibly naming it A5c for traceability.

5. **Terminology check:** the manuscript never uses the tokens "P2", "P6", "prompt_list", "ranking_policy", "R1", "A5c", "A6", or "A7". It uses "five plant-related prompts" (≈ P2 prompt_list default), "manually annotated evaluation frames" (≈ the Phase 12 frozen P6 evaluation set — should be named to avoid confusion with the later Phase 16 hard-case GT set), and "single-prompt SAM3 prior". No P2/P6 confusion exists in the text, but the Phase 19B rewrite should use the frozen-set vocabulary ("frozen P6 evaluation set") explicitly.

## 1. Component-by-component table

| Component | Mentioned in tex (Y/N + line) | Described role in tex | Production status | Classification | Reason |
|---|---|---|---|---|---|
| A6 cross-view consensus | N (no text in Method; searched: no consensus/min-frames/cross-view stage) | Absent; not described | Demoted, default OFF (Phase 17A); Track B showed it is architecturally a no-op (consensus_min_frames=5 vs 4-frame TEST samples); E_A6=−0.0075 on 10-GT pilot | NOT_IN_MANUSCRIPT | Text and production agree; do NOT add A6 to the main Method. Any mention must stay in Discussion/ablation history only. |
| A7 memory propagation / temporal memory | N (Method); future-work only at line 820 (""future systems could combine ... SAM2/3 temporal propagation"") | Absent in Method; explicitly future in Discussion | Demoted, default OFF (Phase 17A); E_A7=+0.0005 (near-neutral) on 10-GT pilot; U01/U11 acted on 1/32 TEST frames | NOT_IN_MANUSCRIPT (MOVE_TO_DISCUSSION is already satisfied) | Correct treatment: A7 described only as future work. Production default OFF. No Method change. |
| q_temp temporal-consistency term | Y — line 205 (foreground-prior descriptor φ, 6th element) | Described as part of the OLD semantic-gated selection score | Not part of production R1 primary score; production is single-frame per-instance ranking | KEEP_AS_OPTIONAL_EXPERIMENT (or DELETE) | Describes the old scoring policy; if §2.3.3 is rewritten to R1, q_temp (and q_contrast) move to the ablation/supplementary description of the superseded selector. |
| q_contrast term in primary score | Y — line 205 (φ descriptor, 5th element) | Described as part of the primary semantic-compatibility score | **EXCLUDED from R1 primary score** (Phase 17C); Spearman contrast bias −0.2927 full / −0.7187 failure frames; R1 rescues 0039/0040/0041 | MOVE_TO_ABLATION | This is the critical mismatch. The manuscript must be rewritten so the primary score excludes q_contrast, or explicitly described as the superseded selector with R1 as production. |
| R1 robust instance ranking | N (never mentioned) | Absent | Production default `--ranking_policy r1` (Phase 17C, commit d704faa8) | NEW: must be ADDED to Method §2.3 and Results §3.3 | The production ranking policy is missing entirely from the manuscript. Add to Method (definition: primary score excludes q_contrast) and Results (rescue evidence). |
| Per-instance candidate generation | Partial — line 191 (5 fixed prompts, K=5) | Described as fixed prompt set per frame | Production generates candidates per plant instance (P2 default prompt_list) | KEEP_WITH_MINOR_EDIT | Align wording to "per-instance candidate generation with the production default prompt list". |
| Semantic-gated multi-prompt selection (SG-MPS) | Y — lines 199-240, Table 5 | Core RAP-FSAM3 stage; ablation shows geometry-only selector fails (F1 0.8429) | Superseded selector family; production uses R1 ranking, not softmax gating | MOVE_TO_ABLATION (mechanism) + REWRITE (production description) | The softmax-gating mechanism is real and is the diagnostic basis of the ranking-failure analysis (geometry-only failure = ranking failure). Keep as ablation/diagnostic; production primary score is R1. |
| Structure-balanced foreground-background prompt refinement (SB-FBPR) | Y — lines 242-273, Table 5 | Core RAP-FSAM3 stage; positive/negative prompt sampling | Consistent with production refinement (A5c-family); no conflict | KEEP | No mismatch. |
| Geometry-consistency foreground correction (GCFC / A5c-family) | Y — lines 275-298, Table 5 | Core RAP-FSAM3 stage; conservative boundary-only correction | Consistent with production A5c refinement behaviour | KEEP | No mismatch; optionally name it A5c for traceability. |
| FFT frame-quality filtering | Y — lines 177-187 | First pipeline stage; removes blurred frames | Consistent; no conflict | KEEP | No mismatch. |
| Foreground-track initialisation (multiview mask consistency) | Y — lines 313-332 | Filters COLMAP sparse points by mask consistency (τ_track=0.9, ≥3 views) | Consistent with production foreground initialisation | KEEP | Note: "multiview mask consistency" here is point-level track filtering, NOT the A6 cross-view-consensus mechanism; keep terminology distinct in the rewrite. |
| Foreground RGB supervision + alpha/bg opacity constraints | Y — lines 334-391 | Training-time foreground objective (λ_mask=0.08, λ_bg=0.02) | Consistent with production | KEEP | No mismatch. |
| View-quality-aware soft loss weighting | Y — lines 393-410 | Per-view weights [0.6,1.0], rgb_only | Consistent with production | KEEP | No mismatch. |
| Mask-guided multi-cue Gaussian pruning | Y — lines 412-437 | Post-training pruning controller (pruning_mode=mask) | Consistent with production | KEEP | No mismatch. |
| M2M3-GO (mask-to-model / model-to-mask micro-matching) | Y — lines 439-547 | Lightweight primitive-level refinement layer; Floor40 optional safeguard | Consistent with production component | KEEP | Framing as "refinement layer, not standalone compression module" matches Table 5 evidence (modest effect). |
| Mesh extraction / TSDF + scale recovery / phenotyping | Y — lines 550-561 | Mesh export, pot-diameter scale recovery, 4 traits | Consistent with production | KEEP | No mismatch. |
| "cross-species" reconstruction claims | N (no cross-species claim in tex) | Absent | Phase 18: cross-species claims limited (3 samples, descriptive only) | NOT_IN_MANUSCRIPT | Do NOT add cross-species claims; any future claim stays descriptive on the existing 3-sample evidence. |

## 2. Verdicts

- **G11 (A6/A7 absent from main Method): PASSED.** A6/A7 are not described as core method stages in ForeSplat.tex. No method-chapter action needed for A6/A7. The manuscript's Discussion (§4.5) already restricts temporal propagation to future work, which is the correct treatment.
- **G11 (Method matches production ranking): FAILED.** §2.3.3 describes the OLD scoring with q_contrast in the primary descriptor. The production R1 policy (q_contrast excluded) is absent from the manuscript. **REQUIRED REWRITE** of §2.3.3 (and correspondingly §3.3/Abstract/Conclusion), plus a new ranking-failure analysis section (Phase 17C evidence).
- **Terminology:** no P2/P6/prompt_list confusion present, but the rewrite should (a) name the production default prompt list (P2 prompt_list) for §2.3.2, (b) name the frozen Phase 12 evaluation set ("P6") where the manuscript says "manually annotated evaluation frames", and (c) not introduce A6/A7 stage language.

## 3. Recommended classification summary

| Classification | Components |
|---|---|
| KEEP (production-consistent) | FFT filtering; SB-FBPR; GCFC/A5c; foreground-track init; RGB/opacity constraints; view-quality weighting; multi-cue pruning; M2M3-GO; mesh/TSDF/scale recovery; phenotyping |
| REWRITE (production mismatch) | Semantic-gated selection §2.3.3 (add R1; exclude q_contrast from primary score) |
| MOVE_TO_ABLATION | Old softmax-gated selector (incl. q_contrast/q_temp terms) as superseded/diagnostic variant; keep Table 5 geometry-only failure row as the ranking-failure diagnostic |
| ADD TO METHOD (missing) | R1 robust instance ranking definition (primary score excludes q_contrast) |
| NOT_IN_MANUSCRIPT (correct) | A6 cross-view consensus; A7 memory propagation; cross-species claims |
