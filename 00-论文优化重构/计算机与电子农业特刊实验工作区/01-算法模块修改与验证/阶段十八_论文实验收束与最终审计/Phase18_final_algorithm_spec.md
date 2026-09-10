# Final Algorithm Specification / 最终算法定义

> Production-code-accurate. Every parameter, default, and formula below is sourced
> from the production code at commit `d704faa8`.
> Single source of truth: `ranking_policy.py` (R1 policy) + `生成RAP-FSAM3掩膜.py`
> (candidate generation, scoring, pipeline).

---

## Component 1: SAM3 Candidate Generation

**Decision:** RETAIN

**What it is:** SAM3 text-prompted instance segmentation. Each text prompt is sent to SAM3 independently; SAM3 returns multiple object instances per prompt. In `per_instance` mode, every individual SAM3 instance mask becomes a separate `Candidate` — no OR-merge, no largest-component filter at generation time.

**Current production defaults:**
| Parameter | Default | Code reference |
|---|---|---|
| `candidate_mode` | `per_instance` | `parse_args()`, line 630 |
| `sam3_mask_threshold` | `0.5` | `parse_args()`, line 638 |
| `save_raw_instance_masks` | `True` | `parse_args()`, line 641 |
| `default_prompt_id` | `P2` | `parse_args()`, line 647 |
| `confidence_threshold` | `0.3` | `parse_args()`, line 622 |

**How it works:**
1. For each prompt (e.g. P2: "entire plant excluding pot"), SAM3 produces K instance masks with scores and bounding boxes.
2. `per_instance` mode: each instance mask → one `Candidate(prompt_id, instance_id=i, mask=m, sam_score=sc, box=box)`.
3. No OR-merge, no largest-component filter applied at generation. Minimal cleanup deferred to downstream.
4. If SAM3 returns zero instances for a prompt, a single empty-mask Candidate is emitted (scored as 0.0 downstream).

**Paper wording:** "SAM3-based per-instance candidate generation produces multiple alternative masks per image region. Each instance is scored independently without early merging."

---

## Component 2: Per-Candidate Feature Scoring (q_components)

**Decision:** RETAIN

**What it is:** Eight features are computed for every candidate: 7 structural plausibility features + 1 SAM confidence score.

**Current production defaults (frozen):**
```
score_weights = "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"
```
Frozen denominator: `denom = 1+1+1+1+1+0.5 = 5.5`.

**Feature definitions and computation (from `score_candidate()`):**

| Feature | Formula | Frozen weight | Notes |
|---|---|---|---|
| `q_area` | `area_quality(area_ratio, args)` — triangular plausibility in [area_min, area_max] | 1.0 | area_min=0.01, area_max=0.80 |
| `q_comp` | `1/(1 + max(comp_count-1, 0))` — single-component rewarded | 1.0 | comp_count via `ndimage.label` |
| `q_edge` | `edge_quality(mask, args)` — boundary-adherence score (penalizes border-hugging) | 1.0 | border_band=2px |
| `q_temp` | `temporal_iou` when `use_temporal_alignment=True`, else **fixed 0.5** | 1.0 | Default disabled → always 0.5 |
| `q_contrast` | `min(1.0, contrast/0.25)` — FG-BG L2 contrast normalized | 1.0 | **Excluded from R1 primary** (see Component 4) |
| `q_leak` | `leakage_quality(bottom_fraction, args)` — vertical overflow | **0.0** | Computed but 0-weighted in production default |
| `q_side` | `side_leakage_quality(side_fraction, args)` — side-pot intrusion | **0.0** | Computed but 0-weighted in production default |
| `sam_score` | `1/(1 + exp(-6*(sam_raw - 0.5)))` — sigmoid-normalized SAM3 confidence | 0.5 | logit-derived |

**Key note:** `q_leak` and `q_side` are computed and stored in `ScoreRecord` as diagnostics, but their frozen weights are 0.0 in the production default. They do not contribute to `base_total`. They are retained for failure diagnosis and ablation studies.

**Empty mask hard gate:** Candidates with zero-area mask return `total_score=0.0`, `primary_score=0.0`, `empty_flag=True`. No temporal-default bonus.

**Paper wording:** "Each candidate is scored on six weighted structural features: area plausibility, component count, boundary adherence, temporal consistency (disabled by default), foreground-background contrast (computed but excluded from the primary ranking), and SAM3 confidence."

---

## Component 3: Candidate Scoring Formula

**Decision:** RETAIN (as part of R1 pipeline)

**Formula (production code, `score_candidate()`, lines 1540–1672):**

```
base_total = (Σ w_i · q_i + 0.5 · q_sam) / 5.5

where Σ includes: area(1.0) + comp(1.0) + edge(1.0) + temp(1.0) + contrast(1.0)
      + leak(0.0) + side(0.0)
```

**Risk penalty (hard gate):**
```
penalty = 0.25   if vertical_coverage < 0.30   (suspected pot/strip-only mask)
           0     otherwise

vertical_coverage = 1.0 - (min_mask_row / image_height)
```

**Semantic gate:**
```
semantic_total = 0.0   (gate disabled by default: --use_semantic_gate = False)
```

**Old total (used only for diagnostic comparison, not for selection under R1):**
```
total = base_total + semantic_total - penalty
```

**Paper wording:** "The base score is a weighted sum of six features normalized by a frozen denominator of 5.5. A 0.25 penalty is applied when vertical coverage falls below 0.30 (30%), flagging masks that only capture a lower strip."

---

## Component 4: R1 Robust Instance Ranking (winner)

**Decision:** RETAIN / DEFAULT

**What it is:** R1 removes `q_contrast` from the primary ranking score. The production default `--ranking_policy r1`. Single source of truth: `ranking_policy.py:primary_and_tiebreak()`.

**Formula (`ranking_policy.py`, lines 44–48):**
```python
# r1 policy:
primary = (base_total * denom - w_contrast * q_contrast) / denom + semantic_total - penalty
# i.e. primary_score = base_total - q_contrast/5.5 - penalty   (since w_contrast=1, denom=5.5)

tie_break = 0.0   (no lexicographic tie-breaking under r1)
```

**Selection:** `argmax(primary_score, tie_break_score)` per default-prompt instance branch. When `candidate_mode=per_instance` and `prompt_selection_mode=single`, the default prompt (P2) may yield multiple instance candidates. `select_best_policy()` returns the lexicographic max on `(primary_score, tie_break_score)`.

**Why R1 was chosen (Phase 17C decision order):**
1. Failure rescue: R1 rescues all 3 identified failures (DouBanLv2_0039/0040/0041, F1 >= 0.95). R0/R2/R3 also rescue all 3.
2. Zero material DEV regressions: R1 = 0. R2/R3 = 0.
3. Zero material Easy21 regressions: R1 = 0.
4. Oracle regret = 0.0 for all policies.
5. Selection churn: 2/32 (6.2%) under R1.
6. Structural simplicity: R1 removes one term. R2 adds tie-break. R3 adds area modulation.

**Evidence (Phase 17C):**
- Failure-frame Spearman(area_ratio, q_contrast): ρ = −0.7187 (p = 0.003781) — small masks strongly correlated with high contrast
- Ablation: q_contrast removal rescues 3/3 failure frames
- Production replay: 142/142 tests PASS under R1

**q_contrast status:** REMOVED from primary ranking score. RETAINED in `ScoreRecord` as `q_contrast`, `contrast_effective` (= q_contrast × q_area), `contrast` (raw value), and `sam_scores` for diagnostic analysis. The contrast component is still useful for understanding failure mechanisms — it should not be deleted from the pipeline, only from the ranking signal.

**Paper wording:** "R1 excludes q_contrast from the primary ranking score: primary_score = base_total - q_contrast/5.5 - penalty. q_contrast remains in the feature set for diagnostic analysis."

---

## Component 5: A6 Cross-View Consensus

**Decision:** DEMOTED / DEFAULT OFF

**What it is:** Per-sample cross-view consensus voting. Builds a robust center estimate, COLMAP geometry-backed foreground support, and static-distractor detection, then refines each frame's selected mask by removing pixels lacking multi-frame consensus and recalling geometrically-supported absent regions.

**Current production default:** `--use_cross_view_consensus = False` (disabled). Must be explicitly enabled.

**Evidence:**
- Phase 14 Easy21 ablation: E_A6 = −0.0028 (negligible negative)
- Phase 17A GT10 hard-case pilot: A6 acts on 4/10 BaiZhang frames → all 4 produce F1 regression (mean ΔF1 = −0.007). 0 improvements. A6 consistently removes correct mask pixels without replacing them.
- Phase 17A A6+A7 combined: 4/4 acted frames regress (identical to A6 alone)

**Paper wording:** "A6 did not provide consistent measurable benefit under the evaluated settings. It is disabled by default."

---

## Component 6: A7 SAM3 Memory Propagation

**Decision:** DEMOTED / DEFAULT OFF

**What it is:** SAM3 video predictor memory engine. Seeds a high-scoring frame's mask into SAM3's temporal memory, then propagates bidirectionally to all other frames in the same sample. Produces an alternative set of per-frame masks.

**Current production default:** `--use_memory_propagation = False` (disabled). Must be explicitly enabled.

**Evidence:**
- Phase 14 Easy21 ablation: E_A7 ≈ 0.0000 (completely neutral)
- Phase 14.1 lifecycle fix: propagation correctness was confirmed after fixing session lifecycle (close on OOM/exception paths). Benefit remains near-neutral.
- Phase 17A GT10: 1 marginal improvement (BaiZhang, ΔF1 = +0.006), 3 neutral, 0 regressions

**Paper wording:** "A7 did not provide consistent measurable benefit under the evaluated settings. It is disabled by default."

---

## Component 7: A6 + A7 Interaction

**Decision:** DEMOTED

**What it is:** Enabling A6 and A7 together. A7 propagates masks; A6 refines them with cross-view consensus.

**Evidence:**
- Phase 14 Easy21: near-neutral combined effect
- Phase 17A GT10: 4/4 acted frames regress (identical pattern to A6 alone)
- 1 material regression on Easy21 when A6+A7 combined

**Paper wording:** "The combined A6+A7 configuration did not provide consistent improvement and is not retained."

---

## Method Flow Diagram (Production Default)

The following flow reflects the **actual production defaults** at commit `d704faa8`:

```
Input RGB image
    ↓
SAM3 multi-instance candidate generation
    (per_instance mode, mask_threshold=0.5, multiple candidates per prompt)
    ↓
Per-candidate feature extraction
    (q_area, q_comp, q_edge, q_temp=0.5, q_contrast,
     q_leak [weight=0], q_side [weight=0], sam_score)
    ↓
Candidate scoring (denom = 5.5, frozen)
    base_total = (Σ w_i·q_i + 0.5·q_sam) / 5.5
    penalty = 0.25 if vertical_coverage < 0.30
    semantic_total = 0 (gate disabled)
    ↓
R1 robust instance ranking
    primary_score = base_total - q_contrast/5.5 - penalty
    (q_contrast excluded from primary; retained for diagnostics)
    ↓
Final mask selection (lexicographic argmax per default prompt branch)
    ↓
Output segmentation mask

[Optional — not in production default; enabled by explicit flags only]
A5c: SPNP refinement + residual repair + corrective geometry
A6:  cross-view consensus
A7:  SAM3 memory propagation
```

---

## Appendix: Production Default Parameters Reference

| Parameter | Value | Source |
|---|---|---|
| `candidate_mode` | `per_instance` | `parse_args()` |
| `sam3_mask_threshold` | `0.5` | `parse_args()` |
| `save_raw_instance_masks` | `True` | `parse_args()` |
| `default_prompt_id` | `P2` | `parse_args()` |
| `score_weights` | `area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5` | `parse_args()` |
| `ranking_policy` | `r1` | `parse_args()` |
| `base_cleanup_mode` | `fsam3_basic` | `parse_args()` (no-op for per_instance) |
| `vertical_coverage_min_ratio` | `0.30` | `parse_args()` |
| `use_temporal_alignment` | `False` | `parse_args()` (store_true) |
| `use_semantic_gate` | `False` | `parse_args()` (store_true) |
| `use_cross_view_consensus` | `False` | `parse_args()` (store_true) |
| `use_memory_propagation` | `False` | `parse_args()` (store_true) |
| `use_spnp_refinement` | `False` | `parse_args()` (store_true) |
| `use_residual_repair` | `False` | `parse_args()` (store_true) |
| `use_corrective_geometry` | `False` | `parse_args()` (store_true) |
| `confidence_threshold` | `0.3` | `parse_args()` |
| `area_min_ratio` | `0.01` | `parse_args()` |
| `area_max_ratio` | `0.80` | `parse_args()` |
| `leakage_max_bottom_fraction` | `0.02` | `parse_args()` |
| `leakage_max_side_fraction` | `0.004` | `parse_args()` |
| `spnp_min_refined_iou` | `0.65` | `parse_args()` |
| Git SHA | `d704faa8` | Phase 17C final commit |
