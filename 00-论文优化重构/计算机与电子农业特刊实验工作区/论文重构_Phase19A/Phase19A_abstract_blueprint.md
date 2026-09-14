# Phase 19A — Abstract Blueprint (ForeSplat Rewrite)

**Generated:** 2026-09-14 | **Target length:** 150–250 words | **Slot order:** fixed (1→6 below)
**Source inputs:** `Phase19A_claim_evidence_matrix.csv`, `Phase19A_claim_audit.csv`, `Phase19A_resolution_status.md`

---

## Fixed slot order

| # | Slot | Content | Evidence anchors |
|---|---|---|---|
| 1 | Background | Agricultural phenotyping need + low-cost constraint | P05/P07 context |
| 2 | Problem | Direct VFM prompting is fragile in potted-plant scenes; a second failure mode is *ranking*, not generation | P03, C01, C02 |
| 3 | Method | RAP-FSAM3 per-instance candidate generation + R1 robust ranking + ForeSplat foreground-object 2DGS + phenotyping | C04, P01, P04, P05, P06 |
| 4 | Evidence | Segmentation F1 = 0.9706 on frozen P6; boundary/leakage gains; R1 rescues 3 catastrophic failures to F1 ≥ 0.95 with 0 regressions; PSNR = 31.09 dB; trait R² up to 0.99 "comparable to" manual | P01, P02, C04, P05, P07 |
| 5 | Downstream significance | Low-cost 3D phenotyping; CompAg ag-application framing | P07, P05 |
| 6 | Constraint note | Limitations-safe wording carried into the abstract | whole claim_audit |

---

## Draft abstract (210 words, compliant)

> Low-cost, image-based 3D plant phenotyping is important for agricultural research, yet accurate foreground extraction and three-dimensional reconstruction remain difficult under realistic acquisition conditions. Vision foundation models (VFMs) offer flexible segmentation, but direct prompting is semantically fragile in potted-plant scenes: on the same frames, prompt wording changed F1 from 0.08 to 0.97. A second, less obvious failure mode is selection rather than generation. On hard frames, high-quality whole-plant candidate masks (oracle F1 = 0.987–0.988) were present, yet a contrast-based scoring signal ranked small, high-contrast distractors above them and collapsed the final F1 to 0. To address both problems we propose RAP-FSAM3, a reconstruction-aware prompt-adaptation pipeline that generates and scores per-instance candidate masks, together with R1, a robust instance-ranking policy that removes the contrast signal from the primary score. R1 rescued all identified catastrophic failures to F1 ≥ 0.95 with zero material regressions, and on the frozen 21-frame evaluation set RAP-FSAM3 achieved F1 = 0.9706 with gains concentrated in boundary consistency and leakage suppression. The selected masks drive ForeSplat, a foreground-object 2D Gaussian Splatting reconstruction that reached PSNR = 31.09 dB across 20 sequences. Mesh-based trait measurements were comparable to manual measurements (R² up to 0.99). The workflow offers a low-cost route to 3D plant phenotyping, with its validation scope stated explicitly.

---

## Slot-by-slot constraints

### Slot 1 — Background
- Must convey: agricultural phenotyping need + low-cost acquisition constraint (smartphone/turntable context is fine to imply, not to detail).
- Allowed: "low-cost, image-based 3D plant phenotyping"; "important for agricultural research".
- Forbidden: "the demand for high-throughput phenotyping is urgent" type unverifiable urgency claims; no numbers here.

### Slot 2 — Problem
- Must convey both failure modes: (a) prompt fragility (P03 — per-prompt F1 spread 0.08–0.97), and (b) ranking failure on hard frames (C01/C02 — high-quality mask present, wrong candidate selected, final F1 → 0).
- Allowed: "prompt wording changed F1 from 0.08 to 0.97"; "selection rather than generation"; "high-quality whole-plant candidate masks (oracle F1 = 0.987–0.988) were present".
- Forbidden: "P6 failed"; "the prompt could not detect the plant"; "candidate generation failed"; "segmentation backbone failed".

### Slot 3 — Method
- Must name: RAP-FSAM3 (per-instance candidate generation + scoring), R1 (robust instance ranking), ForeSplat (foreground-object 2D Gaussian Splatting), phenotyping (mesh-based trait measurement).
- "robust instance-ranking policy" — "robust" is acceptable here only as the R1 policy name (claim_audit row 12).
- Forbidden: A6/A7 must NOT appear in the abstract (no A6/A7 claim); "semantic-gated" detail belongs to Methods, not the abstract.

### Slot 4 — Evidence
- Must include: segmentation F1 = 0.9706 on frozen 21-frame set; gains concentrated in boundary consistency and leakage suppression (NOT "outperforming SAM3"); R1 rescues 3 identified catastrophic failures to F1 ≥ 0.95 with 0 material regressions; PSNR = 31.09 dB (20 sequences); trait R² up to 0.99 with "comparable to" wording.
- R² values: 2 decimal places (A2-C01) — "0.99" not "0.988"; abstract need not list all four traits.
- Allowed: "rescued all identified catastrophic failures to F1 ≥ 0.95 with zero material regressions" (C04 SUPPORTED).
- Forbidden: "significantly"; "outperforming single-prompt SAM3" (P02 — must be replaced by the specific boundary/leakage delta phrasing); "significantly improves"; "strong agreement".

### Slot 5 — Downstream significance
- Must close with CompAg ag-application framing: low-cost route to 3D plant phenotyping.
- Allowed: "offers a low-cost route to 3D plant phenotyping".
- Forbidden: "enables high-throughput field phenotyping at scale" (no field evidence); "generalizes to other crops".

### Slot 6 — Constraint note (not a written sentence — a framing requirement)
- The abstract must carry a limitations-safe tail that signals scope. Recommended: "with its validation scope stated explicitly" or "within the evaluated acquisition settings". This pre-empts reviewer B-EIC-C01 (cross-species) and B-R2-C02 (external baselines) concerns at first contact.

---

## Standing constraint list (applies to the whole abstract)

1. **No "significant"** — neither alone nor as "statistically significant" (no test exists; A2-C01/claim_audit row 16).
2. **No "outperforms"** — replace with the specific metric deltas (P02 allowed wording).
3. **No "generalizes" / "generalization"** — no independent held-out evidence (claim_audit row 6).
4. **No A6/A7 claim** — A6/A7 must not appear anywhere in the abstract (P08; G11).
5. **No "state-of-the-art" / "SOTA"** (claim_audit row 8).
6. **No "strong agreement"** — use "comparable to" (claim_audit row 9).
7. **No "cross-species" reconstruction claim** — B-EIC-C01 PARTIAL (3-sample reconstruction); "cross-species applicability" may appear only if scoped to mask/phenotype levels and carried into a limitation.
8. **R² at 2 decimal places** in the abstract (A2-C01).
9. **Word count 150–250** — draft above is 210 words.

---

## Accepted-word audit of the draft (word-level safety check)

| Token | In draft? | Status |
|---|---|---|
| significant(ly) | No | PASS |
| outperform(s) | No | PASS |
| superior / SOTA / state-of-the-art | No | PASS |
| generaliz(e/ation) | No | PASS |
| strong agreement | No (uses "comparable to") | PASS |
| robust | Yes — "robust instance-ranking policy" (R1 policy name, claim_audit row 12) | PASS (policy-name exception) |
| cross-species | No | PASS |
| A6 / A7 | No | PASS |
| "best" / "highest" unqualified | No | PASS |
