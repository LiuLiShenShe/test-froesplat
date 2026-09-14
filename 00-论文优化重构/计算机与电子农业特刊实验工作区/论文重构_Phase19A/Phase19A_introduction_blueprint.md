# Phase 19A — Introduction Blueprint (FILE A: WP6B)

> Target: 6-paragraph Introduction for the CompAg special-issue manuscript.
> Narrative arc: VFM prior -> 3D Gaussian reconstruction objective -> plant phenotyping.
> Claim constraints drawn from `Phase19A_claim_evidence_matrix.csv`.
> Audit date: 2026-09-14.

---

## Paragraph 1 — Agricultural Need

**Purpose:** Motivate high-throughput, low-cost 3D plant phenotyping as a critical enabler for drought response, climate adaptation, and precision breeding programmes.

**Supporting points:**

1. Breeding programmes and crop-management decisions increasingly depend on 3D morphological traits (plant height, canopy width, leaf length/width) that are difficult or destructive to measure manually [ref1].
2. Climate variability (drought, heat stress) demands rapid phenotypic screening across large populations; manual phenotyping cannot scale [ref2].
3. Controlled-environment RGB imaging is non-destructive, low-cost, and increasingly standard in phenotyping facilities; however, translating 2D images into 3D measurements remains an open challenge [ref3].
4. 3D Gaussian Splatting (3DGS) has recently emerged as a fast, differentiable scene-reconstruction framework suitable for real-time rendering from sparse views [ref4].

**Transition to next paragraph:** While 3DGS provides the reconstruction substrate, the core bottleneck in plant scenes is accurate foreground segmentation — the input mask that isolates the plant from its pot, table, and background.

**Claim constraints:**

- MUST NOT use any claim from the evidence matrix in this paragraph (pure motivation, no results).
- MUST NOT claim "outperforms" or "generalizes" anywhere.
- May reference general phenotyping literature; no specific dataset or metric.

---

## Paragraph 2 — Multi-Plant / Dense-Scene Difficulty

**Purpose:** Explain why reconstructing a single foreground plant in a multi-plant potted-plant scene is hard, and why direct prompt-based VFM segmentation fails.

**Supporting points:**

1. In multi-plant scenes, adjacent foliage overlaps and pot rims create ambiguous boundaries; a single "segment the plant" prompt cannot disentangle the target from neighbours [ref5].
2. Dense canopies produce severe self-occlusion; the visible surface area of the target plant may be a small fraction of the total green region [ref6].
3. Background clutter (tables, greenhouse structures, other pots) introduces false positives when the VFM is given only a coarse language prompt [ref7].
4. Prior mask-free or prompt-only approaches (CLIPSeg, Grounded-SAM without instance grounding) conflate "green region" with "target plant," yielding masks that span multiple plants or include pot/table surfaces [ref8].

**Transition to next paragraph:** These failure modes motivate a closer look at why Vision Foundation Models (VFMs), despite their general segmentation power, produce unreliable masks when prompted directly on potted-plant scenes.

**Claim constraints:**

- MUST NOT cite specific P01-P08 results (those belong to later paragraphs).
- SHOULD use descriptive language about the *difficulty* (not claim P03 yet).
- No forbidden wording from the evidence matrix.

---

## Paragraph 3 — VFM Opportunity: SAM-Family and Prompt Fragility

**Purpose:** Introduce the SAM-family as the most promising VFM substrate for plant segmentation, then present evidence that bare prompting is semantically fragile (P03).

**Supporting points:**

1. Segment Anything Model (SAM) and its successors (SAM2, SAM3) provide strong zero-shot instance segmentation when given suitable prompts (points, boxes, masks) [ref9].
2. However, the *choice* of language prompt critically affects segmentation quality in plant scenes — this is the prompt-robustness problem [ref10].
3. [P03 evidence slot] Empirical evaluation of five plant-relevant prompts on the frozen P6 evaluation set shows dramatic F1 variation: from 0.9692 (plant-instance prompt) to 0.0768 (seedling-oriented prompt on mature plants); this evidence supports the claim that single-prompt VFM segmentation is semantically fragile in potted-plant scenes [P03].
4. The fragility is not a model-capacity issue but a *semantic bias* issue: prompts that emphasise organ-level or background-excluding language produce masks that either under-segment (leaf-only) or over-segment (green-background inclusion) [P03].

**Transition to next paragraph:** This prompt fragility motivates the central methodological insight of this work: instead of relying on a single prompt, one should generate multiple candidate masks from diverse prompts and then rank them using features that correlate with reconstruction quality.

**Claim constraints:**

- MUST cite P03 evidence explicitly (Table 4, prompt robustness data).
- MUST use allowed wording from P03: "single-prompt VFM segmentation is semantically fragile in potted-plant scenes."
- MUST NOT use forbidden wording from P03: "single-prompt VFM segmentation is robust" or "the prompt set generalises to all scenes."
- MUST NOT yet cite P01/P02 numbers (those are the method's own results, not the motivation).

---

## Paragraph 4 — Method Insight: Reconstruction-Aware Prompting + Candidate Ranking

**Purpose:** State the core methodological insight — generate multiple candidates and rank them using geometry/contrast/completeness features — and introduce the 2DGS representation choice.

**Supporting points:**

1. Rather than selecting the "best" prompt a priori, generate candidate masks from multiple prompt instantiations (per plant instance) and score each candidate with geometry-aware quality features (area ratio, compactness, edge conformity, contrast, completeness) [ref11].
2. The ranking features encode reconstruction-relevant properties: a mask that tightly covers the target plant while excluding pots and neighbours will produce better foreground supervision for downstream 3D reconstruction [ref11].
3. A key finding motivating the ranking design is that foreground-background contrast, while intuitive, can paradoxically favour small high-contrast distractors over larger whole-plant candidates (C03 evidence) — this informed the production R1 ranking policy [ref12].
4. For the 3D representation, foreground-object 2D Gaussian Splatting (2DGS) is chosen over 3D Gaussian primitives because plant surfaces are predominantly thin, layered foliage better modelled by 2D oriented disks than 3D volumes [ref13].
5. The 2DGS framework allows mask-guided supervision: foreground RGB loss, alpha opacity constraints, and background suppression can be injected directly into the splatting optimisation, constraining the reconstruction to the masked foreground object [ref14].

**Transition to next paragraph:** Building on these insights, the proposed pipeline — RAP-FSAM3 (Reconstruction-Aware Prompting with Foreground-aware SAM3) followed by ForeSplat (Foreground-Object 2DGS) — integrates per-instance candidate generation, robust ranking, and foreground-conditioned 2D reconstruction.

**Claim constraints:**

- MAY reference C03 (mechanistic, SUPPORTED level) but must use allowed wording: "q_contrast can favor small, high-contrast distractor masks."
- MUST NOT claim C03 as PROVEN (it is SUPPORTED, correlational).
- MUST NOT use "outperforms" or "generalizes" (claim_audit forbidden).
- MUST NOT introduce A6/A7 at any point.

---

## Paragraph 5 — Approach Overview: RAP-FSAM3 + ForeSplat Pipeline

**Purpose:** Present the complete pipeline at a high level, naming the modules in order, and flag the ranking-failure finding as a key design lesson.

**Supporting points:**

1. RAP-FSAM3 comprises: (a) FFT-based frame-quality filtering to exclude blurred frames; (b) per-instance SAM3 candidate generation using the production default prompt list (P2); (c) multi-feature candidate scoring (q_area, q_comp, q_edge, q_temp, q_contrast, q_sam); (d) R1 robust instance ranking that excludes contrast from the primary score to prevent distractor selection; (e) structure-balanced foreground/background prompt refinement; (f) geometry-consistency foreground correction [ref15].
2. ForeSplat receives the selected foreground mask and performs foreground-conditioned 2DGS optimisation: foreground RGB supervision, alpha/background opacity constraints, view-quality-aware loss weighting, mask-guided Gaussian pruning, and M2M3-GO primitive refinement [ref16].
3. The pipeline concludes with mesh extraction (TSDF fusion), scale recovery from a known pot diameter, and virtual phenotypic trait measurement (plant height, canopy width, leaf length, leaf width) [ref17].
4. A critical design lesson from iterative development: the initial scoring policy (including contrast in the primary score) produced catastrophic ranking failures on three DEV-set frames where high-quality candidates existed but were not selected; the R1 policy was developed to address this specific failure mode [C01, C02].
5. A6 (cross-view consensus) and A7 (memory propagation) are optional experimental branches that did not provide consistent measurable benefit under evaluated settings and are disabled by default [P08].

**Transition to next paragraph:** The specific contributions of this work, supported by evidence at defined levels, are summarised below.

**Claim constraints:**

- MUST NOT present A6/A7 as contributions (G14 / C08 forbidden).
- MAY reference C01 (PROVEN) and C02 (PROVEN) for the ranking-failure narrative.
- MUST use allowed wording from P08: "did not provide consistent measurable benefit under evaluated settings."
- MUST NOT claim A6/A7 "improve" or "provide consistent benefit."

---

## Paragraph 6 — Contributions

**Purpose:** Enumerate 3-4 specific contributions with explicit evidence levels; enforce all forbidden-wording constraints.

**Contributions:**

1. **RAP-FSAM3 foreground-prior generation with per-instance candidate ranking.** A reconstruction-aware prompting pipeline that generates diverse SAM3 candidates per plant instance and selects the best candidate using geometry/structure features, with the R1 robust ranking policy that excludes contrast from the primary score. Evidence: F1 = 0.9706 on the frozen P6 evaluation set among tested methods [P01, Table 3]; prompt-robustness analysis demonstrates semantic fragility of direct prompting [P03, Table 4]; module ablation confirms each component contributes to segmentation quality [P04, Table 5].

2. **Foreground-object 2D Gaussian Splatting with mask-guided supervision.** A reconstruction formulation that injects the foreground mask as a hard prior into 2DGS via foreground RGB loss, alpha opacity constraints, and background suppression; plus M2M3-GO primitive refinement. Evidence: PSNR = 31.09 dB, SSIM = 0.9711, LPIPS = 0.0365 on 20-sequence validation [P05, Table 7]; injection-position ablation confirms foreground-object loss injection is necessary over input masking or post hoc pruning [P06, Table 6].

3. **Ranking-failure diagnosis and rescue analysis.** Identification that catastrophic F1 failures (F1 = 0 on three DEV-set frames) were caused by candidate-ranking selection of high-contrast distractors, not candidate generation failure; and that the R1 policy eliminates these failures with zero material regressions. Evidence: oracle audit confirms raw candidate F1 = 0.9867–0.9882 on failure frames [C01]; failure diagnosis classifies root cause as ranking, not generation [C02]; R1 rescue brings all three frames to F1 >= 0.95 with zero regressions on Easy21 [C04, Table: failure_rescue].

4. **Virtual phenotypic trait measurement from foreground meshes.** Scale-recovered mesh extraction with pot-diameter calibration, producing trait measurements comparable to manual measurements. Evidence: plant height R2 = 0.988, canopy width R2 = 0.988, leaf length R2 = 0.974, leaf width R2 = 0.900 [P07, Table 8].

**Claim constraints for this paragraph:**

- MUST NOT use "outperforms," "generalizes," "statistically significant," "state-of-the-art," or any claim_audit forbidden wording.
- MUST use "among tested methods on this evaluation set" style language where comparative claims are made (per claim_evidence_matrix allowed_wording for P01).
- MUST NOT claim A6/A7 as contributions (C08 forbidden; A6/A7 disabled by default, not retained).
- MUST NOT claim "R1 significantly improves general segmentation performance" (C05 FORBIDDEN).
- SHOULD frame R1 contribution as rescue on the identified DEV set, with zero regressions (C04 allowed wording).
- MUST use "comparable to manual measurements" for P07, NOT "strong agreement" or "accurate/superior."
- MUST cite evidence levels: PROVEN for C01/C02, SUPPORTED for C03/C04, DESCRIPTIVE for P01/P05/P07.
- MUST reference specific tables (Table 3, 4, 5, 6, 7, 8) and figures where applicable.

---

## Summary of Citation Slots

| Slot | Source | Evidence Level | Paper Section |
|---|---|---|---|
| P01 | Frozen P6 evaluation metrics (F1=0.9706, ...) | DESCRIPTIVE | Results Table 3 |
| P03 | Prompt robustness (5 prompts, F1 range) | PROVEN | Results Table 4 |
| P04 | Module ablation (geometry-only fails) | DESCRIPTIVE | Results Table 5 |
| P05 | 2DGS reconstruction quality | DESCRIPTIVE | Results Table 7 |
| P06 | Injection-position ablation | DESCRIPTIVE | Results Table 6 |
| P07 | Phenotypic trait agreement | DESCRIPTIVE | Results Table 8 |
| P08 | A6/A7 negative ablation | NOT_SUPPORTED | Limitations / Figure D |
| C01 | Candidate generation succeeded on failure frames | PROVEN | Results (oracle audit) |
| C02 | Failures caused by ranking, not generation | PROVEN | Results (ranking failure) |
| C03 | q_contrast favours distractors (correlational) | SUPPORTED | Results / Discussion |
| C04 | R1 eliminates identified failures, zero regressions | SUPPORTED | Results (R1 rescue) |

---

*FILE A of WP6B. Date: 2026-09-14.*
