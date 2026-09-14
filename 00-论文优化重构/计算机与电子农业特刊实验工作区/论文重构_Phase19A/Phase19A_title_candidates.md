# Phase 19A — Title Candidates (ForeSplat Rewrite)

**Generated:** 2026-09-14 | **Target venue:** Computers and Electronics in Agriculture (CompAg) — special issue on vision foundation models in agriculture
**Source inputs:** `Phase19A_claim_evidence_matrix.csv`, `Phase19A_claim_audit.csv`, `Phase19A_resolution_status.md`, `Phase19A_method_component_audit.md` (B-DA-C05 naming), `Phase18_report.md`

---

## Hard constraints (apply to all candidates)

1. **No "cross-species"** in the title — B-EIC-C01 is PARTIALLY_RESOLVED (reconstruction validation = 3 architecturally representative samples only). Use "architectural diversity" / plant phenotyping, not cross-species.
2. **No "robust"** as a general adjective — allowed only inside the method name "R1 robust instance ranking"; "robust" in a title is a general quality claim (claim_audit row 12). Avoid.
3. **No "SOTA / state-of-the-art / outperforming"** (claim_audit rows 4, 8).
4. **No "generaliz(e/ation)"** (claim_audit row 6).
5. **Brand naming consistency (B-DA-C05):** ForeSplat = complete pipeline; RAP-FSAM3 = segmentation/mask component; F2DMAS as the pipeline name is NOT used in main text — do not introduce a new brand in the title that conflicts with main-text naming.
6. Must surface (in some combination): VFM-guided foreground-object 2DGS + plant phenotyping + the ranking insight. Special-issue readers need the VFM angle; reviewers need the scientific spine.

---

## Baseline evaluation

**Existing title:** *ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping*

- **Strength:** names the two cores (VFM-guided foreground-object 2DGS + low-cost plant phenotyping); keeps the established "ForeSplat" brand; explicit low-cost angle is CompAg-relevant; no overclaim tokens; the method-chapter naming (ForeSplat pipeline, RAP-FSAM3 mask component) is consistent with B-DA-C05.
- **Risk:** "Vision-Foundation-Model-Guided" is generic — dozens of VFM+GS papers could share it; it hides the paper's most distinctive scientific finding (the ranking-failure diagnosis and R1 fix), which is the strongest novelty evidence after Phase 17C/18; gives no hint that the paper contains a diagnosis→fix→rescue story, so reviewers may miss the hard-case contribution.
- **Verdict:** strong, safe baseline — but under-sells the new science. Use as the **pipeline-brand anchor** (kept verbatim as the canonical method name), and choose a candidate that adds the ranking insight as the headline.

---

## Candidate 1 (existing title — baseline)

- **EN:** ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping
- **中:** ForeSplat：视觉基础模型引导的前景物体二维高斯泼溅用于低成本植物三维表型分析
- **Strength:** safe; brand-retained; low-cost visible; consistent with main-text naming; zero overclaim risk.
- **Risk:** hides the ranking-failure + R1 contribution; "Vision-Foundation-Model-Guided" is generic and weakly differentiating for the special issue.
- **Recommendation:** **backup** — deploy as the canonical method-name anchor (first mention in the manuscript) but not as the headline title.

---

## Candidate 2 (ranking spine — recommended primary)

- **EN:** ForeSplat: When Segmentation Succeeds but Selection Fails — VFM-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping
- **中:** ForeSplat：当分割成功而选择失败时——视觉基础模型引导的前景物体二维高斯泼溅用于低成本植物三维表型分析
- **Strength:** the "segmentation succeeds but selection fails" phrase is the paper's most memorable, honest finding (C01/C02 PROVEN: oracle F1 ≈ 0.987–0.988 present, wrong candidate selected, F1 → 0); retains ForeSplat brand + phenotyping purpose + low-cost; signals a diagnosis→fix story without overclaiming; VFM angle visible for the special issue.
- **Risk:** title length (compete with journal limits); "selection" needs a one-line definition early in the abstract so readers do not misread it as dataset selection; reviewers focused on reconstruction may want the reconstruction novelty front and center (mitigated by keeping "Foreground-Object 2D Gaussian Splatting" in the title).
- **Recommendation:** **use** (primary candidate).

---

## Candidate 3 (prompt→selection arc — backup)

- **EN:** From Fragile Prompts to Reliable Candidate Selection: VFM-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost Plant Phenotyping
- **中:** 从脆弱提示到可靠候选选择：面向低成本植物表型的视觉基础模型引导前景物体二维高斯泼溅
- **Strength:** captures both failure modes in one arc — prompt fragility (P03, per-prompt F1 0.08–0.97) and selection reliability (C02/C04/R1); ag-audience friendly; "reliable candidate selection" is a design goal, not an overclaim.
- **Risk:** "reliable" is a soft quality adjective — must be scoped as "reliable candidate selection" (the R1 design goal), never as "reliable segmentation"; drops the ForeSplat brand from the headline; slightly longer.
- **Recommendation:** **backup** — good if the editors prefer a problem→solution arc over a brand-led title.

---

## Candidate 4 (phenotyping-forward — backup for ag-applied emphasis)

- **EN:** Low-Cost 3D Plant Phenotyping via Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting
- **中:** 基于视觉基础模型引导的前景物体二维高斯泼溅的低成本植物三维表型分析
- **Strength:** maximum CompAg relevance — phenotyping and low-cost are front and center; concise; zero overclaim risk.
- **Risk:** ranking insight invisible; "Vision-Foundation-Model-Guided" generic; least differentiating of the set; the paper's hard-case science is buried.
- **Recommendation:** **avoid as primary** — acceptable only if the special-issue call is explicitly application-first and the ranking story is judged too method-heavy for the venue.

---

## Candidate 5 (diagnosis-and-fix framing — situational)

- **EN:** Diagnosing and Fixing Candidate-Selection Failures in Vision-Foundation-Model-Based Plant Segmentation and 3D Reconstruction
- **中:** 基于视觉基础模型的植物分割与三维重建中候选选择失败的诊断与修复
- **Strength:** research-process framing matches the actual paper structure (oracle audit → diagnosis → R1 fix → rescue, Phase 17B/17C); novel angle for the special issue; explicitly honest about a failure-mode study.
- **Risk:** reads like a methods-comparison/benchmark paper rather than a phenotyping-tool paper; under-sells the reconstruction contribution (ForeSplat) and the downstream phenotyping value that CompAg reviewers expect; longer; no brand.
- **Recommendation:** **avoid** as the primary title; may be repurposed as the Section title for the ranking-failure analysis subsection (Results §3.3).

---

## Recommendation summary

| Rank | Candidate | Role |
|---|---|---|
| 1 | **Candidate 2** (When Segmentation Succeeds but Selection Fails) | Primary headline title |
| 2 | Candidate 1 (existing baseline) | Pipeline-brand anchor; first-mention canonical name; fallback if length limits force a simpler title |
| 3 | Candidate 3 (From Fragile Prompts to Reliable Candidate Selection) | Backup if editors prefer problem→solution arc |
| 4 | Candidate 4 (phenotyping-forward) | Backup only if application-first framing is required |
| 5 | Candidate 5 (diagnosis-and-fix) | Repurpose as Results subsection title; not a journal title |

**Cross-species note:** none of the candidates contains "cross-species" — required by B-EIC-C01. If any title variant is later proposed with "cross-species", it must be rejected unless reconstruction validation is extended beyond 3 samples.
