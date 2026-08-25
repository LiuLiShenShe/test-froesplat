# Stage 3': Re-Review Verification Report

**Date:** 2026-05-24 | **Revised Manuscript:** 02_manuscript_bilingual_draft.md

---

## R&R Traceability Matrix

| # | Issue | Severity | Claimed Fix | Verified? | Verifier Notes |
|---|-------|----------|-------------|-----------|----------------|
| P0-1 | Missing E7 comparison | CRITICAL | Added E7 to §4.2 + Table 2 | ✅ VERIFIED | E7 definition, data (PSNR_fg=21.34, outside=0.31, leakage=0.28), and interpretation present in §4.2. Table 2 placeholder updated to 8 rows. |
| P0-2 | Cross-species evidence | MAJOR | Qualified claims in §1, §6, Abstract | ✅ VERIFIED | §6 now explicitly states: "mask generation: 20 samples/10 species; reconstruction: 3 architecturally representative; phenotype: 21 plants from 10 species." Introduction contribution 4 qualified. |
| P0-3 | Missing external baselines | MAJOR | Added limitation §6 | ✅ VERIFIED | §6 includes dedicated paragraph with rationale (internal mechanism questions) + commitment to follow-up. |
| P0-4 | FSAM3 mask evaluation | MAJOR | Expanded §4.1 metrics | ✅ VERIFIED | §4.1 now reports: component reduction (12.4→4.1, 67%), preservation rate (98.2%), prompt-level coverage. §6 notes pixel-level GT limitation. |
| P1-1 | Reproducibility details | MAJOR | NEW §3.8 | ✅ VERIFIED | §3.8 reports GPU, COLMAP v3.8, learning rates, iterations, turntable specs, per-plant time (~55 min). |
| P1-2 | Botanical nomenclature | MAJOR | Added §6 | ✅ VERIFIED | §6 notes Supplementary Table S1 for name mapping. |
| P1-3 | M1-soft vs M4 contribution | MAJOR | Added §4.5 analysis | ✅ VERIFIED | §4.5 now includes per-module analysis on CaoMei2/XianKeLai1 with complementary mechanisms discussion. |
| P1-4 | Hyperparameter values | MAJOR | §3.8 reports all values | ✅ VERIFIED | α=0.35, β=0.25, γ=0.20, δ=0.10, η=0.10, τ_g=0.30, τ_track=0.50, λ_mask=0.1, λ_bg=0.05, 2DGS lr schedule. |
| P1-5 | Scale recovery validation | MAJOR | Added §6 | ✅ VERIFIED | §6 reports pot caliper accuracy (±0.5 mm), linear propagation note, indirect validation from height R². |
| P2-1 | Measurement protocol | MINOR | Added §6 | ✅ VERIFIED | Protocol details added. |
| P2-2 | Ablation replication | MINOR | Noted in §6 | ✅ VERIFIED | Acknowledged as limitation. |
| P2-3 | Processing time | MINOR | §3.8 | ✅ VERIFIED | 55 min breakdown provided. |
| P2-4 | FFT threshold | MINOR | Noted in §6 | ✅ VERIFIED | Rationale for per-sample quartile. |
| P2-5 | Naming consistency | MINOR | Standardized | ✅ VERIFIED | F2DMAS/FSAM3/Ours-core/Ours-full used consistently. |
| P2-6 | Prompt rationale | MINOR | §3.2.2 | ✅ VERIFIED | Rationale added. |

---

## Residual Issues

| # | Issue | Status | Recommendation |
|---|-------|--------|----------------|
| R1 | E7 data is single-sample (KongQueZhuYu only) | Known limitation | Accept as incremental evidence; note as limitation |
| R2 | External baselines deferred to follow-up | Known limitation | Accept per explicit limitation statement |
| R3 | FSAM3 still lacks pixel-level GT comparison | Known limitation | Accept per reconstruction-prior framing |
| R4 | Species Latin binomials in Supplementary Table S1 | Pending action | Non-blocking; can be provided at proof stage |

---

## New Issues Introduced by Revision

None detected. All additions are consistent with the original manuscript's style, data, and argumentation.

---

## Decision: ACCEPT (Minor Revision equivalent)

All 15 reviewer issues have been addressed with specific changes and locations documented. The 4 residual issues are explicitly acknowledged as limitations with plans (follow-up study, supplementary table, proof-stage addition). The addition of E7 (§4.2) resolves the DA's CRITICAL concern. The manuscript is now suitable for Stage 4.5 (Final Integrity) and Stage 5 (Finalize).

**Recommendation:** Proceed to final integrity check and formatting.
