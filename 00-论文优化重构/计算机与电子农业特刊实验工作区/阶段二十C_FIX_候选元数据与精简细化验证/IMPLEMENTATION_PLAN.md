# Phase20C-FIX implementation plan

1. Audit frozen Phase20A–C configurations, historical P12/P17A candidate identities,
   mask format, input hashes, original metadata precision, and the SPNP call path.
2. Add an identity-keyed cache metadata reader without changing ranking_policy.py,
   scoring weights, existing CLI options, candidate masks, or historical outputs.
   Strict replay requires metadata and mask/RGB SHA256; legacy caches warn explicitly.
   Cover shuffled rows, non-contiguous instance IDs, missing/invalid metadata, hashes,
   exact ties/order, and the frozen historical scores with CPU tests.
3. Freeze a new metadata snapshot and run V0/V1 CPU score replay on Easy21/GT10.
   Historical rounded scores are not automatically treated as full precision:
   any exact reconstruction must be justified from independent archived arithmetic,
   identity matching and cross-checks, otherwise report METADATA_UNVERIFIED.
4. Run two-frame V4/V5 GPU smoke tests, then Easy21/GT10 SPNP-only runs in new
   directories. Hold all non-factor settings fixed. Do not require COLMAP for SPNP.
5. Evaluate selected/final masks with the frozen Phase20B metric definitions;
   report sample-equal aggregates separately from frame-paired differences, failures,
   acceptance, changed pixels and real runtime. Compare V3 only on valid matched data.
6. Audit the three COLMAP gaps against registered image names and actual model
   camera/coordinate identities. Never synthesize observations or rebuild COLMAP.
7. Prepare TEST32 annotation assets and a fixed, manual-label-gated evaluator, without
   evaluating test accuracy or tuning parameters. Record the human work outstanding.
8. Generate the evidence-backed method decision and all requested deliverables;
   run focused then broader tests, verify protected hashes/history, explicitly stage
   small artifacts, commit and push without force. Stop before Phase20D.

## Risks / verification

- Old CSV confidence is rounded to six decimals. Full-score equivalence needs a
  documented numerical check, not an assumption or a directory-position join.
- GT10 full-chain geometry may remain unavailable. This does not block SPNP-only.
- SPNP may regress some frames, HD95 or leakage. Report all fixed metrics; prefer a
  stable supported method, not a longer chain. All sets here are DEV/regression.
- Network search: Bocha returned HTTP 403 (quota exhausted); the requested GitHub
  page did not return usable web content. Experimental claims use local frozen
  repository evidence, not search snippets or training-data recollection.
- /home/test/.codex/RTK.md is absent; no instructions can be read from that path.

## Frozen decision rule (before new GPU results)

R1 must retain all three hard-case target instances and Easy21 selections. Recommend
V4 only if neither dataset loses a rescued target or gains empty/severe outputs,
sample-equal final F1/IoU/Boundary-F1 are not worse than V1, and sample-equal HD95
and Leakage do not regress on either dataset. Do not trade a favorable frame-mean
metric against an unfavorable sample-equal metric. If V4 fails this conservative
multi-metric rule, retain V1 as the default and SPNP as an optional ablation.
V3 additionally requires reliable geometry and consistent benefit over V4; no
claim of full GT10 validation from a subset. No GT-dependent runtime switching.
