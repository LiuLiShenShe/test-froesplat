# Phase20E-1 final delivery audit — 2026-10-14

This continuation did not restart optimization, alter frozen inputs or enter E2.
Existing final checkpoints and real float renders were reused only for read-only
verification and metric recomputation.

## Verification performed

- Eight required final models: native iteration 30000, successful checkpoint
  restore, PLY/checkpoint parameter identity PASS, reload render max error 0.0.
- 68 distinct frozen validation cameras: WanNianQing1 27, CaoMei2 14,
  HongZhang 27. Actual model-camera renders: 177; table rows: 204.
  The additional 27 rows are HongZhang B=C labelled aliases, not fresh training
  or fresh rendering. Empty/invalid final renders: zero.
- Full PSNR/SSIM/LPIPS replay on the 177 saved float renders: 531/531 metric
  values PASS, maximum absolute difference 0.0. No PNG quantization is used.
- All eight evaluations have exactly the same recorded metric code, package
  versions, LPIPS/VGG weight identities and common-reference policy.
- All 6831 E0 managed records rehashed, 6513 distinct inodes, no mismatch;
  57 production source files and frozen CUDA extension identities unchanged.
  See reports/FINAL_INPUT_RECHECK.json for the exact verification timestamp.
- Frozen-environment standard-library regression tests: 13 passed, no skips.
  See logs/FINAL_REGRESSION_TESTS.txt. System pytest also passed 13 tests;
  pytest is not installed into the frozen training environment, so no dependency
  change was made and unittest is the supported replay runner there.
- Three existing real held-out composite figures were directly viewed again
  with the native image viewer; no synthetic reconstruction image or external
  vision API was used. No human sign-off is implied.

## Read-only acceptance remediation

The Git boundary audit now uses NUL-separated names, avoiding the default
quotePath escaping of Chinese filenames after staging. G15 additionally requires
full 177-render / 531-metric replay coverage and rejects missing, duplicate,
stale or nonfinite rows. Seven new unit tests verify those edge cases. These
changes are stage-local; no training, loss, rasterizer, R1 or metric definition
was changed. Interpreter/test caches are excluded by the stage .gitignore.
The stage .gitattributes preserves csv.writer CRLF, valid Matplotlib SVG path
spacing and raw failure-trace indentation. Other code/report whitespace checks
remain enabled. All eight final and both failed raw stdout/stderr hashes were
rechecked against their PROCESS records; no raw log byte was altered.

## Interpretation retained

A->B regresses on all three full-image common raw RGB metrics in all scenes,
while background energy and Gaussian counts are reduced. Intentional background
removal and target quality must not be conflated. WanNianQing1 B->C has tiny
favourable primary deltas (+0.001684 dB PSNR, +0.000790 SSIM, -0.000728 LPIPS)
and +4978 Gaussians. CaoMei2 B->C regresses (-0.025290 dB PSNR, -0.004376 SSIM,
+0.002335 LPIPS) and adds 13763 Gaussians. HongZhang B=C is exactly unchanged.
Per-view B->C directions recomputed from metrics/PAIRED_VIEW_DIFFERENCES.csv:

- WanNianQing1: IMPROVED=14, MIXED=11, REGRESSED=2, NO_CHANGE=0.
- CaoMei2: IMPROVED=1, MIXED=7, REGRESSED=6, NO_CHANGE=0.
- HongZhang: IMPROVED=0, MIXED=0, REGRESSED=0, NO_CHANGE=27.

No independent plant contour or 3D GT is available for these held-out cameras.
Background/alpha/silhouette calculations are explicitly predicted-R0 proxies;
physical holes and floating geometry are not verified. One-seed, three-scene
results and observed same-config native CUDA replay variation cannot establish
statistical significance, broad generalization or a robust causal R1 benefit.

Successful final-run training wall-clock total: 21847.500431463122 seconds.
Two preserved logging-failed attempts add 1206.402529234998 seconds, excluded
from final-model averages. Shared-machine contention means wall-clock speed
and sampled device memory are not isolated algorithm/process measures.

## Publication and stop boundary

Only small E1 scripts, complete configs, genuine metrics, reports, real composite
figures and log indices are eligible for Git publication. External checkpoint,
PLY, original-resolution rendering and failed-output assets are identified by
relative data-root paths and SHA256, not uploaded to Git. Exact publication
commit identities and remote confirmation are in DELIVERY_RECEIPT.md.

NEXT_PHASE=PHASE20E2 is a handoff recommendation, not execution permission.
R1 is not established as the default reconstruction foreground. A future
predeclared independent-scene / repeated-seed / independent-ROI protocol is
recommended before any broad claim; separate Mesh/M2M3-GO controls may be planned
but no Mesh, phenotyping, M2M3-GO or manuscript work is performed here.

Sources: local E0/E1 frozen manifests, raw training traces, restored checkpoints,
real renderer outputs and per-view CSVs. Bocha search on 2026-10-14 returned HTTP
403 quota/credit failure. No experiment value is substituted from web results or
model memory. The referenced /home/test/.codex/RTK.md is absent on this machine.
