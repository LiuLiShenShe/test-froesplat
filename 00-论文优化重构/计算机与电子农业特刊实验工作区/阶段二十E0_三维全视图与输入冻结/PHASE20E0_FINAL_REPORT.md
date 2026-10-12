# ForeSplat Phase20E-0 — full-view foreground and matched input freeze

Baseline `f53bf8f3`; date 2026-10-12, timezone Asia/Shanghai. Initial worktree clean.
New files are contained in this stage; production algorithms and original historical
assets are protected. No formal 2DGS training or mesh reconstruction was executed.

## Validation evidence

`logs/G1_G15_VALIDATION.json`: 45/45 scene-specific acceptance checks.
`logs/SOURCE_PROTECTION_FINAL.json`: 7914 original historical/model/code files
fully SHA256-verified, then final size/mtime/ctime rechecked without overwrite.
`logs/DELIVERY_VERIFICATION.json`: 532 selections, 928 candidate instances,
1596 dataset group-views, no optimizer steps/checkpoints/meshes or outside-stage edits.
`logs/MANAGED_CACHE_RESTORE_VERIFICATION.json`: all 6831 managed files re-hashed;
exact camera symlink routing is verified. Approximate actual disk use is 9.6GiB,
while file-byte manifest sum includes repeated hardlinks (11.287678255GB).
CPU results: `logs/CPU_PHASE20E0_FINAL.txt` (34 passed, no skips),
`logs/CPU_HISTORICAL_REGRESSION_FINAL.txt` (76 passed),
`logs/CPU_PHASE20D_LEGACY_FINAL.txt` (107 passed), and the final delivery recheck.
Total final stage plus historical regression checks: 217 passed. No unrelated
test/code changes or new formatter were introduced.
All nine group loader/forward checks used the original local CUDA extensions;
they are input checks, not training outcomes or fabricated checkpoints.

| sample | registered | candidate_views | new_instances | R0_R1_switches | train_mask_differences | ready | training_runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 213 | 213 | 339 | 3 | 3 | True | 3 |
| CaoMei2 | 106 | 106 | 111 | 1 | 1 | True | 3 |
| HongZhang | 213 | 213 | 472 | 0 | 0 | True | 2 |

## Answers to the twelve requested questions

1. Registered counts: CaoMei2=106, WanNianQing1=213, HongZhang=213; exact image IDs/names
   from binary models are in the identity manifest. No view was silently removed.
2. Current full P6 covered views: table above. Coverage counts cache-verified real frames.
3. New GPU candidate instance counts: table above; zero-detection placeholders excluded
   from instance counts and separately flagged. Historical and new sets never conflated.
4. Real R0/R1 instance switches: table above; score sources/decompositions are retained.
5. B/C real differences: train and validation counts in `reports/SCENE_SUMMARY.csv` and
   mapped pixel hashes. Identical train inputs require no duplicate optimization.
6. A status: versioned historical reconstruction only when all original masks pair
   to actual raw RGB; otherwise explicitly named raw RGB control, not historical
   ForeSplat reproduction. Existing RGBA is incomplete/inconsistent and never silently
   called the recovered historical training version. See per-scene A_group and coverage.
7. Mapping verification: native COLMAP camera/RGB two-view tests, all observation
   conversion and real 3D projections; per-scene JSON/CSV retain measured residuals.
8. A/B/C input readiness: 3/3 scenes completed through real loader/forward.
9. Actually trainable scenes: WanNianQing1, CaoMei2, HongZhang.
10. Blocked/partial scenes: NONE;
    exact historical unreadable paths and recovery status are reported, not discarded.
11. Distinct Phase20E1 training count: 8; all commands present and unexecuted.
12. All nine worthwhile: NO; do not duplicate pixel-identical training input or run blocked scenes.

## Important limitations and corrected provenance

All three A datasets are **A_raw_RGB_control_v1**, not restored historical
ForeSplat. Legacy camera-grid RGBA coverage is 2/106, 193/213 and 185/213 for
CaoMei2, WanNianQing1 and HongZhang. HongZhang's 12 corrupt original RGBs were
recovered from verified registered raw COLMAP copies without modifying originals;
no scene remains blocked for canonical RGB. Detailed damaged paths are retained.
Alternate old RGBA recovery was explicitly tested and rejected because its alpha
does not reproduce known original SAM/registered-crop identities; this evidence
is in `reports/LEGACY_RGBA_RECOVERY_PROOF.json`, not inferred from missing-file counts.

Actual switches occur in TRAIN views only: CaoMei2 raw0139; WanNianQing1 raw0037,
raw0038 and raw0165. HongZhang B/C are identical in all 213 registered views AND
the actual loader-consumed training RGB hash. Its duplicate B training is excluded;
one foreground checkpoint can represent B/C. No future metric influenced this decision.

FFT re-numbering invalidates equal-stem joins. CaoMei2 TEST32 raw0146 actually joins
FFT crop0129 (unregistered), not registered crop0146/raw0170. Original crop is full-size black-masked RGB,
not spatial crop; foreground-conditioned camera estimation is explicitly preserved.
WanNianQing1 0227 remains unregistered and cannot support existing-camera 3D gains.

The current 2DGS loader stores RGBA alpha but defaults do not apply it to RGB loss or
initialization. Treatment uses explicit black RGB pre-composition while all groups
share camera bundle, original RGB, split, sparse points and optimizer settings.
Inherited sparse points may have a historical filtering/revision history: source and
target ID sets differ for WanNianQing1 and HongZhang; all shared XYZ are identical.
The chosen common **target** sparse version (including its existing PLY) is immutable
across A/B/C and no treatment applies additional mask-guided initialization filtering.
This stage demonstrates engineering input correctness, not segmentation accuracy
on all views or improved 3D geometry. No independent geometry GT is available.
Historical candidate model lineage has a documented archive limit: existing
instances have immutable mask/score/box provenance and matching archived recipe/
Phase20D model hashes, but the original Phase17A generation-time checkpoint digest
was not independently saved. New instances carry exact generation hashes.
Counts are computed from current manifests; archive precision remains six decimals.

Bocha web retrieval was attempted and returned HTTP403 quota exhausted. Context7
official COLMAP docs and direct official source retrieval were used for API/convention
checks; experiment numbers come exclusively from local measured assets and real runs.
Requested `/home/test/.codex/RTK.md` does not exist; it was not invented or substituted.

```text
PHASE20E0 STATUS: PASS

SCENE_READY:
CaoMei2: YES
WanNianQing1: YES
HongZhang: YES

FULL_VIEW_CANDIDATES:
WanNianQing1=213/213; CaoMei2=106/106; HongZhang=213/213

R0_R1_SWITCHES:
WanNianQing1=3; CaoMei2=1; HongZhang=0

CAMERA_MAPPING: VERIFIED
MATCHED_2DGS_INPUT: READY
TRAINING_PLAN: 8
PHASE20E1 READY: YES
```
