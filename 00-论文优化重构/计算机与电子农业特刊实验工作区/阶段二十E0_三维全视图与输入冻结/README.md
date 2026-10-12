# Phase20E-0 — delivered input freeze

Start with `PHASE20E0_FINAL_REPORT.md`. This stage actually generated full registered
view candidates and paired input datasets; it did not run 2DGS optimization or mesh
reconstruction. Production R1, 2DGS and M2M3 algorithms remain unchanged.

## Measured result

| Scene | Registered / P6 covered | New instances | R0/R1 switches | Future distinct runs |
| --- | --- | --- | --- | --- |
| CaoMei2 | 106 / 106 | 111 | 1 train view | 3 |
| WanNianQing1 | 213 / 213 | 339 | 3 train views | 3 |
| HongZhang | 213 / 213 | 472 | 0 | 2 |

All three A groups are **A_raw_RGB_control_v1**, not recovered historical ForeSplat.
Alternate old RGBA was tested and cannot certify missing original alpha identities.
HongZhang B/C are pixel-identical and actual consumed training RGB is identical;
exclude its duplicate B optimization and share the C foreground checkpoint later.

## Assets and evidence

- Managed data: `/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1`, outside Git, about 9.6GiB.
- Every source/candidate/RGB/alpha/camera/split hash is in the root CSV manifests.
- Exact restore and read-only verification: `ARTIFACT_RECOVERY.md` and
  `scripts/verify_managed_cache.py`; preserve sparse camera directory symlinks.
- `reports/ACCEPTANCE_GATES_G1_G15.csv` contains 45 real scene-specific checks.
- `reports/*_PREFLIGHT.json` records all 1596 real group-view loads and nine actual
  CUDA no-optimizer renders. Forward previews are not trained reconstructions.
- `PHASE20E1_COMMANDS.md` contains eight unexecuted future optimization commands;
  the wrapper defaults to dry configuration inspection without the execution flag.
- Hold-out reference is the same canonical raw RGB for every treatment; masks are
  not independent geometric GT. No true Chamfer/geometric F-score claim is supported.

Raw logs preserve early failed attempts and their later corrected reruns, including
native output spacing; they are not silently erased. CSV uses LF in Git while
managed artifacts and historical file bytes are untouched by text normalization.

## Identity corrections

CaoMei2 historical crop is full-size black-masked RGB, not a spatial bbox crop.
FFT renumbering means registered crop0146 is raw0170; TEST32 raw0146/crop0129 is not
registered. WanNianQing1 raw0227 is also unregistered. Correcting raw joins does not
change any of the predeclared registered camera train/validation assignments.
Damaged WanNianQing1 input0136 has conflicting historical undistorted RGB. Actual
native CPU SIFT strongly associates the current raw0136 with original COLMAP
image_id133 observations, with adjacent wrong-frame negative controls; the old
derived RGB is not used. See `reports/WanNianQing1_0136_SIFT_KEYPOINT_IDENTITY.json`.

No formal training, mesh processing, M2M3-GO execution or manuscript modification
is authorized by this delivered stage. Phase20E1 requires a separate user request.
