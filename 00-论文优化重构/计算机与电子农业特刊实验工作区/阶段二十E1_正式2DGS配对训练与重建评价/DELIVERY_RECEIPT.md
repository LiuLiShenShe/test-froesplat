# Phase20E-1 delivery receipt

## Verified publication

- Verification time: `2026-10-14T11:46:40+08:00` (Asia/Shanghai).
- Branch: `main`.
- Frozen baseline / source / configuration: `3e909f93bf1c318458b8f0c561249357c4db4ae1`.
- Experiment artifact commit: `addc10a840a8676efb19b5901c58f0e8a153192c`.
- Normal `git push origin main` completed successfully; no force push.
- `git ls-remote origin refs/heads/main` returned the same experiment SHA.
- The worktree was clean immediately after the experiment commit and before
  this documentation-only receipt; `git diff --check` passed.
- The artifact commit contains 186 new files, 60,266,055 bytes, exclusively
  under this Phase20E-1 stage. No checkpoint, PLY, Python cache, API credential,
  paper edit, or production algorithm change was committed.

This receipt and the final-report publication record form a separate,
documentation-only child commit to avoid pretending that a document can embed
its own commit SHA. Its hash is obtained with
`git log -1 --format=%H -- <this-stage>/DELIVERY_RECEIPT.md`. The final user
delivery records that child commit, remote equality, and the final clean-tree
check after its normal push.

## Scientific and execution acceptance

- Final acceptance: **PASS, G1-G20 all 20 passed**.
- Final acceptance timestamp: `2026-10-14T11:38:47.283752+08:00`.
- Independent final trainings: **8/8**, all 30,000 iterations.
- Reloaded and validated final checkpoints / PLY models: **8/8**.
- Held-out model-camera renders: **177 actual renders**, 204 metric-table
  rows including 27 explicitly labelled HongZhang B=C aliases, over 68 frozen
  validation cameras. No ninth optimization was performed.
- Float-render replay: **531/531 PSNR/SSIM/LPIPS values**, maximum absolute
  difference **0.0**. Scene summaries recomputed from per-view records.
- Frozen-input recheck: **6,831 managed records** and **57 production source
  files** unchanged.
- Regression tests: **13 passed**, executed by standard-library unittest in
  the frozen training Python environment.
- Final eight successful runs: **21,847.500431463122 s** total wall-clock.
  The **1,206.402529234998 s** failed-attempt overhead is separately preserved,
  never mixed into completed-run statistics.
- Results: A vs B **REGRESSED** on the common raw-RGB reference; B vs C
  **MIXED**; R1 pilot 3D benefit **MIXED**, not a demonstrated general benefit.
- No independent 3D GT and no matching independent manual contours on the
  68 validation cameras. Proxy ROI/alpha observations are not geometric GT.

Operational PASS certifies truthful completion and reproducibility of this
pilot, not a positive R1 claim. Scene-level effects, negative cases, native
repeatability limits, and foreground-conditioned CaoMei2 camera provenance
remain explicit in the final report and handoff.

## Artifact preservation and reproducibility

- Frozen external data root:
  `/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1`.
- Completed model root: the data root's `phase20e1_outputs/`.
- Partial failed runs: the data root's `phase20e1_failed_attempts/`.
- `reports/LARGE_ARTIFACT_MANIFEST.csv` records paths relative to the data
  root, absolute current locations, sizes, and SHA256 for large models/renders.
- `TRAINING_RUN_MANIFEST.csv`, `configs/`, `logs/`, checkpoint validation and
  evaluation CSVs link inputs, source/config revision, commands, and results.
- `REPRODUCTION.md` describes relocation, restoring the frozen environment,
  rerendering, and metric replay. Models and full-size render caches are
  retained externally, not duplicated in Git.

### Delivery evidence SHA256

The final-report digest below includes its publication-record appendix in
this documentation commit; other evidence digests are unchanged from the
experiment artifact commit.

| File relative to stage | SHA256 |
| --- | --- |
| `PHASE20E1_FINAL_REPORT.md` | `d4153cc5bf72576f9dca0ad6666b8934ac2702b83e9da37d7f65878673b10c6e` |
| `reports/FINAL_ACCEPTANCE.json` | `961762755241c3ef025779f45a6b2a314607cfabd5c694a98191d2669be397ee` |
| `reports/RESULT_STATUS.json` | `ca22ad072b594e0679ce130a4810e234e8645ee2246bf070287f6dd549c9c5ad` |
| `reports/FINAL_INPUT_RECHECK.json` | `cf749670bb3c28eccb3c18a2be6e7b17db3ece7a90e304f1d1c0969126ae6cb9` |
| `reports/FLOAT_RENDER_METRIC_RECOMPUTATION.csv` | `ed1c3fae9456ede6ed34c81b99310a87360323ce071fcd7dd668dd5c2665c61b` |
| `HELDOUT_PER_VIEW_METRICS.csv` | `4a1216f3d1e1ad4bf025f07cd107109525584cefa094c81d6ec789e9af3f9473` |
| `HELDOUT_SCENE_SUMMARY.csv` | `fea47b46cc332b8c114599561c9f802450df97d4dd39b0f47635992f7e4c1c86` |
| `CHECKPOINT_VALIDATION.csv` | `c3186ebbd093d011950eec21c4b99dfb2967963c21842056b5aec875b21588f6` |
| `TRAINING_RUNTIME_AND_GPU.csv` | `0cc73db7dace51a7157b873f1e0a3ca5d6c9053aa5e42549d5df1cdf5a11150f` |
| `logs/FINAL_REGRESSION_TESTS.txt` | `6dca087d6ef59ff8811034e322507a7e49663507726eb99b2bda2656943877c3` |
| `reports/LARGE_ARTIFACT_MANIFEST.csv` | `e2c4cfe53e730a676360849f816ea0441de37581c00f64717e9550b18e505f13` |

## Stop boundary

No Phase20E-2, M2M3-GO, Mesh reconstruction, phenotyping, full historical
retraining, or manuscript modification was started. `NEXT_PHASE: PHASE20E2`
is a handoff recommendation only. Additional independent scenes and
predeclared repeat seeds are recommended before adopting R1 as a general
foreground reconstruction module.

Sources: frozen local input inventories, actual optimization logs, restored
models, float renders, and recomputed per-view evidence. Bocha web search was
attempted but returned HTTP 403 (quota/credit); no web-derived experimental
result is asserted. Figure review used the model's direct image viewing.
