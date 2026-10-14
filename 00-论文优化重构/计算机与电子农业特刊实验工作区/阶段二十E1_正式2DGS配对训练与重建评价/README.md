# Phase20E-1 — matched 2DGS training and held-out evaluation

Baseline: `3e909f93`; GPU: physical device 1, frozen UUID. This stage does not
modify training algorithms, input images/masks, camera bundles, R1 or manuscripts.

## Execution plan

1. Verify all E0 managed hashes, complete configs, camera routing and GPU identity.
2. Run WanNianQing1 A/B/C using the unchanged E0 execution wrapper.
3. Load final models and evaluate every frozen held-out camera against common raw RGB.
4. Run/evaluate CaoMei2 A/B/C, then HongZhang A/C, with the same settings.
5. Generate paired tables, predefined-view figures, failure records and acceptance.
6. Check the scoped diff, commit and push only small E1 deliverables to origin/main.

`scripts/train_observed.py` invokes the actual E0 wrapper through `runpy`, retaining
its full frozen configuration and optimization call. Observer hooks only record
existing losses/counts and verify loaded identities before `training_setup`.
The final optimizer checkpoint is exported **after** unchanged training returns.
Native training logs at iteration 7000/30000 use treatment-specific reference
images and are secondary, never substituted for common-reference evaluation.

Large artifacts remain outside Git in the frozen E0 `phase20e1_outputs` directory.
Full stdout/stderr, checkpoint and render paths are indexed by SHA256. Training
finishes the native loop at 30000 (the upstream implementation performs its last
optimizer update at 29999); this convention is identical in every group.

HongZhang B/C is one foreground experiment, not independent replications. This is
a three-scene, one-seed feasibility pilot; no independent 3D ground truth or mesh
evaluation is claimed. CaoMei2 cameras are historically foreground-conditioned.

## Sources and environment

- Local E0 frozen manifests/configs and production source at baseline `3e909f93`.
- Context7 retrieved `/pytorch/pytorch` on 2026-10-13; synchronization/peak-memory
  patterns from PyTorch official CUDA notes and `test/test_cuda.py`.
- Bocha search was attempted but returned HTTP 403; no search result is fabricated.
- `/home/test/.codex/RTK.md` was referenced but does not exist on this machine.

See `configs/EXPERIMENT_PLAN.json` for the result-independent evaluation protocol.
