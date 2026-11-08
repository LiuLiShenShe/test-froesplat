# Independent ForeSplat SfM adapters

The pinned official AMB3R clustering, coarse registration, keyframe mapping,
non-keyframe refinement and SfMemory execute unchanged. The adapter never
concatenates unrelated clusters as an already aligned global reconstruction.
See [source audit](../../docs/SFM_SOURCE_AUDIT.md).

All commands use physical GPU1 and uv isolated environments. Default `--views 0`
uses all 208 usable KongQueZhuYu RGB/Mask pairs; the shared 210-row manifest keeps
the two corrupt RGB records. Existing nonempty output directories are preserved.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-amb3r/bin/python frontends/sfm/run.py --model amb3r --output experiments/KongQueZhuYu/attempts/NEW_AMB3R_SFM_RUN
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python frontends/sfm/run.py --model fast3r --output /tmp/NEW_FAST3R_SFM_RUN --persistent-output experiments/KongQueZhuYu/attempts/NEW_FAST3R_SFM_RUN
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python frontends/sfm/finalize.py --source /tmp/NEW_FAST3R_SFM_RUN --destination experiments/KongQueZhuYu/attempts/NEW_FAST3R_SFM_RUN --canonical experiments/KongQueZhuYu/NEW_FAST3R_SFM_ALIAS
```

The temporary-output flag records storage provenance; the finalize command
copies every artifact, verifies file sets/sizes and full-frame camera/input
records, checks original prediction arrays, and only then creates the alias.
`--local-compression none` stores every returned prediction losslessly, avoiding
repeated deflate between model calls. The complete bundle still retains raw NPZ
and complete PLY; visualization files are separate.

`Fast3RForAMB3RSfM.extract_amb3r_sfm_features(frames)` accepts images in
`(B,T,3,H,W)`, range `[-1,1]`, and returns `(B*T,1024)` mean-pooled original
Fast3R encoder tokens. The official pipeline currently uses one scene (`B=1`).
`run_amb3r_sfm(frames, cfg=None, keyframe_memory=None, benchmark_conf0=None)`
returns `world_points (B,T,H,W,3)`, `world_points_conf (B,T,H,W,1)`, and C2W
`pose (B,T,4,4)`, plus raw camera/local-head diagnostics. `PipelineBridge` only
squeezes confidence's final singleton for the real official source ABI.

Each Fast3R call uses its first image as its local reference. Original confidence
values pass directly; numerical range compatibility is not probability
calibration. Official PnP uses 100 iterations and first-view global focal
estimation. Real solver failures and incomplete observation are distinct.
The final map reference is the actual officially selected initialization frame,
not necessarily original frame0. Dynamic alignment/fusion matrices are not
individually archived; original local calls, accepted provenance, memory stages
and exported maps/poses remain available.

Both final SfM maps have ambiguous normalized scale. AMB3R metric depth is
retained per native call; it is not used to claim a metrically calibrated final
SfM map. Final intrinsics are the latest actually participating local K because
official SfMemory does not optimize/store global K. Actual foreground projection
checks document that limitation. The shared bundle reader maps K back using the
saved pixel transform; no COLMAP tracks or GGGS compatibility are fabricated.

Completed runs are `experiments/KongQueZhuYu/AMB3R_SfM` and
`experiments/KongQueZhuYu/Fast3R_AMB3R_SfM`, both 208 registered / 0 unmapped.
`MODEL_RUN_REPORT.md` preserves model/run provenance;
`GEOMETRY_RUN_REPORT.md` contains the separate quantitative evaluation;
`persistence_verification.json` records copy scope. The combined solver success
does not establish accurate thin-leaf geometry; use the actual evaluation.
