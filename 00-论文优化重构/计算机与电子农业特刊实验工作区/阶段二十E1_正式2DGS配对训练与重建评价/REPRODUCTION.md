# Reproduction and recovery

## Frozen inputs and model identity

Checkout baseline `3e909f93` and the E1 delivery commit in `DELIVERY_RECEIPT.md`.
E0 manifests are indexed by repo-relative path and SHA256 in
`configs/FROZEN_MANIFEST_INDEX.csv`; complete byte-identical training configs are
copied to `configs/`. Managed data are restored from the E0 manifest, including
explicit common camera symlinks. `reports/LARGE_ARTIFACT_MANIFEST.csv` includes
both absolute convenience paths and paths relative to the managed-data root.
Use these identities, not filenames alone, to restore checkpoints and renders.

The original machine binds physical GPU1 by UUID. Exact-machine replay uses:

```bash
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PYTHON=/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python
STAGE=/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十E1_正式2DGS配对训练与重建评价
"$PYTHON" "$STAGE/scripts/preflight.py"
"$PYTHON" "$STAGE/scripts/run_matrix.py"
```

Run in a fresh output namespace only after a documented replay configuration
freeze, preserving the original experiment. The wrapper intentionally rejects
existing outputs, changed config hashes or different physical GPU UUIDs. A
different-machine replay must explicitly record path/hardware relocation and
regenerate only a separate replay freeze; it must not rewrite original evidence.
The data are not distributable through Git alone: restore lawful original assets
and verify the manifest. No silent download or fabricated checkpoint is allowed.

## Evaluation replay without training

`recompute_metrics.py` verifies stored float render/reference SHA256, recomputes
full RGB PSNR/SSIM/LPIPS with the pinned local versions and compares to per-view
CSV. It does not quantize through PNG and never uses treatment-specific RGB GT.

```bash
"$PYTHON" "$STAGE/scripts/recompute_metrics.py"
"$PYTHON" "$STAGE/scripts/summarize.py"
"$PYTHON" "$STAGE/scripts/figures.py"
"$PYTHON" "$STAGE/scripts/preflight.py" --final
"$PYTHON" "$STAGE/scripts/acceptance.py"
```

The final regression tests use Python's standard-library runner in this same
frozen training environment (which has no `pytest` installed):

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest discover -s "$STAGE/tests" -p 'test_*.py' -v
```

The final acceptance checks require full replay of all 531 primary metric
values from the 177 distinct float renders. A limited `--sample`, `--group` or
`--limit` diagnostic replay overwrites the replay CSV and deliberately does not
satisfy G15; rerun the full command before acceptance. HongZhang B's 27 alias
rows are checked for byte/metric identity but do not count as extra renders.
Git path checks use NUL separation, so staging Chinese-named files cannot
misclassify them as outside-stage changes.

Independent re-rendering uses the actual group's frozen config, final PLY and
checkpoint; no model borrows another treatment's training camera/data directory.
HongZhang B is only a C-model table/figure alias. Re-render to a separately indexed
directory to preserve the first render; the evaluator refuses silent overwrite.

## Failure handling

The actual initial Matplotlib logging failure and its costs are preserved in
`logs/failed_attempts/` and external `phase20e1_failed_attempts/`. Stage-local
RGB-buffer compatibility adds no optimizer/loss changes; unit tests prove buffer
semantics. Failed A/B attempts were fully restarted with frozen seed 0/config,
not resumed from RNG-incomplete checkpoints. Complete-model tables exclude these
partial attempts while explicitly reporting additional failed wall cost.

Native saved optimizer checkpoints do not contain random camera-stack state;
an interrupted run must not be advertised as an exactly equivalent resume.
Keep the full failure and perform a new documented same-config run if needed.
No alternative seed, per-group parameter tuning or favourable-view replacement.
