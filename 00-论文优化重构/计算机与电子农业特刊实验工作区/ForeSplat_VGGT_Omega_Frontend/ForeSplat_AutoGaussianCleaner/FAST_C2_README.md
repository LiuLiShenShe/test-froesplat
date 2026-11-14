# Frozen C2_raw low-I/O runner

Measured results and all image comparisons: [three-plant report](reports/C2_fast_validation/RESULTS.md).
All three observed-background black-pixel reductions exceed 99%, but
ChangShouHua1 suffers visible dark-leaf/pot over-pruning, and WangWenCao1 loses
nearly the entire foreground in two dark observed views. These results do not
support applying the unchanged rule set to all 19 plants without review.

`fast_clean_c2.py` reuses the unchanged `cleaner/adaptive/legacy.py::rules`
with `version=3, normalized=False`. It retains all eight historical rules.
Only unused adaptive features are omitted. KNN uses the historical 65-neighbor
query, self-ID exclusion, default NumPy distance argsort and first 20 neighbors;
queries remain in batches of 16,384 with eight CPU workers. DC arithmetic keeps
the original field precision, while XYZ, scale exponentiation and opacity use
the historical float64 arithmetic.

Run from `ForeSplat_AutoGaussianCleaner`, using the existing uv-managed environment:

```bash
../.venv/bin/python fast_clean_c2.py \
  --scene configs/scenes/ChangShouHua1.json \
  --output /absolute/path/to/new_empty_result_directory
```

Defaults: combined application PLY read/write limit 20 MiB/s, memory working-set
estimate budget 8 GiB, CPU only. Only `cleaned_gaussians.ply` and
`cleanup_report.json` are produced. Existing files are refused. The final PLY
is staged once and published without copying or overwriting. Original fields,
record bytes, record order and endianness survive unchanged. No PLY SHA scans,
removed PLYs, viewer copies, NPZs, memmaps or scratch models are created.

One input pass is used when the conservative memory estimate fits. Otherwise,
compact raw features and the keep mask are held in RAM, followed by a second
sequential input pass to write retained original records. The budget is a
conservative estimate, not an OS RSS limit; the JSON records actual peak RSS.
CPU time is process CPU time including worker threads. Wall time includes
rate-limiter waits and final fsync. Application trailing-one-second throughput
does not measure or cap disk-device physical peaks or kernel writeback.

Historical verification, using a new empty output directory:

```bash
../.venv/bin/python fast_clean_c2.py \
  --scene configs/scenes/KongQueZhuYu.json \
  --historical-ids experiments/GGGS_Cleaner_Comparison/baselines/C2_raw/original_gaussian_ids.npy \
  --expected-retained 428967 \
  --output /absolute/path/to/new_empty_equivalence_directory
```

The completed run is in `experiments/C2_fast_validation/`. Only ChangShouHua1,
DouBanLv1 and WangWenCao1 were tested after the historical equivalence check.
Their native PLYs and cleanup resource reports are in separate plant folders.
Quality images, metrics, manifests and the recorded source hashes are under
`reports/C2_fast_validation/`.

Independent native GGGS evaluation uses physical GPU1 and the existing
`fast_ply.evaluate` external-model interface, without altering its metrics:

```bash
../.venv/bin/python -m fast_ply.evaluate \
  --scene configs/scenes/ChangShouHua1.json \
  --external-models reports/C2_fast_validation/ChangShouHua1_models.json \
  --skip-baselines --skip-ablations --views 16 --max-side 960 \
  --output /absolute/path/to/new_empty_evaluation_directory
```

Each comparison uses the same observed cameras, images, masks and resolution
for original and cleaned models. These plants have no historical fixed free
camera protocol; their observed-background measurements must not be compared
as if they used KongQueZhuYu's eight free cameras. Foreground masks and skeletons
are structural proxies, not per-leaf semantic ground truth. Evaluation time
and its additional PLY reads are excluded from cleanup performance reports.

Regression checks:

```bash
PYTHONPATH=.. ../.venv/bin/python -m pytest tests/fast_ply/test_c2.py -q
```

Six checks cover exact legacy features/reason bits/decisions, more than 65
duplicate centers, small-model KNN boundary cases, both byte orders, unknown
native fields, one/two-pass parity, logical byte counts, source preservation,
memory-budget rejection and overwrite rejection.
