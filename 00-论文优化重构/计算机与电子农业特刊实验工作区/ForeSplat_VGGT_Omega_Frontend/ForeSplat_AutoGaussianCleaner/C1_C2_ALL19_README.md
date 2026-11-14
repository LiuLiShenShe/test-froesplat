# Frozen C1_raw / C2_raw: all 19 plants

The completed models, render panels and measured results belong under
`experiments/C1_C2_all19/`. Open `summary/index.html` for the full image index,
`summary/all19_results.csv` for numeric results, and `summary/conclusions.md`
for the direct visual comparison and per-plant recommendations.

`fast_clean_c1_c2.py` shares one input read and raw feature extraction, calls
the unchanged `legacy.rules(..., version=2, normalized=False)` for C1, and
uses the already verified C2 raw feature adapter plus
`legacy.rules(..., version=3, normalized=False)` for C2. The historical feature
definitions and thresholds are unchanged. Unused adaptive quantities are not
computed. C2 KNN runs once in batches of 16,384 using the historical k=65
query, self-ID exclusion and NumPy distance ordering, taking the first 20.

Every source model comes from the current scene JSON `model` field. Old
`existing_model` manifest flags are ignored. Each plant runs in its own CPU
process. One `Meter(20)` limits aggregate logical PLY read/write traffic across
both final exports. Both models preserve the native record bytes, unknown
fields, all SH, rotations, opacity, scales, filter_3D, endianness and row order.
No removed/candidate/viewer PLYs, NPZ, per-point CSV, random-I/O memmaps,
training, attribution, or extra geometry are used. A final PLY temporarily
uses `.partial` then is published without copying or replacing existing data.

The default memory estimate budget is 8 GiB. If the full records do not fit,
only compact raw features and keep masks remain in memory; one further
sequential input pass writes both outputs through the same limiter. The
budget is a conservative capacity estimate, not an OS-enforced RSS limit.

Historical KongQueZhuYu IDs must match 443,681 for C1 and 428,967 for C2 before
any other plant runs. Historical PLYs are reused by hard link. Existing C2
results for ChangShouHua1, DouBanLv1 and WangWenCao1 are reused only after
checking source path/size/mtime, the frozen legacy source hash, freshly
recomputed thresholds/reason counts/remaining counts, and the native header.
This avoids generating another equivalent payload. Links are final model
entries and consume no new payload bytes. Reports disclose the original
provenance and zero current export I/O for these reused results.

From `ForeSplat_AutoGaussianCleaner`, use the existing uv-managed environment:

```bash
../.venv/bin/python fast_clean_c1_c2.py \
  --scene configs/scenes/CaoMei1.json --output /absolute/new_empty_plant_folder
```

The recorded serial batch runner is `scripts/run_c1_c2_all19.py`; it preserves
completed stages and refuses to overwrite outputs. Each plant has a
`batch_report.json`, two method `cleanup_report.json` files, and separate
evaluation logs. No visually poor plant interrupts later processing.

Evaluation is separate and uses physical GPU1 only. `fast_ply.evaluate` keeps
the native renderer, camera mapping, foreground photometry and artifact metrics
unchanged. An optional `--view-indices` selects existing cameras explicitly;
the same indices are used for all three models. Other plants use the existing
farthest-view selection with four views. For ChangShouHua1 and WangWenCao1,
known dark failure views from the prior experiment are deliberately included
along with other observed directions. No full 208-view or free-view render
is run. The gray composition stays at 0.5 and exposure is unchanged. Target
RGB reference images are saved separately to check plant preservation.

Timing is measured rather than inferred. Each method contribution comprises
the measured shared read/features time plus its own decision and actual export;
shared time therefore appears in both method reports and must not be summed
to estimate batch cost. Reused exports cost zero this run. The actual batch
wall/CPU totals are reported separately. RSS and application peak I/O belong
to the joint per-plant process, not standalone method runs. GPU rendering time
and its additional PLY reads are excluded. Application one-second throughput
does not measure the disk-device physical peak or guarantee kernel writeback
or other processes stay under the same limit.

Regression validation:

```bash
PYTHONPATH=.. ../.venv/bin/python -m pytest \
  tests/fast_ply/test_c1_c2.py tests/fast_ply/test_c2.py -q
```

Seven checks exercise the legacy masks/bits/parameters, duplicate centers,
small-model KNN boundary, original bytes in both endiannesses, dual-output
one/two-pass equivalence, input preservation and overwrite/memory refusal.

`scripts/report_c1_c2_all19.py` regenerates the tables, Chinese-labelled
overview, focus panels and conclusions from the saved measurements and
`summary/visual_reviews.json`; it does not run cleaning or rendering again.
`scripts/verify_c1_c2_all19.py` audits all 38 output headers, source stamps,
code hashes, reuse provenance, camera identities and image dimensions. Its
result is `summary/verification.json`, with no full PLY payload rescan.
