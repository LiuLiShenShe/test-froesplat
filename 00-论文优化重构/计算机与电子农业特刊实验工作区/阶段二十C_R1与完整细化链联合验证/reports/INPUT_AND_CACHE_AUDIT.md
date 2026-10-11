# INPUT_AND_CACHE_AUDIT

- Audited frames: 31; READY: 31/31.
- Input RGB, GT, candidate dimensions, candidate symlink targets, and historical score CSV row counts are checked without rewriting any source asset.
- RGB identity uses the frozen Phase12 `frame_inputs` source for Easy21 and Phase17A P00 `00_input_trackA` source for GT10; an invalid legacy raw-frame file is not used as a reference.
- SAM3 repo/checkpoint and GPU snapshot are recorded in `configs/PRECHECK_SNAPSHOT.json`.
- The first SAM3 smoke run exposed a missing `pkg_resources` dependency; the runner uses the Phase20C-local runtime compatibility shim, with production sources unchanged.
- Easy21 COLMAP observations: all frames present.
- GT10 missing COLMAP observations: DouBanLv2_0037, DouBanLv2_0039, DouBanLv2_0041.
- Therefore GT10 V0/V1 is executable; complete GT10 V2/V3 is BLOCKED until the missing sparse observations are supplied and matched.
- Candidate mask aliases and historical score CSVs exist, but production reuse does not replay `sam_score`; this is an explicit equivalence limitation, not a hidden parameter change.
- A6/A7 are OFF in all eight frozen configs; factor invariance is recorded in `configs/FACTOR_AUDIT.csv`.
