# Frozen v8 batch continuation

## Evidence and scope

All four development experiments and their direct browser reviews exist. The
frozen protocol retains WangWenCao1 as a visual failure. CaoMei1 has completed
both native stages; the existing serial driver is retiring large trial files.
GPU1 waits throughout this CPU/disk postprocessing. The remaining heldout
models must use the unchanged v8 sources, configuration and inventory.

## Implementation steps

1. Add a bounded scheduler to `run_historical_two_stage.py`: exactly one cleaning
   child, with at most one CPU postprocessing job overlapping the next child.
   Drain postprocessing before success, propagate both kinds of failure, and
   retain a serial option. Scheduling changes do not change frozen identities.
   Verify execution order, overlap, the outstanding-job bound and both failure
   paths with lightweight fake jobs. Expected engineering time: 20–30 minutes.
2. Record GPU and postprocessing phases separately. Publish numeric completion
   before CPU retirement/capture and preserve the detailed final report when
   updating batch progress. Ensure interruption/error records cannot leave a
   dead GPU child advertised as running. Verify with temporary output fixtures.
3. Resume the existing v8 batch only after an orderly driver handover. Existing
   COMPLETE files are validated by the public CLI and not rerun. Confirm the
   frozen-source/config hashes, GPU1 placement, actual child and growing logs.
4. Inspect newly captured actual screenshots, record candid visual findings,
   select reviewed files, audit lineage/hashes and refresh the report/gallery.
   Never turn screenshot generation into a claim of direct visual review.

## Risks and verification

Disk traffic is shared; use only one postprocessing worker and allow at most
one next scene, preventing an unbounded queue of large trial files. Never stop
a native cleaning child merely to deploy scheduler changes. Preserve failed
attempts, original/history PLYs, accepted rounds and exact edit evidence.
GPU stage timing and wall time including retirement/capture remain distinct.
No speedup claim is valid until both are measured. No heldout or universal
visual-success claim is inferred from passing unit tests or a running batch.

Source: actual local run records. Bocha search returned HTTP403 quota exhaustion.
