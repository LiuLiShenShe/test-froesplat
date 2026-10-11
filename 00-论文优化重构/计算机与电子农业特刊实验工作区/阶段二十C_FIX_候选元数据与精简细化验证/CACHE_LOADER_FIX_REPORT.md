# Cache loader fix report

## Root cause and bounded change

Historical `load_existing_candidates()` read raw masks but set every confidence to 0.0,
discarded candidate boxes and score lists, and inferred ordering from filenames.
Actual `score_candidate()` transforms confidence via `sigmoid(6*(sam_score-0.5))` and
includes it with frozen weight 0.5 / denominator 5.5. Restoring metadata is necessary
even where the selected candidate happened to agree.

Production change is confined to candidate metadata import, optional CLI additions,
cache loading, and lossless JSON metadata persistence. No change to `ranking_policy.py`,
R1 formula, weights, mask processing, SAM3/SPNP mechanism or selection strategy.

Existing CLI options remain valid. New optional options:

- `--candidate_metadata_dir`: explicit snapshot directory; JSON schema version 1.
- `--candidate_metadata_strict`: require complete candidate ID coverage, exact RGB and
  mask-file SHA256, finite confidence/threshold, box and unique original candidate order.

The default loader automatically finds JSON snapshots or legacy per-frame CSVs inside
the cache. CSV rows join on image/prompt/instance ID, never ranking or filesystem order.
Legacy CSVs lack hashes and receive `METADATA_UNVERIFIED` warnings; missing metadata
retains historical fallback behavior with an explicit warning and never claims equivalence.
Malformed/contradictory metadata always raises, including non-strict mode.
Strict mode rejects missing metadata, missing instances, corrupt masks/images and scores.

New generations save full float confidence and original metadata in JSON alongside the
legacy CSV, preventing six-decimal CSV loss on future runs. Existing CSV fields/format
remain available. JSON hash validation does not magically restore older discarded digits.

## Tests and frozen behavior

`tests/test_candidate_metadata.py` tests shuffled JSON rows, non-contiguous IDs, original
order under exact ties, legacy CSV identity matching, missing metadata/instances/fields,
duplicate identities/orders, invalid confidence, RGB/mask identity mismatch, strict stop
and explicit external metadata directories. Test-first baseline: 15 failures at the old
loader interface; first patch: 15/15 pass. Final focused/broader totals are recorded in
`TEST_RESULTS.txt` and `PHASE20C_FIX_FINAL_REPORT.md`.

`scripts/replay_scores.py` checks 55 frozen real candidates, all unchanged non-SAM terms,
historical totals to the declared `1e-7` archive tolerance, and selections on all 31 frames.
No previously correct Easy21 selection changes; three original R0 failures remain R1 rescues.
Production V0/V1 logs and selected mask identities are independently checked by the fixed
evaluator, rather than treating an offline score-only run as a production run.

The strict instance-coverage check was tightened during execution after early jobs loaded
their modules. Every actual source snapshot has exact complete coverage; the additional
check changes no mask or score on these runs. Each run records its actual source SHA256;
the final source is checked by focused tests and final frozen CPU replay. A final
metadata-only precision field preserves `ARCHIVED_6_DECIMALS` in future JSON re-saves,
and an unverified provenance status cannot be upgraded merely by writing new hashes.
Neither change alters scoring, selection, refinement or any result mask. The actual
run source hashes remain recorded instead of being relabelled as final-source runs.

Source: production `candidate_metadata.py`, `生成RAP-FSAM3掩膜.py`; local tests, immutable
Phase12/P17A metadata and `CANDIDATE_SCORE_REPLAY.csv`. No library API assumptions used.
