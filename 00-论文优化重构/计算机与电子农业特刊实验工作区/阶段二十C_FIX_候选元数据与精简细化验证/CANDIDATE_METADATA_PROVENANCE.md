# Candidate metadata provenance

## Identity and immutable sources

- Easy21: Phase12 `P6_raw_baseline`, 21 original RGB frames and 25 P6 instances.
- GT10: Phase17A `02_trackA_variants/P00_Control`, 10 original RGB frames and 30 P6 instances.
- All 55 instances are joined by `(image_name, prompt_id, explicit instance_id)`.
  IDs come from original inference's instance enumeration and the per-instance filename,
  not from filesystem sorting or candidate ranking. `排名` is diagnostic rank, not order.
- Snapshot files: `metadata/{Easy21,GT10}/候选元数据_<sample>_<frame>.json`.
  Every snapshot preserves score, box (including legitimate negative predicted coordinates),
  threshold, original instance order, source image name/RGB SHA256 and mask-file SHA256.
- Each frozen alias matches the original raw mask's SHA256 and Phase20C manifest hash;
  candidate pixels are binary uint8 `{0,255}`, original resolution 2160×3840.
  Actual mask area equals the identity-keyed historical detail row to `1e-14`.
- Original generation input RGB and Phase20C RGB have identical SHA256. The selected
  source score row is independently checked by image/prompt/actual mask area; no position
  in a directory or score CSV is used as the identity join. Ambiguity stops preparation.

## Confidence precision — explicit limitation

Historical `候选评分明细.csv` stores `SAM3分数` rounded to **six decimal places**.
Historical `提示词评分.csv` also formats the full confidence list to six decimals.
These are real saved confidences, not zero placeholders, inferred confidence or newly
generated candidates. We restore exactly these archived decimals, and never invert a
score formula, snap to a presumed dtype, or invent extra confidence digits.

All 55 original boxes/IDs/thresholds/orders and archived confidences are VERIFIED.
The pre-serialization tensor confidences were **not saved** at full precision. Therefore
`CANDIDATE SCORE PROVENANCE: PARTIAL` applies to the strongest byte-exact historical
claim; it does not mean any candidate is missing a matched archived score/box.
Fresh SAM3 inference could create a different candidate set and would be a NEW_CANDIDATE_SET,
not a lossless recovery of an old tensor. No unnecessary text-prompt regeneration is run.

## Scoring check

`CANDIDATE_SCORE_REPLAY.csv` includes all real confidences, sigmoid `q_sam`, risk penalty,
R0/R1 primary scores, mask/RGB identities, candidate F1/IoU and historical score errors.
The maximum error against original historical total is **6.814438091584663e-08**.
`SCORE_EQUIVALENT` in this delivery means numeric scoring equivalent **at the explicitly
declared archived-precision tolerance 1e-7**, not byte-identical unrounded tensor scoring.
That tolerance follows the existing six-decimal serialization; it is not a ranking weight,
selection change or an optimized threshold. Scoring formulas and weights remain frozen.
Analytical maximum primary-score perturbation is `5e-7 × 1.5 × 0.5 / 5.5 =
6.818181818e-8` (sigmoid derivative bound 1.5); the reported observed maximum is
within this serialization bound. Unrounded score provenance remains partial.

Restored V0/R0 and V1/R1 selection changes versus Phase20C: **0/62** frame/policy pairs.
Both policies are additionally checked against Phase17C's offline frozen reference.
`ROUNDING_SELECTION_CERTIFICATE.csv` bounds every original confidence within its
six-decimal rounding interval ±5e-7 without assigning a guessed value. Monotonicity
of the frozen sigmoid gives a worst-case winner/runner-up margin; all **62/62**
frame/policy selections are certified, minimum multi-candidate margin **0.01414987528**.
Thus the missing extra digits cannot change any observed R0/R1 selection, although
unrounded historical primary-score bytes still cannot be recovered.
Report status must never erase the six-decimal limit. Any genuinely unmatched future
score/box/hash is `METADATA_UNVERIFIED` and strict replay stops.

## Reproduction and recovery

Run `scripts/prepare_fix.py` to recreate aliases and verify immutable sources, then
`scripts/replay_scores.py`. `CANDIDATE_METADATA_PROVENANCE.csv` records the source paths,
CSV/parameter hashes, mask hashes and image hashes. If original ignored RGB/candidate
files are unavailable outside this machine, the original frozen data must be restored
from the experiment archive; Git contains metadata snapshots but does not fabricate masks.
See `ARTIFACT_RECOVERY.md` for complete commands and artifact hashes.

Sources: Phase12/P17A original `候选评分明细_*.csv`, `提示词评分.csv`, `参数.json`;
Phase20C `PHASE20C_INPUT_MANIFEST.csv`; this delivery's provenance/replay CSVs.
