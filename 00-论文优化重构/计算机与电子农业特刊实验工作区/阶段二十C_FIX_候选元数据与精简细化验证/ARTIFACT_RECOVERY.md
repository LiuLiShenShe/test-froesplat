# Artifact recovery and exact run recipes

Repository-relative stage root:
`00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_FIX_候选元数据与精简细化验证/`.
Actual workspace root is `/data/fj/F2DMAS`.

## Tracked versus local-only

Tracked: production fix/tests/scripts, original-confidence snapshots in `metadata/`,
V4/V5 and replay JSON configs, original run stdout/stderr in `logs/*.txt`, small
production JSON/CSV copies in `small_run_outputs/`, execution records, metric tables,
and `artifact_hashes/*.csv` with full path/size/SHA256 for every actual local artifact.

Local-only (Git ignored): `runs/` masks/intermediate masks/RGB/RGBA/overlays;
`inputs/`, `candidates/`, `smoke_inputs/` symlinks and `annotation_package/rgb/` links.
No image or checkpoint is force-added. Historical masks/GT/checkpoint are untouched.

## Required source archive

- `sam3/sam3.pt`: frozen checkpoint SHA256 in `PROTECTED_SOURCE_HASHES.json`.
- `第三方源码/SAM3-latest`: original frozen SAM3 source; processor source hash recorded.
- Phase12 `P6_raw_baseline` original raw instance masks and metadata.
- Phase17A `P00_Control` original raw instance masks and metadata.
- Phase20C manifest's original RGB and `GT_potted_clean`/challenge masks.
- TEST annotation package additionally needs frozen Phase16 raw RGB and Phase17A
  `U00_Control` candidates, not new guessed masks or pseudo-GT.

Every source path/hash is explicit in `CANDIDATE_METADATA_PROVENANCE.csv`, source
manifest and protected hash JSON. Git contains the small historical score CSVs but
does not contain all ignored original RGB/PNG/checkpoint files. External reproduction
must restore these exact originals from the experiment archive. If they are absent,
the scripts fail; they do not regenerate different SAM candidates and call them replay.

## Commands (run from repository root)

```bash
WORK_DIR='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_FIX_候选元数据与精简细化验证'
PYTHON=/home/test/biosoft/enter/bin/python
$PYTHON "$WORK_DIR/scripts/prepare_fix.py"
$PYTHON "$WORK_DIR/scripts/replay_scores.py"
$PYTHON "$WORK_DIR/scripts/certify_rounding.py"
for GROUP in V0 V1; do
  for DATASET in Easy21 GT10; do
    $PYTHON "$WORK_DIR/scripts/run_fix.py" --group "$GROUP" --dataset "$DATASET"
  done
done
for GROUP in V4 V5; do
  $PYTHON "$WORK_DIR/scripts/run_fix.py" --group "$GROUP" --dataset Easy21 --smoke CaoMei1_0000
  $PYTHON "$WORK_DIR/scripts/run_fix.py" --group "$GROUP" --dataset GT10 --smoke DouBanLv2_0039
done
for GROUP in V4 V5; do
  for DATASET in Easy21 GT10; do
    $PYTHON "$WORK_DIR/scripts/run_fix.py" --group "$GROUP" --dataset "$DATASET"
  done
done
$PYTHON "$WORK_DIR/scripts/evaluate_fix.py"
$PYTHON "$WORK_DIR/scripts/audit_geometry_test32.py"
$PYTHON "$WORK_DIR/scripts/archive_small_outputs.py"
$PYTHON -I -m pytest "$WORK_DIR/tests" -q
$PYTHON "$WORK_DIR/scripts/generate_fix_reports.py"
```

Each production run refuses a nonempty output directory. To repeat on the same
machine, use a separate checkout/stage root or explicitly create new configs and
output paths; never remove or overwrite original runs. The delivered configs store
absolute paths and are hashed; if roots must be relocated, record the rewritten
path-only configs as a separate replay and retain all content hashes.

Runtime uses the same Python/Torch environment as Phase20C (`/home/test/biosoft/enter/bin/python`,
Torch 2.9.1+cu128) and the historical runtime_compat path for `pkg_resources`.
Candidate metadata snapshots are full JSON, with historical scores at archived six
decimals. Precise original tensor confidence is NOT reconstructed. R1 is byte-frozen.

## Local evidence availability

`execution_records/` provides independent status, actual source SHA256, command,
configuration and GPU memory snapshot for each run. `RUN_TIMES.csv` records real
wall/pipeline timing. Small production logs are copied byte-for-byte and individually
hashed in `SMALL_RUN_OUTPUT_INVENTORY.csv`. Large file hashes live in `artifact_hashes/`.

The first V4 Easy21 smoke finished while the runner's stronger smoke validator was
being added. Its saved masks were manually validated by a deterministic CPU command
before any full GPU launch; the JSON records `validation_note`. The first full launch
attempt stopped at the gate before production started. It was restarted only after
all four smoke validations; no output mask was rerun or overwritten.

## Independent validation is NOT a command in this stage

After human annotations and separate review, a later explicit Phase20D may call
`scripts/evaluate_independent_test.py` with frozen predictions/reviews and a new
output path. The pending review gate is intentionally exercised here and blocks;
`annotation_gate_expected_block.txt` documents that no test accuracy was calculated.

Source: actual local artifacts, immutable manifests and runner/evaluator scripts.
