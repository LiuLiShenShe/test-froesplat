# Frozen TEST32 R0/R1 commands — not executed

Status: `BLOCKED_BY_GT_GATE`. These commands are a recipe only. No command in this
file has been used to create a prediction or accuracy result in this completion.

## Preconditions

- Both review CSVs contain actual human provenance, timestamps, target descriptions,
  cube decisions, frame-specific visual evidence and `APPROVED` status.
- Every final annotation is an original-coordinate LabelMe `polygon` export containing
  `potted_plant`, `plant`, `pot`, and human-confirmed `blue_cube` only when present.
- Final annotation SHA256 is recorded in both tables and the unchanged gate passes.
- External RGB and all 71 U00_Control P6 masks exist at the frozen paths.

## Gate

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_独立GT与困难场景验证'
PYTHON=/home/test/biosoft/enter/bin/python
$PYTHON "$STAGE/scripts/validate_annotations.py" \
  --output "$STAGE/completion/logs/annotation_gate_after_archive_01.json"
```

Current blocked history is retained in logs/annotation_gate_completion.json. Use a new
filename for each preflight, never overwrite old evidence. Do not run the evaluator
when the command exits `2`.

## Frozen production smoke and full run

The existing production entry must be called directly; do not copy its ranking code.
Use new empty output roots and preserve all Phase20C-FIX runs.

```bash
PROD='00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py'
INPUTS="$STAGE/inputs/TEST32"
CANDIDATES="$STAGE/candidates/TEST32"
META="$STAGE/metadata/TEST32"
OUT="$STAGE/completion/predictions"
export PYTHONPATH="$PWD/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证/scripts/runtime_compat${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
COMMON=(--sam3_repo '第三方源码/SAM3-latest' --sam3_checkpoint 'sam3/sam3.pt'
  --device cpu --candidate_mode per_instance --prompt_list P6 --default_prompt_id P6
  --prompt_selection_mode single --score_weights 'area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5'
  --candidate_source_dir "$CANDIDATES" --candidate_metadata_dir "$META"
  --candidate_metadata_strict --reuse_existing_candidates)

# The installed parser has no smoke-frame flag. Each input directory must contain
# only the frozen RGB symlink for its named frame; record its hash before running.
$PYTHON "$PROD" "${COMMON[@]}" --input_dir "$STAGE/completion/smoke_inputs/CaoMei2_0146" \
  --output_dir "$OUT/smoke_base_CaoMei2_0146" --ranking_policy base
$PYTHON "$PROD" "${COMMON[@]}" --input_dir "$STAGE/completion/smoke_inputs/CaoMei2_0146" \
  --output_dir "$OUT/smoke_r1_CaoMei2_0146" --ranking_policy r1
$PYTHON "$PROD" "${COMMON[@]}" --input_dir "$STAGE/completion/smoke_inputs/WanNianQing1_0227" \
  --output_dir "$OUT/smoke_base_WanNianQing1_0227" --ranking_policy base
$PYTHON "$PROD" "${COMMON[@]}" --input_dir "$STAGE/completion/smoke_inputs/WanNianQing1_0227" \
  --output_dir "$OUT/smoke_r1_WanNianQing1_0227" --ranking_policy r1

# Only after all four smoke identity/score/output checks pass:
$PYTHON "$PROD" "${COMMON[@]}" --input_dir "$INPUTS" --output_dir "$OUT/r0" --ranking_policy base
$PYTHON "$PROD" "${COMMON[@]}" --input_dir "$INPUTS" --output_dir "$OUT/r1" --ranking_policy r1
```

No new candidate inference is permitted. The output wrapper must reject non-empty
roots; the exact production parser remains authoritative.

## Evaluation

One completion evaluator must import the fixed polygon rasterizer and frozen metric
implementation, use identical candidates for R0/R1, preserve raw and cleanup masks,
fail on ID/hash/dimension mismatch, and write all requested per-frame, per-sequence,
summary, oracle and difficulty outputs. This completion has no numeric placeholders.
