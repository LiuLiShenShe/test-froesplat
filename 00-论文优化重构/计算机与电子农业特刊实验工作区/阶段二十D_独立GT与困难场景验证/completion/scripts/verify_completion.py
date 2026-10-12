from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path


STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE / "scripts"))
from phase20d_common import FIX, METRICS, PROD, ROOT, read_csv, relative, sha256, unique_index, write_csv, write_json
import validate_annotations as original_gate


OBJECTIVE_FIELDS = {"annotation_sha256", "human_difficulty_labels", "image_width", "image_height", "geometry_validation_status"}
REQUIRED_FILES = (
    "PHASE20D_COMPLETION_FINAL_REPORT.md", "GT32_GEOMETRY_REPORT.csv", "HUMAN_REVIEW_ARCHIVE_REPORT.md",
    "GT_GATE_RESULT.json", "R0_R1_TEST32_COMMANDS.md", "TEST32_PER_FRAME_EVALUATED.csv",
    "TEST32_PER_SEQUENCE_EVALUATED.csv", "TEST32_SUMMARY_EVALUATED.csv", "TEST32_SWITCH_CASE_AUDIT.md",
    "DIFFICULTY_STRATIFIED_RESULTS.csv", "R0_R1_INDEPENDENT_COMPARISON.md", "FINAL_FOREGROUND_METHOD_LOCK.md",
    "PHASE20E_HANDOFF.md", "figures/README.md")


def baseline_csv(path):
    saved = subprocess.check_output(["git", "show", f"22073ca5:{relative(path)}"], cwd=ROOT)
    return list(csv.DictReader(io.StringIO(saved.decode("utf-8-sig"))))


def check_table_preservation(before, after):
    previous, current = unique_index(before, "frame_id"), unique_index(after, "frame_id")
    if len(before) != 32 or len(after) != 32 or set(previous) != set(current):
        raise ValueError("Review tables must keep exactly the same 32 identities")
    for stem, row in previous.items():
        for field, value in row.items():
            if field in OBJECTIVE_FIELDS and not value:
                continue
            if current[stem].get(field) != value:
                raise ValueError(f"Existing record overwritten: {stem}/{field}")
        if set(current[stem]) - set(row) - OBJECTIVE_FIELDS:
            raise ValueError("Unexpected new review field")


def require_blank_metrics(rows, fields):
    if any(row.get(field, "") != "" for row in rows for field in fields):
        raise ValueError("Blocked output must not contain unmeasured numeric metrics")


def verify(stage, output):
    completion = stage / "completion"
    for name in REQUIRED_FILES:
        if not (completion / name).is_file():
            raise ValueError(f"Missing required deliverable: {name}")
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    expected = {row["frame_id"] for row in frames}
    if len(frames) != 32 or len(expected) != 32:
        raise ValueError("Frozen frame coverage changed")
    original_gate.verify_frozen_inputs(stage, frames)
    tables = []
    for name in ("TEST32_ANNOTATION_REVIEW.csv", "DIFFICULTY_HUMAN_VERIFICATION.csv"):
        path = stage / name
        current = read_csv(path)
        check_table_preservation(baseline_csv(path), current)
        if {row["frame_id"] for row in current} != expected:
            raise ValueError("Review/frozen frame identity disagreement")
        tables.append({"path": relative(path), "sha256_after": sha256(path), "existing_fields_preserved": True})
    archive = read_csv(completion / "GT32_SOURCE_ARCHIVE.csv")
    geometry = read_csv(completion / "GT32_GEOMETRY_REPORT.csv")
    for rows in (archive, geometry):
        if len(rows) != 32 or len(unique_index(rows, "frame_id")) != 32 or {row["frame_id"] for row in rows} != expected:
            raise ValueError("GT archive/geometry coverage mismatch")
    for row in archive:
        source, canonical = ROOT / row["source_json"], ROOT / row["canonical_json"]
        if sha256(source) != row["source_sha256_before"] or sha256(source) != row["source_sha256_after"] or sha256(canonical) != row["canonical_sha256"]:
            raise ValueError(f"Original annotation byte identity changed: {row['frame_id']}")
    backup = read_csv(completion / "GT32_LOCAL_BACKUP_INDEX.csv")
    if len(backup) != 32 or len(unique_index(backup, "frame_id")) != 32 or {row["frame_id"] for row in backup} != expected:
        raise ValueError("Private byte-exact backup must preserve every frozen human export")
    for row in backup:
        if sha256(ROOT / row["local_backup"]) != row["source_sha256"] or row["backup_sha256"] != row["source_sha256"]:
            raise ValueError("Manual export backup identity mismatch")
    gate_path = completion / "logs/annotation_gate_completion.json"
    saved_gate = json.loads(gate_path.read_text())
    fresh_gate = original_gate.preflight(stage)
    if saved_gate != fresh_gate or saved_gate != json.loads((completion / "GT_GATE_RESULT.json").read_text()):
        raise ValueError("Gate result cannot be reproduced from the archived sources")
    if saved_gate["evaluation_allowed"] or saved_gate["accuracy_metrics_computed"] or saved_gate["predictions_accessed"]:
        raise ValueError("Blocked delivery unexpectedly claims evaluation or prediction access")
    status = json.loads((completion / "COMPLETION_STATUS.json").read_text())
    if status["phase20d_completion"] != "PARTIAL" or status["foreground_method"] != "PROVISIONAL" or status["ready_for_phase20e"] != "NO":
        raise ValueError("Blocked stage cannot be marked PASS/LOCKED/ready")
    wide_fields = [f"{group}_{metric}" for group in ("r0", "r1", "paired_delta") for metric in METRICS]
    per_frame = read_csv(completion / "TEST32_PER_FRAME_EVALUATED.csv")
    if len(per_frame) != 32 or {row["frame_id"] for row in per_frame} != expected:
        raise ValueError("Blocked result frame coverage mismatch")
    require_blank_metrics(per_frame, wide_fields + ["r0_selected_id", "r1_selected_id", "oracle_id", "selection_changed"])
    per_sequence = read_csv(completion / "TEST32_PER_SEQUENCE_EVALUATED.csv")
    if len(per_sequence) != 8 or {row["sequence"] for row in per_sequence} != {row["sequence"] for row in frames}:
        raise ValueError("Blocked result sequence coverage mismatch")
    summary = read_csv(completion / "TEST32_SUMMARY_EVALUATED.csv")
    for rows in (per_sequence, summary):
        require_blank_metrics(rows, wide_fields)
    difficulty = read_csv(completion / "DIFFICULTY_STRATIFIED_RESULTS.csv")
    require_blank_metrics(difficulty, ["r0_f1", "r1_f1", "paired_delta", "r0_failure_count", "r1_failure_count"])
    if any(row["evaluation_status"] != "BLOCKED_NOT_EVALUATED" for rows in (per_frame, per_sequence, summary, difficulty) for row in rows):
        raise ValueError("All unmeasured result rows must be explicitly BLOCKED")
    protected = [PROD / name for name in ("ranking_policy.py", "candidate_metadata.py", "生成RAP-FSAM3掩膜.py")]
    protected += [stage / "scripts/validate_annotations.py", FIX / "scripts/evaluate_independent_test.py",
                  stage / "TEST32_INPUT_MANIFEST.csv", stage / "configs/FROZEN_TEST_SPEC.json",
                  ROOT / "2d-gaussian-splatting-main/train.py", ROOT / "2d-gaussian-splatting-main/render.py"]
    hashes = []
    for path in protected:
        if path.read_bytes() != subprocess.check_output(["git", "show", f"22073ca5:{relative(path)}"], cwd=ROOT):
            raise ValueError(f"Protected production/Gate/training file changed: {path}")
        hashes.append({"path": relative(path), "sha256": sha256(path), "baseline": "22073ca5", "unchanged": "YES"})
    original_sources = read_csv(stage / "SOURCE_HASHES.csv")
    for row in original_sources:
        path = ROOT / row["path"]
        if sha256(path) != row["sha256"] or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Protected archived source changed: {path}")
    changed = subprocess.check_output(["git", "-c", "core.quotePath=false", "diff", "--name-only", "22073ca5"], cwd=ROOT).decode().splitlines()
    prefix = relative(stage) + "/"
    permitted = {prefix + name for name in ("TEST32_ANNOTATION_REVIEW.csv", "DIFFICULTY_HUMAN_VERIFICATION.csv")}
    if any(path not in permitted and not path.startswith(prefix + "completion/") for path in changed):
        raise ValueError("Changes outside the permitted completion/review scope")
    write_csv(output.with_name(output.stem + "_CODE_HASHES.csv"), hashes)
    result = {"status": "PASS_PREPARATION_PROTECTION_CHECK_ONLY", "frozen_frames": 32, "frozen_sequences": 8,
              "original_candidates": 71, "original_source_hashes_checked": len(original_sources),
              "original_annotation_hashes_checked": len(archive), "unchanged_protected_code_files": len(protected),
              "local_byte_exact_backups_checked": len(backup),
              "manual_d_cells_preserved": 160, "review_tables": tables, "gate_reproduced": True,
              "evaluation_allowed": False, "predictions_executed": False, "accuracy_metrics_computed": False,
              "independent_cpu_evaluation_test": "BLOCKED_GT_GATE", "metric_recompute_consistency_test": "BLOCKED_GT_GATE",
              "blank_metric_rows_checked": len(per_frame) + len(per_sequence) + len(summary) + len(difficulty)}
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Verify honest blocked output, preserved human work and source boundaries")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.stage_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
