from __future__ import annotations

import json
import subprocess

from phase20d_common import BASELINE, METRICS, PROD, ROOT, WORK, git_bytes, read_csv, relative, sha256, write_json


DELIVERABLES = ["PHASE20D_FINAL_REPORT.md", "TEST32_GT_AUDIT.md", "TEST32_ANNOTATION_REVIEW.csv",
                "TEST32_INPUT_MANIFEST.csv", "TEST32_PER_FRAME_RESULTS.csv", "TEST32_SAMPLE_SUMMARY.csv",
                "R0_R1_INDEPENDENT_COMPARISON.md", "TEST32_SELECTION_CHANGE_AUDIT.md",
                "DIFFICULTY_HUMAN_VERIFICATION.csv", "DIFFICULT_CASE_FIGURE_PLAN.md",
                "FINAL_FOREGROUND_METHOD_LOCK.md", "PHASE20E_INPUT_REQUIREMENTS.md"]


def main():
    if not all((WORK / name).is_file() for name in DELIVERABLES):
        raise ValueError("Requested deliverable or BLOCKED schema missing")
    inputs = read_csv(WORK / "TEST32_INPUT_MANIFEST.csv")
    candidates = read_csv(WORK / "TEST32_CANDIDATE_PROVENANCE.csv")
    reviews = read_csv(WORK / "TEST32_ANNOTATION_REVIEW.csv")
    difficulties = read_csv(WORK / "DIFFICULTY_HUMAN_VERIFICATION.csv")
    if len(inputs) != 32 or len(candidates) != 71 or len(reviews) != 32 or len(difficulties) != 32:
        raise ValueError("Delivery coverage mismatch")
    if any(row["input_status"] != "VERIFIED" for row in inputs):
        raise ValueError("Unverified source input in final delivery")
    if any(row["approval"] != "PENDING" or row["reviewer"] or row["second_reviewer"] or row["annotation_sha256"] for row in reviews):
        raise ValueError("Human decisions unexpectedly populated in no-GT delivery")
    if any(row["status"] != "PENDING_HUMAN_VISUAL_REVIEW" or row["human_difficulty_labels"] or row["difficulty_evidence_json"] for row in difficulties):
        raise ValueError("Human visual evidence unexpectedly populated")
    if list((WORK / "annotation_package/annotations").glob("*.json")) or (WORK / "runs").exists() or (WORK / "figures").exists():
        raise ValueError("Unexpected GT, prediction run or figure in blocked stage")
    results = read_csv(WORK / "TEST32_PER_FRAME_RESULTS.csv")
    summaries = read_csv(WORK / "TEST32_SAMPLE_SUMMARY.csv")
    if len(results) != 64 or any(row["status"] != "BLOCKED" or any(row[metric] for metric in METRICS) for row in results + summaries):
        raise ValueError("Blocked result schemas contain fabricated test values")
    for entry in read_csv(WORK / "SOURCE_HASHES.csv"):
        path = ROOT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Original artifact changed: {path}")
    for name in ("ranking_policy.py", "生成RAP-FSAM3掩膜.py", "candidate_metadata.py"):
        if git_bytes(PROD / name) != (PROD / name).read_bytes():
            raise ValueError(f"Protected production source changed: {name}")
    changes = subprocess.check_output(["git", "-c", "core.quotepath=false", "diff", "--name-only", BASELINE], cwd=ROOT, text=True).splitlines()
    prefix = relative(WORK) + "/"
    if any(not path.startswith(prefix) for path in changes):
        raise ValueError("Changes outside isolated Phase20D stage")
    source_files = list(WORK.rglob("*.py"))
    for path in source_files:
        compile(path.read_text(), str(path), "exec")
    snapshot = json.loads((WORK / "configs/FROZEN_V1_PARAMETERS.json").read_text())
    r0 = json.loads((WORK / "configs/FROZEN_R0_PARAMETERS.json").read_text())
    different_keys = [key for key in snapshot["parameters"] if snapshot["parameters"][key] != r0["parameters"][key]]
    if different_keys != ["ranking_policy"]:
        raise ValueError("R0/R1 parameter snapshots differ beyond frozen ranking policy")
    write_json(WORK / "logs/FINAL_BOUNDARY_AUDIT.json", {
        "status": "PASS", "source_commit": BASELINE, "source_hashes_verified": len(read_csv(WORK / "SOURCE_HASHES.csv")),
        "frames": 32, "sequence_ids": 8, "original_candidates": 71, "manual_annotations": 0,
        "new_scoring_runs": 0, "new_prediction_runs": 0, "accuracy_evaluation_runs": 0,
        "metric_values_in_blocked_schemas": 0, "human_decisions_generated_by_ai": 0,
        "outside_stage_tracked_changes": 0, "original_sources_unchanged": True,
        "no_2dgs_m2m3_colmap_or_manuscript_change": True,
        "r0_v1_parameter_difference": different_keys, "python_syntax_verified": len(source_files),
        "human_gate": "USER_ANNOTATION_REQUIRED", "final_lock": "PROVISIONAL", "phase20e_ready": "NO"})
    print("PASS: original hashes and algorithms unchanged; all independent metrics BLOCKED; stage-only delivery")


if __name__ == "__main__":
    main()
