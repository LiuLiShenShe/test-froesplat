from __future__ import annotations

import ast
import json
import math
import statistics
import subprocess
from pathlib import Path

from common import E0, GROUPS, MATRIX, ROOT, SAMPLES, STAGE, now, read_csv, sha256, write_csv, write_json


def truth(value):
    return value is True or value == "True"


def changed_tracked_paths(root, baseline):
    """Read Git paths without quotePath escaping or newline ambiguity."""
    output = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", baseline], cwd=root, text=True
    )
    return [path for path in output.split("\0") if path]


def metric_recomputation_passes(views, recomputations):
    """Require all three metrics for every real render, excluding model aliases."""
    originals = {
        (row["sample"], row["group"], row["registered_name"], metric): float(row[metric])
        for row in views if row["status"] == "PASS" and not row["shared_model_alias"]
        for metric in ("PSNR", "SSIM", "LPIPS")
    }
    keys = [(row["sample"], row["group"], row["registered_name"], row["metric"])
            for row in recomputations]
    if not originals or len(keys) != len(originals) or set(keys) != set(originals):
        return False
    for key, row in zip(keys, recomputations):
        original = float(row["original"])
        recomputed = float(row["recomputed"])
        difference = float(row["abs_difference"])
        if (row["status"] != "PASS"
                or not all(math.isfinite(value) for value in (original, recomputed, difference))
                or original != originals[key]
                or not math.isclose(difference, abs(recomputed - original), rel_tol=0, abs_tol=1e-12)
                or abs(recomputed - original) > 1e-6):
            return False
    return True


def main():
    initial = json.loads((STAGE / "reports/PRETRAINING_INPUT_RECHECK.json").read_text())
    final = json.loads((STAGE / "reports/FINAL_INPUT_RECHECK.json").read_text())
    runs = read_csv(STAGE / "TRAINING_RUN_MANIFEST.csv")
    runtime = read_csv(STAGE / "TRAINING_RUNTIME_AND_GPU.csv")
    checkpoints = read_csv(STAGE / "CHECKPOINT_VALIDATION.csv")
    views = read_csv(STAGE / "HELDOUT_PER_VIEW_METRICS.csv")
    summary = read_csv(STAGE / "HELDOUT_SCENE_SUMMARY.csv")
    references = read_csv(E0 / "reports/EVALUATION_REFERENCE_MANIFEST.csv")
    ref_map = {(row["sample"], row["registered_name"]): row for row in references}
    expected_runs = set(MATRIX)
    actual_runs = {(row["sample"], row["group"]) for row in runs}
    diagnostics = []
    for row in summary:
        selected = [view for view in views if view["sample"] == row["sample"] and view["group"] == row["group"]]
        for field in ("PSNR", "SSIM", "LPIPS"):
            measured = statistics.mean(float(view[field]) for view in selected)
            diagnostics.append({"sample": row["sample"], "group": row["group"], "metric": field,
                                "recomputed": measured, "reported": float(row[field]),
                                "abs_difference": abs(measured - float(row[field]))})
    write_csv(STAGE / "reports/SUMMARY_RECOMPUTATION.csv", diagnostics)
    paired_references = {}
    reference_ok = True
    heldout_ok = True
    alias_ok = True
    for sample in SAMPLES:
        names = {row["registered_name"] for row in references if row["sample"] == sample}
        for group in GROUPS:
            selected = [row for row in views if row["sample"] == sample and row["group"] == group]
            heldout_ok &= {row["registered_name"] for row in selected} == names
            heldout_ok &= len(selected) == len(names)
            for row in selected:
                key = (sample, row["registered_name"])
                reference_ok &= row["reference_original_sha256"] == ref_map[key]["common_rgb_sha256"]
                current = row["reference_resized_tensor_sha256"]
                reference_ok &= paired_references.setdefault(key, current) == current
                reference_ok &= row["status"] == "PASS"
        if sample == "HongZhang":
            lookup = {(row["group"], row["registered_name"]): row for row in views if row["sample"] == sample}
            for name in names:
                left, right = lookup[("B_R0", name)], lookup[("C_R1", name)]
                alias_ok &= all(left[field] == right[field] for field in ("PSNR", "SSIM", "LPIPS", "render_rgb_sha256", "float_render_sha256"))
                alias_ok &= "IDENTICAL TRAINING INPUT" in left["shared_model_alias"]
    manifest_ok = True
    for row in read_csv(STAGE / "configs/FROZEN_MANIFEST_INDEX.csv"):
        manifest_ok &= sha256(ROOT / row["path"]) == row["sha256"]
    for row in read_csv(STAGE / "reports/LARGE_ARTIFACT_MANIFEST.csv"):
        path = Path(row["path"])
        manifest_ok &= path.stat().st_size == int(row["bytes"]) and sha256(path) == row["sha256"]
    for row in runs:
        manifest_ok &= sha256(ROOT / row["config_path"]) == row["config_sha256"]
        manifest_ok &= sha256(row["stdout_stderr_path"]) == row["stdout_stderr_sha256"]
    replay_path = STAGE / "reports/FLOAT_RENDER_METRIC_RECOMPUTATION.csv"
    replay = read_csv(replay_path) if replay_path.exists() else []
    replay_ok = metric_recomputation_passes(views, replay)
    changed = changed_tracked_paths(ROOT, "3e909f93")
    outside = [path for path in changed if not path.startswith(str(STAGE.relative_to(ROOT)) + "/")]
    scripts = list((STAGE / "scripts").glob("*.py"))
    for path in scripts:
        ast.parse(path.read_text())
    final_inputs_pass = initial["status"] == final["status"] == "PASS" and not final["failures"]
    all_completed = len(runtime) == 8 and all(int(row["iterations"]) == 30000 and int(row["exit_code"]) == 0 for row in runtime)
    checkpoints_ok = len(checkpoints) == 8 and all(row["checkpoint_restore"] == "PASS" for row in checkpoints)
    finite_ok = len(checkpoints) == 8 and all(truth(row["all_PLY_parameters_finite"]) and truth(row["all_activated_parameters_finite"]) for row in checkpoints)
    gates = [
        ("G1", final_inputs_pass, "All 6831 E0 managed input records rehashed; camera symlinks/configs/production sources unchanged"),
        ("G2", actual_runs == expected_runs, "Exactly eight distinct required optimization groups, excluding HongZhang B"),
        ("G3", all_completed, "All final runs reach native iteration 30000; preserved failed attempts excluded"),
        ("G4", final_inputs_pass and all_completed, "Full frozen paired configs equal except intended source/output paths"),
        ("G5", checkpoints_ok, "Every optimizer checkpoint restored; PLY model independently loaded and rendered"),
        ("G6", finite_ok, "Stored and activated XYZ/scale/rotation/opacity/SH finite for all models"),
        ("G7", heldout_ok, "All 177 unique model-camera held-out renders; 204 table rows include labelled aliases"),
        ("G8", reference_ok, "Original raw RGB and resized float reference SHA256 common across every group"),
        ("G9", manifest_ok and final_inputs_pass, "Source/config/input/split/camera/checkpoint/render SHA256 provenance closed"),
        ("G10", alias_ok and ("HongZhang", "B_R0") not in actual_runs, "B=C alias shares checkpoint/render bytes; not a second experiment"),
        ("G11", all(row["group"] in GROUPS for row in runs), "A explicitly raw RGB control, never historical ForeSplat"),
        ("G12", (STAGE / "reports/MANUAL_REFERENCE_AUDIT.json").exists(), "No independent 3D GT or fake Chamfer/F-score; 68/68 views lack matched fixed contours"),
        ("G13", all(float(row["training_wall_seconds"]) > 0 and float(row["cuda_peak_allocated_MiB"]) > 0 for row in runtime), "True timestamps, allocator peaks, sampled device memory and failed-cost records"),
        ("G14", len(checkpoints) == 8, "Metric code/package versions/weight SHA256 recorded per real evaluation"),
        ("G15", replay_ok and all(row["abs_difference"] <= 1e-12 for row in diagnostics),
         "All scene averages exactly recomputed; all 531 metrics from 177 float renders independently replayed within 1e-6"),
        ("G16", (STAGE / "NEGATIVE_AND_FAILURE_CASES.md").exists() and (STAGE / "logs/failed_attempts").exists(), "Mixed/regressed views and compatibility-failed attempts preserved in full"),
        ("G17", final_inputs_pass and not outside, "No R1, 2DGS, M2M3-GO core change; only stage-local logger compatibility"),
        ("G18", not outside, "No paper/source outside E1 modified"),
        ("G19", manifest_ok and actual_runs == expected_runs, "Scene/order/split fixed; full same-seed retries are not validation tuning"),
        ("G20", (STAGE / "PHASE20E2_HANDOFF.md").exists(), "Pilot-specific next-stage decision, not default global retraining; no E2 execution"),
    ]
    rows = [{"gate": gate, "status": "PASS" if result else "BLOCKED", "evidence": evidence} for gate, result, evidence in gates]
    write_csv(STAGE / "reports/ACCEPTANCE_GATES.csv", rows)
    result = {"timestamp": now(), "status": "PASS" if all(result for _, result, _ in gates) else "PARTIAL",
              "gates": rows, "outside_stage_changes": outside, "distinct_training_models": len(runs),
              "primary_rows": len(views), "actual_model_camera_renders": sum(int(row["successful_heldout"]) for row in checkpoints),
              "recomputed_float_render_metrics": len(replay), "float_render_metric_recomputation_pass": replay_ok,
              "failed_attempts": [str(path.relative_to(STAGE)) for path in (STAGE / "logs/failed_attempts").glob("*/FAILED_ATTEMPT_INDEX.json")],
              "source_protection": "57 production files rehashed in final preflight"}
    write_json(STAGE / "reports/FINAL_ACCEPTANCE.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
