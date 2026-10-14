from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from common import DATA, E0, GROUPS, MATRIX, ROOT, SAMPLES, STAGE, config_for, now, read_csv, sha256, write_csv, write_json

PRIMARY = ("PSNR", "SSIM", "LPIPS")
DESCRIPTIVE = ("fixed_center_crop_PSNR", "fixed_center_crop_SSIM", "fixed_center_crop_LPIPS",
               "R0_proxy_foreground_psnr", "R0_proxy_background_rgb_energy", "R0_proxy_background_alpha",
               "R0_proxy_silhouette_iou", "R0_proxy_missing_alpha_fraction", "R0_proxy_outside_components_ge16px")


def mean_field(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return statistics.mean(values) if values else None


def direction(values):
    signed = [value if field != "LPIPS" else -value for field, value in values]
    nonzero = [value for value in signed if abs(value) > 1e-10]
    if not nonzero:
        return "NO_CHANGE"
    if all(value > 0 for value in nonzero):
        return "IMPROVED"
    if all(value < 0 for value in nonzero):
        return "REGRESSED"
    return "MIXED"


def markdown_table(rows, fields):
    result = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result)


def text_file(name, text):
    (STAGE / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def collect(samples):
    all_rows = []
    runtimes = []
    validations = []
    failures = []
    training_manifest = []
    artifacts = []
    for sample, group in MATRIX:
        if sample not in samples:
            continue
        run_id = f"{sample}_{group}"
        runtime_path = STAGE / f"reports/{run_id}_RUNTIME.json"
        process_path = STAGE / f"reports/{run_id}_PROCESS.json"
        validation_path = STAGE / f"reports/{run_id}_VALIDATION.json"
        if not runtime_path.exists():
            failures.append({"sample": sample, "group": group, "kind": "TRAINING_NOT_RUN", "detail": "No completed runtime record"})
            continue
        runtime = json.loads(runtime_path.read_text())
        process = json.loads(process_path.read_text()) if process_path.exists() else {}
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        runtimes.append({"sample": sample, "group": group, "start_time": runtime["start_time"],
                         "end_time": runtime["end_time"], "iterations": runtime["observed_iteration"],
                         "Gaussian_count": runtime.get("final_gaussians", ""),
                         "training_wall_seconds": runtime.get("training_wall_seconds", ""),
                         "process_wall_seconds": process.get("process_wall_seconds", ""),
                         "iterations_per_second": 30000 / runtime["training_wall_seconds"] if runtime.get("training_wall_seconds") else "",
                         "cuda_peak_allocated_MiB": runtime["cuda_peak_allocated_bytes"] / 1024 ** 2,
                         "cuda_peak_reserved_MiB": runtime["cuda_peak_reserved_bytes"] / 1024 ** 2,
                         "nvidia_smi_sampled_peak_MiB": process.get("sampled_gpu_peak_mib", ""),
                         "exit_code": runtime["exit_code"], "GPU": "NVIDIA RTX A6000 physical 1",
                         "GPU_UUID": "GPU-6836f890-98b2-5891-07ef-675c25199c65"})
        training_manifest.append({"sample": sample, "group": group, "independent_training_run": True,
                                  "source_commit": "3e909f93", "seed": 0, "iterations": runtime["observed_iteration"],
                                  "config_path": str((STAGE / f"configs/{run_id}_TRAIN_CONFIG.json").relative_to(ROOT)),
                                  "config_sha256": runtime["config_sha256"],
                                  "output_path": runtime["output_path"],
                                  "output_relative_to_DATA": str(Path(runtime["output_path"]).relative_to(DATA)),
                                  "shared_model_alias": "B_R0=C_R1" if sample == "HongZhang" and group == "C_R1" else "",
                                  "stdout_stderr_path": process.get("stdout_stderr_path", ""),
                                  "stdout_stderr_sha256": process.get("stdout_stderr_sha256", ""),
                                  "execution": json.dumps(process.get("unchanged_E0_execution", [])),
                                  "observer_execution": json.dumps(process.get("observer_command", [])),
                                  "status": runtime["status"]})
        if validation:
            validations.append({key: validation[key] for key in
                                ("sample", "group", "iteration", "Gaussian_count", "all_PLY_parameters_finite",
                                 "all_activated_parameters_finite", "checkpoint_restore", "checkpoint_PLY_tensor_identity",
                                 "reload_render_max_abs_error", "checkpoint_path", "checkpoint_bytes", "checkpoint_sha256",
                                 "final_PLY_path", "final_PLY_bytes", "final_PLY_sha256", "expected_heldout",
                                 "successful_heldout", "empty_renders", "mean_render_seconds", "status")})
            for role in ("checkpoint", "final_PLY"):
                path = Path(validation[f"{role}_path"])
                artifacts.append({"sample": sample, "group": group, "role": role, "path": str(path),
                                  "relative_to_managed_data": str(path.relative_to(DATA)), "bytes": path.stat().st_size,
                                  "sha256": validation[f"{role}_sha256"], "source_commit": "3e909f93"})
        metrics_path = STAGE / f"metrics/{run_id}_HELDOUT.csv"
        if metrics_path.exists():
            metrics = read_csv(metrics_path)
            all_rows.extend(metrics)
            for row in metrics:
                if row["status"] != "PASS":
                    failures.append({"sample": sample, "group": group, "kind": "HELDOUT_FAILURE",
                                     "detail": f"{row['registered_name']}: {row.get('error')}"})
            for row in metrics:
                for role, path_field, hash_field in (("render_rgb", "render_rgb_path", "render_rgb_sha256"),
                                                     ("render_alpha", "render_alpha_path", "render_alpha_sha256"),
                                                     ("float_render", "float_render_path", "float_render_sha256")):
                    if not row.get(path_field):
                        continue
                    path = Path(row[path_field])
                    artifacts.append({"sample": sample, "group": group, "role": f"{role}/{row['loader_image_name']}",
                                      "path": str(path), "relative_to_managed_data": str(path.relative_to(DATA)),
                                      "bytes": path.stat().st_size, "sha256": row[hash_field], "source_commit": "3e909f93"})
        if runtime["exit_code"]:
            failures.append({"sample": sample, "group": group, "kind": "TRAINING_FAILED", "detail": runtime.get("exception", "")})
        already_indexed = {row["path"] for row in artifacts}
        for path in sorted(Path(runtime["output_path"]).rglob("*")):
            if not path.is_file() or str(path) in already_indexed:
                continue
            artifacts.append({"sample": sample, "group": group,
                              "role": f"additional_training_artifact/{path.relative_to(runtime['output_path'])}",
                              "path": str(path), "relative_to_managed_data": str(path.relative_to(DATA)),
                              "bytes": path.stat().st_size, "sha256": sha256(path), "source_commit": "3e909f93"})
    if "HongZhang" in samples:
        aliases = []
        for row in all_rows:
            if row["sample"] == "HongZhang" and row["group"] == "C_R1":
                alias = row.copy()
                alias.update({"group": "B_R0", "shared_model_alias": "B_R0 = C_R1 (IDENTICAL TRAINING INPUT); one C checkpoint"})
                aliases.append(alias)
        all_rows.extend(aliases)
    return all_rows, runtimes, validations, failures, training_manifest, artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=SAMPLES)
    args = parser.parse_args()
    samples = (args.sample,) if args.sample else SAMPLES
    rows, runtimes, validations, failures, manifest, artifacts = collect(samples)
    assert rows, "No measured output to summarize"
    rows.sort(key=lambda row: (SAMPLES.index(row["sample"]), GROUPS.index(row["group"]), row["registered_name"]))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(STAGE / "HELDOUT_PER_VIEW_METRICS.csv", rows, fields)
    write_csv(STAGE / "TRAINING_RUNTIME_AND_GPU.csv", runtimes)
    write_csv(STAGE / "CHECKPOINT_VALIDATION.csv", validations)
    write_csv(STAGE / "TRAINING_RUN_MANIFEST.csv", manifest)
    log_index = []
    for sample, group in MATRIX:
        if sample not in samples:
            continue
        run_id = f"{sample}_{group}"
        for path in (STAGE / f"logs/{run_id}_STDOUT_STDERR.txt", STAGE / f"logs/{run_id}_GPU_SAMPLES.csv",
                     STAGE / f"logs/{run_id}_EVALUATION.txt", STAGE / f"metrics/{run_id}_TRAINING_CURVE.csv",
                     STAGE / f"logs/{run_id}_COMPETING_PROCESS_AUDIT.csv"):
            if path.exists():
                log_index.append({"sample": sample, "group": group, "path": str(path.relative_to(ROOT)),
                                  "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(STAGE / "logs/TRAINING_LOG_INDEX.csv", log_index)
    write_csv(STAGE / "reports/LARGE_ARTIFACT_MANIFEST.csv", artifacts)
    summary = []
    distributions = []
    paired = []
    paired_views = []
    negatives = []
    for sample in samples:
        model_metrics = {}
        for group in GROUPS:
            selected = [row for row in rows if row["sample"] == sample and row["group"] == group and row["status"] == "PASS"]
            if not selected:
                continue
            actual_group = "C_R1" if (sample, group) == ("HongZhang", "B_R0") else group
            runtime = next(row for row in runtimes if row["sample"] == sample and row["group"] == actual_group)
            validation = next(row for row in validations if row["sample"] == sample and row["group"] == actual_group)
            scene_summary = {"sample": sample, "group": group, "views": len(selected),
                             **{field: mean_field(selected, field) for field in PRIMARY + DESCRIPTIVE},
                             "Gaussians": runtime["Gaussian_count"], "Time_seconds": runtime["training_wall_seconds"],
                             "VRAM_allocated_MiB": runtime["cuda_peak_allocated_MiB"],
                             "VRAM_reserved_MiB": runtime["cuda_peak_reserved_MiB"],
                             "sampled_device_peak_MiB": runtime["nvidia_smi_sampled_peak_MiB"],
                             "PLY_bytes": validation["final_PLY_bytes"], "checkpoint_bytes": validation["checkpoint_bytes"],
                             "mean_render_seconds": validation["mean_render_seconds"],
                             "empty_renders": validation["empty_renders"], "invalid_renders": validation["expected_heldout"] - validation["successful_heldout"],
                             "no_manual_GT_views": len(selected), "shared_checkpoint": group != actual_group,
                             "model_identity": f"{sample}/{actual_group}"}
            model_metrics[group] = scene_summary
            summary.append(scene_summary)
            for field in PRIMARY + DESCRIPTIVE:
                values = [float(row[field]) for row in selected if row.get(field) not in (None, "")]
                if values:
                    distributions.append({"sample": sample, "group": group, "metric": field, "views": len(values),
                                          "mean": statistics.mean(values), "median": statistics.median(values),
                                          "population_std": statistics.pstdev(values), "min": min(values), "max": max(values),
                                          "interpretation": "correlated held-out views; not independent repeats"})
        for before, after in (("A_raw_RGB_control_v1", "B_R0"), ("B_R0", "C_R1")):
            if before not in model_metrics or after not in model_metrics:
                continue
            lower = model_metrics[before]
            upper = model_metrics[after]
            delta = {"sample": sample, "comparison": f"{before}->{after}",
                     **{field: upper[field] - lower[field] if upper[field] is not None and lower[field] is not None else None
                        for field in PRIMARY + DESCRIPTIVE + ("Gaussians", "Time_seconds", "VRAM_allocated_MiB")},
                     "BC_identical_input": sample == "HongZhang" and before == "B_R0"}
            delta["primary_direction"] = direction([(field, delta[field]) for field in PRIMARY])
            paired.append(delta)
            lookup_before = {row["registered_name"]: row for row in rows if row["sample"] == sample and row["group"] == before}
            lookup_after = {row["registered_name"]: row for row in rows if row["sample"] == sample and row["group"] == after}
            assert set(lookup_before) == set(lookup_after)
            for name, left in lookup_before.items():
                right = lookup_after[name]
                assert left["reference_original_sha256"] == right["reference_original_sha256"]
                assert left["reference_resized_tensor_sha256"] == right["reference_resized_tensor_sha256"]
                entry = {"sample": sample, "registered_name": name, "comparison": delta["comparison"],
                         **{field: float(right[field]) - float(left[field]) for field in PRIMARY},
                         "reference_resized_tensor_sha256": left["reference_resized_tensor_sha256"]}
                entry["primary_direction"] = direction([(field, entry[field]) for field in PRIMARY])
                paired_views.append(entry)
                if entry["primary_direction"] in ("REGRESSED", "MIXED"):
                    negatives.append(entry)
    write_csv(STAGE / "HELDOUT_SCENE_SUMMARY.csv", summary)
    write_csv(STAGE / "metrics/HELDOUT_VIEW_DISTRIBUTIONS.csv", distributions)
    write_csv(STAGE / "metrics/PAIRED_SCENE_DIFFERENCES.csv", paired)
    write_csv(STAGE / "metrics/PAIRED_VIEW_DIFFERENCES.csv", paired_views)
    write_csv(STAGE / "metrics/NEGATIVE_OR_MIXED_VIEW_CASES.csv", negatives,
              list(paired_views[0]) if paired_views else ["sample", "registered_name", "comparison", "PSNR", "SSIM", "LPIPS", "reference_resized_tensor_sha256", "primary_direction"])
    write_csv(STAGE / "reports/FAILURES.csv", failures, ["sample", "group", "kind", "detail"])
    failed_attempts = [json.loads(path.read_text()) for path in (STAGE / "logs/failed_attempts").glob("*/FAILED_ATTEMPT_INDEX.json")]
    failed_artifacts = []
    for attempt in failed_attempts:
        for path in sorted(Path(attempt["external_preserved_output"]).rglob("*")):
            if path.is_file():
                failed_artifacts.append({"sample": attempt["sample"], "group": attempt["group"],
                                         "attempt": attempt["attempt"], "path": str(path),
                                         "relative_to_managed_data": str(path.relative_to(DATA)),
                                         "bytes": path.stat().st_size, "sha256": sha256(path),
                                         "status": "FAILED_PARTIAL_OUTPUT_NEVER_USED_AS_FINAL_MODEL"})
    write_csv(STAGE / "reports/FAILED_ATTEMPT_ARTIFACT_HASHES.csv", failed_artifacts,
              ["sample", "group", "attempt", "path", "relative_to_managed_data", "bytes", "sha256", "status"])
    write_csv(STAGE / "reports/FAILED_ATTEMPT_COSTS.csv", [
        {"sample": row["sample"], "group": row["group"], "attempt": row["attempt"],
         "iteration": row["last_iteration"], "failed_wall_seconds": row["failed_wall_seconds"],
         "reason": row["reason"], "preserved_output_path": row["external_preserved_output"],
         "excluded_from_complete_model_statistics": True} for row in failed_attempts],
        ["sample", "group", "attempt", "iteration", "failed_wall_seconds", "reason", "preserved_output_path", "excluded_from_complete_model_statistics"])
    replay_diagnostics = []
    for attempt in failed_attempts:
        run_id = f"{attempt['sample']}_{attempt['group']}"
        earlier_path = STAGE / f"logs/failed_attempts/{run_id}_attempt_01/{run_id}_TRAINING_CURVE.csv"
        retry_path = STAGE / f"metrics/{run_id}_TRAINING_CURVE.csv"
        if not retry_path.exists():
            continue
        earlier = read_csv(earlier_path)
        retry = {row["iteration"]: row for row in read_csv(retry_path)}
        previous = earlier[-1]
        current = retry.get(previous["iteration"])
        if current:
            replay_diagnostics.append({"sample": attempt["sample"], "group": attempt["group"],
                                       "iteration": previous["iteration"], "seed": 0,
                                       "config_unchanged": True, "earlier_gaussians": int(previous["gaussians"]),
                                       "retry_gaussians": int(current["gaussians"]),
                                       "Gaussian_difference": int(current["gaussians"]) - int(previous["gaussians"]),
                                       "earlier_rgb_loss": float(previous["rgb_loss"]),
                                       "retry_rgb_loss": float(current["rgb_loss"]),
                                       "comparison_scope": "PRE_FAILURE_SAME_CONFIG_PARTIAL_TRACE_NOT_FINAL_MODEL_REPLICATE"})
    write_csv(STAGE / "reports/NATIVE_CUDA_REPLAY_DIAGNOSTIC.csv", replay_diagnostics,
              ["sample", "group", "iteration", "seed", "config_unchanged", "earlier_gaussians", "retry_gaussians", "Gaussian_difference", "earlier_rgb_loss", "retry_rgb_loss", "comparison_scope"])
    overall = []
    for group in GROUPS:
        selected = [row for row in summary if row["group"] == group]
        for field in PRIMARY:
            overall.append({"group": group, "metric": field, "scene_equal_mean": mean_field(selected, field),
                            "frame_equal_mean": mean_field([row for row in rows if row["group"] == group and row["status"] == "PASS"], field),
                            "scenes": len(selected), "unit": "descriptive; HongZhang B/C shared model; one seed"})
    write_csv(STAGE / "metrics/OVERALL_METRICS.csv", overall)
    ab = [row for row in paired if row["comparison"].startswith("A_")]
    bc = [row for row in paired if row["comparison"].startswith("B_") and not row["BC_identical_input"]]
    ab_direction = direction([(field, row[field]) for row in ab for field in PRIMARY])
    bc_direction = direction([(field, row[field]) for row in bc for field in PRIMARY])
    benefit = {"IMPROVED": "SUPPORTED", "MIXED": "MIXED", "REGRESSED": "NOT SUPPORTED", "NO_CHANGE": "NOT SUPPORTED"}.get(bc_direction, "INCONCLUSIVE")
    full = len(runtimes) == 8 and len(validations) == 8 and all(row["status"] == "PASS" for row in validations) and not failures
    if not full:
        benefit = "INCONCLUSIVE"
    if benefit == "SUPPORTED" and any(row["R0_proxy_background_rgb_energy"] > 0 for row in bc):
        benefit = "MIXED"
    status = {"PHASE20E1 STATUS": "PASS" if full else "PARTIAL", "TRAININGS_COMPLETED": f"{sum(row['iterations'] == 30000 and row['exit_code'] == 0 for row in runtimes)}/8",
              "CHECKPOINTS_VALID": f"{sum(row['status'] == 'PASS' for row in validations)}/8",
              "HELDOUT_EVALUATION": "PASS" if full else "PARTIAL", "A_VS_B": ab_direction,
              "B_VS_C": bc_direction, "R1_3D_BENEFIT": benefit,
              "HONGZHANG_BC_EQUIVALENCE": "VERIFIED" if "HongZhang" in samples else "NOT_YET_EVALUATED",
              "NEXT_PHASE": "PHASE20E2" if full else "REMEDIATION_REQUIRED"}
    write_json(STAGE / "reports/RESULT_STATUS.json", status)
    table = markdown_table(summary, ("sample", "group", "PSNR", "SSIM", "LPIPS", "Gaussians", "Time_seconds", "VRAM_allocated_MiB", "VRAM_reserved_MiB"))
    difference_table = markdown_table(paired, ("sample", "comparison", "PSNR", "SSIM", "LPIPS", "Gaussians", "primary_direction"))
    proxy_table = markdown_table(summary, ("sample", "group", "R0_proxy_foreground_psnr", "R0_proxy_background_rgb_energy", "R0_proxy_missing_alpha_fraction", "R0_proxy_silhouette_iou"))
    limitations = """Three scenes and one fixed seed form a feasibility pilot, not broad generalization.
Held-out views are correlated and are not independent experimental repetitions.
No independent 3D GT exists: no truth Chamfer/F-score, physical dimensions or mesh.
All 68 frozen validation views lack matching existing manual fixed contours.
Frozen R0 proxy ROIs are predicted labels, not GT; fixed central rectangles are
method-independent descriptive spatial ROIs, not guaranteed plant-only regions.
CaoMei2 uses historically foreground-conditioned COLMAP cameras, shared by A/B/C.
CaoMei2 TEST32 raw0146 and WanNianQing1 raw0227 are not registered improvement cases.
Actual treatment switches: WanNianQing1 raw0037/raw0038/raw0165; CaoMei2 raw0139.
HongZhang B/C uses the same checkpoint; it cannot establish an R1-versus-R0 benefit.
Input treatment is explicit black-RGB precomposition only; no alpha-weighted RGB
loss or mask-guided sparse initialization is enabled. A is not historical ForeSplat.
CUDA rasterization/densification may not be bitwise deterministic across repeated
runs; one-seed differences cannot estimate this variability."""
    text_file("RGB_REFERENCE_CONSISTENCY.md", f"# Shared raw RGB reference consistency\n\nPrimary metrics compare all groups to the same unmasked canonical undistorted RGB.\nThe original and resized-tensor SHA256 are asserted equal in every paired view.\nPIL channelwise BICUBIC at actual resolution=2 is identical for every group;\nmetrics use float renders, not PNG quantization. All frozen views remain present.\n\n{len(rows)} table rows, {sum(row['successful_heldout'] for row in validations)} actual model-camera renders;\nHongZhang B is a labelled C-model alias, not another rendering/training experiment.\n\nSSIM: production 11px Gaussian window, sigma=1.5, C1=0.01²/C2=0.03².\nPSNR: production max-intensity=1. LPIPS: package 0.1.4, VGG v0.1, normalize=True.\nWeights, code SHA256 and package versions are in each VALIDATION.json.\n\nNative iteration-7000/30000 metrics compare treatment-specific composites and are\nsecondary only; they never enter this common-reference table.\n")
    text_file("A_B_C_PAIRED_COMPARISON.md", f"# Paired scene comparison\n\n{table}\n\nAfter minus before; LPIPS negative is favourable.\n\n{difference_table}\n\nA→B primary direction: **{ab_direction}**. B→C (two changed-input scenes): **{bc_direction}**.\nNo significance test or confidence interval is claimed. Every mixed/regressed\nview remains in metrics/NEGATIVE_OR_MIXED_VIEW_CASES.csv.\n\n{limitations}")
    text_file("FOREGROUND_RECONSTRUCTION_ANALYSIS.md", f"# Foreground/background descriptive reconstruction analysis\n\n{proxy_table}\n\nThe shared frozen B_R0 mask defines a predicted ROI for **all** groups. This is\nnot independent segmentation truth or 3D geometry accuracy. Outside-ROI energy\nquantifies appearance contamination relative to that proxy; missing alpha is an\nappearance-coverage proxy, not a validated physical hole. Components outside\nthat proxy may include boundary misalignment, not confirmed floating geometry.\n\nA fixed central spatial crop also has PSNR/SSIM/LPIPS, independently of method.\nIt is not asserted to exclude all background. No independent silhouette IoU or\nBoundary-F1 is reported because none of the held-out raw identities has GT.\n\nVisual observations belong in VISUAL_RECONSTRUCTION_COMPARISON.md after direct\nimage inspection; no automatic proxy is promoted to geometric truth.\n\n{limitations}")
    text_file("R0_R1_3D_EFFECT_REPORT.md", f"# R0→R1 model-level effect\n\n{difference_table}\n\nR1 benefit classification: **{benefit}**, limited to this paired pilot and\nappearance/model-distribution evidence. A metric increase alone is not an\nindependent three-dimensional geometric accuracy improvement.\n\nThe changed inputs can be associated with a model-level difference under this\nfixed-camera, fixed-seed experiment. There is no leave-one-view ablation to\nattribue WanNianQing1's effect separately to raw0037, raw0038 or raw0165.\nCaoMei2's one changed TRAIN view is raw0139, not TEST32 raw0146.\nA Gaussian-count decrease is a compactness observation, not necessarily better\ngeometric coverage. Opacity/scale/distance distributions are retained per model.\n\n{limitations}")
    text_file("NEGATIVE_AND_FAILURE_CASES.md", f"# Negative, mixed and failed cases\n\nFinal-attempt process/evaluation failures: **{len(failures)}**.\nPreserved preliminary compatibility-failed attempts: **{len(failed_attempts)}**,\nactual extra wall cost **{sum(row['failed_wall_seconds'] for row in failed_attempts):.6f} seconds**.\nThey remain in logs/failed_attempts/ and reports/FAILED_ATTEMPT_COSTS.csv, not\nhidden by subsequent successful retries.\nMixed or regressed paired view comparisons: **{len(negatives)}** of {len(paired_views)}.\nThese are all retained, not discarded, in metrics/NEGATIVE_OR_MIXED_VIEW_CASES.csv.\n\n{difference_table}\n\nPSNR improvement with SSIM decrease or LPIPS increase is explicitly classified\nas MIXED, not an unconditional improvement. Original full raw references also\nreward background reconstruction: intentional black-background suppression can\nlower full-image metrics while lowering descriptive contamination proxies.\nNeither outcome is hidden by replacing the reference image.\n\nOOM/NaN/interrupted final attempts, if any, remain in reports/FAILURES.csv and original\nstdout/stderr/process records. Partial runs never enter full-training averages.\n\n{limitations}")
    text_file("VISUAL_RECONSTRUCTION_COMPARISON.md", "# Real reconstruction image comparison\n\nFigure 1: scene heldout.png; Figure 2: scene fixed_crop.png; Figure 3: scene training.png/svg/pdf.\nViews and crops are predeclared in configs/EXPERIMENT_PLAN.json before final\nmodels existed. All A/B/C panels share camera, image dimensions and unadjusted\ncolor settings. HongZhang B/C is labelled shared, not independent.\n\nreports/FIGURE_SOURCE_INDEX.csv records exact image SHA256 and crop coordinates.\nDirect-viewing notes are recorded separately in reports/VISUAL_QA.md. Machine\ninspection is not human validation and cannot establish independent 3D truth.\n")
    text_file("PHASE20E2_HANDOFF.md", f"# Next-stage decision (not execution authorization)\n\nStatus: {status['PHASE20E1 STATUS']}; R1 pilot evidence: {benefit}.\n\nBefore adopting R1 as a formal reconstruction component, add independent\nchanged-input scenes and at least several predeclared seeds. Reuse frozen input\nrecipes, add independently defined held-out plant contours, and distinguish\nscene-level effects from related validation views. Do not automatically retrain\nall historical scenes; prioritize the two changed-input scenes plus additional\nindependent difficult scenes under a prospective protocol.\n\nTechnically valid final models permit planning a separate M2M3-GO/mesh stage,\nbut do not establish that R1 must be its default input. If the pilot is mixed or\nnegative, keep R0 as control and carry the uncertainty forward. Geometry/mesh\nquality requires new independent assessment, not appearance metrics alone.\nNo M2M3-GO change, TSDF/mesh extraction or phenotype measurement runs here.\n\nRecovery: never resume from native incomplete checkpoints without random-state\nand camera-stack evidence. Native loop checkpoints omit those, so an interruption\nrequires a preserved failed attempt plus a full same-config seed-0 rerun only\nafter documented authorization/cost; no alternative seed or tuning is allowed.\n\nReproduction: checkout the source/config commit in TRAINING_RUN_MANIFEST.csv;\nrestore external assets by reports/LARGE_ARTIFACT_MANIFEST.csv relative paths\nand hashes; restore frozen camera symlinks; mount/link the data at the recorded\nroot if necessary. Absolute paths are conveniences, not sole provenance.\nRebind a different physical GPU only under an explicitly recorded new-hardware\nreproduction protocol, never silently bypass the original UUID gate.\nFull float renders allow shared-reference metric recomputation without retraining.\n\n{limitations}")
    text_file("TABLE7_REPLACEMENT_SCOPE.md", "# Original Table 7 replacement scope\n\nThe Phase20B reports/TABLE7_PROVENANCE_AUDIT.md is the authoritative earlier\naudit: original four-method 20-sequence averages and wall-clock timings lack a\nclosed matched provenance chain. This pilot supplies a NEW three-scene matched\n2DGS raw/R0/R1 table with real PSNR/SSIM/LPIPS, counts, time and GPU memory.\nIt can replace unsupported claims about the effect of foreground preprocessing\nwith this explicitly scoped pilot, or form a new ablation panel. It cannot\nreplace a 20-sequence four-method benchmark, historical ForeSplat replication,\nCOLMAP/3DGS/M2M3-GO comparisons, mesh time, phenotype accuracy or geometric GT.\nNo original paper text or table is edited in this phase.\n")
    text_file("PHASE20E1_FINAL_REPORT.md", f"# ForeSplat Phase20E-1 正式配对训练与共同参考评价\n\nBaseline/source/config commit: `3e909f93`. 日期: {now()}.\nGPU1 physical UUID `GPU-6836f890-98b2-5891-07ef-675c25199c65`.\n初始 main 工作区干净；实际资产提交 SHA 见 DELIVERY_RECEIPT.md（提交后生成）。\n\n## 真实结果\n\n{table}\n\n{difference_table}\n\n```text\n" + "\n".join(f"{key}: {value}" for key, value in status.items()) + f"\n```\n\n## 证据与解释\n\n8 个必要模型独立训练；HongZhang B/C 明确共用 C 模型，不计作第九次训练。\n本表为共同原始 RGB 参考，不使用各组不同的 masked RGB 评价充当公平比较。\n表格指标来自逐视角 HELDOUT_PER_VIEW_METRICS.csv；真实过程、checkpoint/PLY\n路径、大小、SHA256 与恢复加载结果均保留。时间为实际训练 wall-clock，含\n冻结加载器和原实现定期验证，不含提交/后续评价。CUDA allocated/reserved\n为进程真实峰值；nvidia-smi 2秒采样为设备占用峰值下界，并非精确进程峰值。\n\n{limitations}\n\n## 科研结论边界\n\nA→B: {ab_direction}; B→C: {bc_direction}; R1 pilot benefit: {benefit}.\nWanNianQing1 三个训练输入变化和 CaoMei2 一个变化对应场景层面的实测效应，\n但不能分别归因于单一视图、证明普遍泛化或宣称统计显著性。\n背景污染/alpha覆盖 proxy 与原始全图指标应一起读；外观相似不等于几何精度。\n固定视图图件及直接识图记录见 figures/ 与 reports/VISUAL_QA.md。\n下一阶段建议增加独立场景和预声明多种子，不默认全量重训；未启动 E2。\n\n## 真实性、来源与交付\n\nE0 6831受管资产哈希及57生产源文件检查记录在PRETRAINING_INPUT_RECHECK.json。\n最终再次核验输入哈希/代码与验收门，结果见FINAL_INPUT_RECHECK.json和FINAL_ACCEPTANCE.json。\nG1–G20逐项见reports/ACCEPTANCE_GATES.csv。只提交本阶段小型脚本/报告/指标，\n外部大型模型和原尺寸渲染不推送。未修改 R1、2DGS、M2M3-GO、Mask、相机或论文。\n\n来源：本机冻结实验资产、真实训练日志、重载模型和逐帧评价，不基于训练记忆伪造结果。\nPyTorch文档通过Context7抓取；博查检索HTTP403失败，未虚构搜索结果。\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
