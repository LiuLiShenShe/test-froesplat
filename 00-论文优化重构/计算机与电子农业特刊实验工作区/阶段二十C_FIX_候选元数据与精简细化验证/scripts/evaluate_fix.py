from __future__ import annotations

import json
import math
import hashlib
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from fix_common import OLD, ROOT, WORK, load_module, manifest, read_csv, relative, sha256, write_csv, write_json


METRICS = ["selected_f1", "final_f1", "final_iou", "final_hd95", "final_boundary_f1", "final_leakage", "oracle_regret"]


def mask(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L")) > 0


def pixel(mask_array, gt):
    true_positive = np.logical_and(mask_array, gt).sum()
    denominator = mask_array.sum() + gt.sum()
    union = np.logical_or(mask_array, gt).sum()
    return float(2 * true_positive / denominator) if denominator else 1.0, float(true_positive / union) if union else 1.0


def sample_equal(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-shard", choices=["V0", "V1", "V4", "V5"])
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    metric_module = load_module("phase20c_fix_legacy_metrics", OLD / "scripts/compute_phase20c_metrics.py")
    evaluate = metric_module.load_segmentation_metrics()
    replay = read_csv(WORK / "R0_R1_SELECTION_REPLAY.csv")
    candidate_replay = {value["candidate_id"]: value for value in read_csv(WORK / "CANDIDATE_SCORE_REPLAY.csv")}
    replay_index = {(value["dataset"], value["sample"], value["frame"], value["run"]): value for value in replay}
    frame_rows, old_rows, stage_rows, runtime_rows = [], [], [], []
    groups = [args.group_shard] if args.group_shard else ["V0", "V1", "V4", "V5"]
    for dataset in ("Easy21", "GT10"):
        for group in groups:
            output = WORK / "runs" / group / dataset
            status_path = output / "runner_status.json"
            if not status_path.exists() or json.loads(status_path.read_text())["returncode"] != 0:
                raise RuntimeError(f"Run not complete: {output}")
            status = json.loads(status_path.read_text())
            log = json.loads((output / "运行日志.json").read_text())
            runtime_rows.append({"dataset": dataset, "group": group, "wall_sec_including_load_and_io": status["wall_sec"],
                                 "pipeline_elapsed_sec": log["elapsed_sec"], "gpu_used": int(group in ("V4", "V5")),
                                 "runtime_comparability": "CPU_BASELINE_VS_GPU_SPnp; WALL_TIME_NOT_KERNEL_TIME",
                                 "output": str(output)})
    evaluation_frames = [] if args.merge_shards else manifest()
    for frame in evaluation_frames:
        dataset, stem, sample, number = frame["dataset"], frame["stem"], frame["sample_id"], frame["frame"]
        gt = mask(ROOT / frame["gt_path"])
        candidates = []
        for raw in sorted((ROOT / frame["candidate_alias_dir"]).glob(f"mask_{stem}_*.png")):
            candidate_mask = mask(raw)
            f1, iou = pixel(candidate_mask, gt)
            candidates.append((raw.stem.removeprefix("mask_"), raw, f1))
        oracle = max(value[2] for value in candidates)
        evaluation_cache = {}
        frame_groups = groups + (["V3_HISTORY"] if args.group_shard in (None, "V4") else [])
        for group in frame_groups:
            output = WORK / "runs" / group / dataset if group != "V3_HISTORY" else OLD / "runs/V3" / dataset
            if not (output / "最终掩膜" / f"mask_{stem}.png").exists():
                continue
            selected = mask(output / "选择后掩膜" / f"mask_{stem}.png")
            spnp = mask(output / "正负提示细化掩膜" / f"mask_{stem}.png")
            final = mask(output / "最终掩膜" / f"mask_{stem}.png")
            selected_f1, _ = pixel(selected, gt)
            matching = [candidate_id for candidate_id, raw, _ in candidates if np.array_equal(selected, mask(raw))]
            if len(matching) != 1:
                raise ValueError(f"Nonunique selected mask identity: {dataset}/{group}/{stem}")
            expected_policy = "restored_R1" if group in ("V1", "V4", "V3_HISTORY") else "restored_R0"
            reference = replay_index[(dataset, sample, number, expected_policy)]
            selection_same = matching[0] == reference["selected_candidate_id"]
            if group != "V3_HISTORY" and not selection_same:
                raise ValueError(f"Production selection disagrees with CPU replay: {group}/{stem}")
            final_identity = hashlib.sha256(final).hexdigest()
            if final_identity not in evaluation_cache:
                evaluation_cache[final_identity] = evaluate(gt, final)
            values = evaluation_cache[final_identity]
            selection_log = next(value for value in read_csv(output / "提示词选择.csv") if value["图像"] == f"{stem}.jpg")
            changed = int(np.logical_xor(selected, spnp).sum())
            historical_spnp_path = OLD / "runs/V3" / dataset / "正负提示细化掩膜" / f"mask_{stem}.png"
            historical_spnp_equal = int(np.array_equal(spnp, mask(historical_spnp_path))) if group in ("V4", "V5") and historical_spnp_path.exists() else ""
            row = {"dataset": dataset, "sample": sample, "frame": number, "group": group,
                   "selected_candidate_id": matching[0], "primary_score": candidate_replay[matching[0]]["phase20c_zero_score_R1_primary"] if group == "V3_HISTORY" else reference["primary_score"],
                   "risk_penalty": reference["risk_penalty"], "q_sam": reference["q_sam"] if group != "V3_HISTORY" else 1 / (1 + math.exp(3)),
                   "selected_f1": selected_f1, "final_f1": float(values["f1"]), "final_iou": float(values["iou_fg"]),
                   "final_hd95": float(values["hd95_px"]), "final_boundary_f1": float(values["boundary_f1"]),
                   "final_leakage": float(values["leakage_energy"]), "oracle_regret": oracle - selected_f1,
                   "final_empty": int(not final.any()), "severe_accuracy_failure_proxy_f1_lt_0_5": int(values["f1"] < 0.5),
                   "hard_case_target_identity_preserved": int(selected_f1 >= 0.5 and values["f1"] >= 0.5) if sample == "DouBanLv2" and number in ("0039", "0040", "0041") else "",
                   "spnp_enabled": int(group in ("V4", "V5", "V3_HISTORY")), "spnp_accepted": int(selection_log["SPNP接受"]),
                   "spnp_reason": selection_log["SPNP原因"], "spnp_changed_pixels": changed,
                   "matches_historical_V3_spnp_mask_pixels": historical_spnp_equal,
                   "changed_pixel_ratio_per_image": changed / gt.size,
                   "changed_pixel_ratio_per_selected_mask": changed / max(selected.sum(), 1),
                   "selected_mask_sha256": sha256(output / "选择后掩膜" / f"mask_{stem}.png"),
                   "final_mask_sha256": sha256(output / "最终掩膜" / f"mask_{stem}.png"),
                   "output": str(output), "historical_score_comparability": "SELECTION_EQUIVALENT_MASK_COMPARISON_ONLY" if group == "V3_HISTORY" else "ARCHIVED_PRECISION_REPLAY",
                   "geometry_coordinate_status": "UNVERIFIED" if group == "V3_HISTORY" else "NOT_REQUIRED"}
            (old_rows if group == "V3_HISTORY" else frame_rows).append(row)
            for stage, current in [("selected", selected), ("spnp", spnp), ("residual", mask(output / "残差修复掩膜" / f"mask_{stem}.png")),
                                   ("gcfc", mask(output / "几何修正掩膜" / f"mask_{stem}.png")), ("final", final)]:
                stage_values = {"f1": pixel(current, gt)[0]} if stage not in ("selected", "final") else None
                if stage == "selected":
                    stage_values = {"f1": selected_f1}
                if stage == "final":
                    stage_values = values
                stage_rows.append({"dataset": dataset, "group": group, "sample": sample, "frame": number,
                                   "stage": stage, "f1": float(stage_values["f1"]), "mask_area": int(current.sum())})
        print(f"Evaluated {stem}", flush=True)
    if args.group_shard:
        write_csv(WORK / "metric_shards" / f"{args.group_shard}_frames.csv", frame_rows)
        if old_rows:
            write_csv(WORK / "metric_shards" / "V3_history_frames.csv", old_rows)
        write_csv(WORK / "metric_shards" / f"{args.group_shard}_stages.csv", stage_rows)
        return
    if args.merge_shards:
        for group in groups:
            loaded = read_csv(WORK / "metric_shards" / f"{group}_frames.csv")
            for row in loaded:
                for field in METRICS + ["final_empty", "severe_accuracy_failure_proxy_f1_lt_0_5", "spnp_accepted", "spnp_changed_pixels", "changed_pixel_ratio_per_image"]:
                    row[field] = float(row[field])
            frame_rows.extend(loaded)
            stage_rows.extend(read_csv(WORK / "metric_shards" / f"{group}_stages.csv"))
        old_rows = read_csv(WORK / "metric_shards/V3_history_frames.csv")
        for row in old_rows:
            for field in METRICS + ["final_empty", "severe_accuracy_failure_proxy_f1_lt_0_5", "spnp_accepted", "spnp_changed_pixels", "changed_pixel_ratio_per_image"]:
                row[field] = float(row[field])
    write_csv(WORK / "V4_V5_PER_FRAME.csv", frame_rows)
    write_csv(WORK / "V3_HISTORICAL_COMPARISON.csv", old_rows)
    write_csv(WORK / "STAGE_METRICS.csv", stage_rows)
    write_csv(WORK / "RUN_TIMES.csv", runtime_rows)
    summaries = []
    for dataset in ("Easy21", "GT10"):
        for group in ("V0", "V1", "V4", "V5", "V3_HISTORY"):
            values = [value for value in frame_rows + old_rows if value["dataset"] == dataset and value["group"] == group]
            if not values:
                summaries.append({"dataset": dataset, "group": group, "status": "GEOMETRY_NOT_AVAILABLE", "frames": 0})
                continue
            for aggregation in ("sample_equal_mean", "frame_equal_mean"):
                summary = {"dataset": dataset, "group": group, "status": "DESCRIPTIVE_ONLY_COORDINATES_UNVERIFIED" if group == "V3_HISTORY" else "COMPLETE",
                           "aggregation": aggregation, "frames": len(values), "samples": len({value["sample"] for value in values})}
                for field in METRICS + ["changed_pixel_ratio_per_image"]:
                    summary[field] = sample_equal(values, field) if aggregation == "sample_equal_mean" else float(np.mean([value[field] for value in values]))
                summary.update({"empty_count": sum(value["final_empty"] for value in values),
                                "severe_accuracy_failure_proxy_count": sum(value["severe_accuracy_failure_proxy_f1_lt_0_5"] for value in values),
                                "spnp_accepted_count": sum(value["spnp_accepted"] for value in values),
                                "spnp_acceptance_rate": sum(value["spnp_accepted"] for value in values) / len(values),
                                "spnp_changed_frames": sum(value["spnp_changed_pixels"] > 0 for value in values)})
                summaries.append(summary)
    write_csv(WORK / "V4_V5_METRICS.csv", summaries)
    paired_rows, paired_summary = [], []
    indexed = {(value["dataset"], value["group"], value["sample"], value["frame"]): value for value in frame_rows + old_rows}
    for dataset in ("Easy21", "GT10"):
        for effect, treatment, baseline in [("R1", "V1", "V0"), ("SPNP_R1", "V4", "V1"), ("SPNP_R0", "V5", "V0"),
                                            ("R1_WITH_SPNP", "V4", "V5"), ("FULL_CHAIN_MINUS_SPNP_DESCRIPTIVE", "V3_HISTORY", "V4")]:
            differences = []
            for frame in [value for value in manifest() if value["dataset"] == dataset]:
                key = (dataset, treatment, frame["sample_id"], frame["frame"])
                if key not in indexed:
                    continue
                treated = indexed[key]
                control = indexed[(dataset, baseline, frame["sample_id"], frame["frame"])]
                row = {"dataset": dataset, "effect": effect, "sample": frame["sample_id"], "frame": frame["frame"],
                       **{field: treated[field] - control[field] for field in METRICS}}
                paired_rows.append(row)
                differences.append(row)
            if differences:
                for aggregation in ("frame_level_paired_difference", "sample_equal_paired_difference"):
                    row = {"dataset": dataset, "effect": effect, "aggregation": aggregation, "frames": len(differences)}
                    for field in METRICS:
                        row[field] = float(np.mean([value[field] for value in differences])) if aggregation == "frame_level_paired_difference" else sample_equal(differences, field)
                    row.update({"positive_f1_frames": sum(value["final_f1"] > 1e-12 for value in differences),
                                "negative_f1_frames": sum(value["final_f1"] < -1e-12 for value in differences),
                                "unchanged_f1_frames": sum(abs(value["final_f1"]) <= 1e-12 for value in differences)})
                    paired_summary.append(row)
    write_csv(WORK / "FRAME_PAIRED_DIFFERENCES.csv", paired_rows)
    write_csv(WORK / "PAIRED_METRICS.csv", paired_summary)
    failures = []
    for dataset in ("Easy21", "GT10"):
        base = next(value for value in summaries if value["dataset"] == dataset and value["group"] == "V1" and value.get("aggregation") == "sample_equal_mean")
        refined = next(value for value in summaries if value["dataset"] == dataset and value["group"] == "V4" and value.get("aggregation") == "sample_equal_mean")
        for field, direction in [("final_f1", 1), ("final_iou", 1), ("final_boundary_f1", 1), ("final_hd95", -1), ("final_leakage", -1)]:
            if direction * (refined[field] - base[field]) < -1e-12:
                failures.append(f"{dataset}:{field}")
        if refined["empty_count"] > base["empty_count"] or refined["severe_accuracy_failure_proxy_count"] > base["severe_accuracy_failure_proxy_count"]:
            failures.append(f"{dataset}:empty_or_severe")
    write_json(WORK / "DECISION_CHECK.json", {"predeclared_rule": "IMPLEMENTATION_PLAN.md", "V4_failed_conditions": failures,
               "recommended_method": "V1" if failures else "V4", "independent_accuracy_evaluated": False,
               "all_runs_complete": True, "V3_is_not_score_equivalent_or_coordinate_verified": True})


if __name__ == "__main__":
    main()
