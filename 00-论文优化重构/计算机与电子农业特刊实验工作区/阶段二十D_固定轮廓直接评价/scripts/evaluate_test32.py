from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from direct_common import DIFFICULTIES, METRICS, OLD, WORK, frames, local_path, metric_function, read_csv, read_mask, relative, sha256, write_csv, write_json


def direction(value, tolerance=1e-12):
    return "IMPROVED" if value > tolerance else "REGRESSED" if value < -tolerance else "UNCHANGED"


def overlap(ground, prediction):
    if ground.shape != prediction.shape:
        raise ValueError("Oracle grid mismatch")
    intersection = int((ground & prediction).sum())
    denominator = int(ground.sum()) + int(prediction.sum())
    union = int((ground | prediction).sum())
    return 2 * intersection / denominator if denominator else 0.0, intersection / union if union else 0.0


def grouped_values(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample"]].append(float(row[metric]))
    return np.asarray([np.mean(values) for values in groups.values()], dtype=np.float64)


def summarize(rows, rasterizer, subset, method):
    reductions = [("sample_equal_mean", np.mean, True), ("frame_equal_mean", np.mean, False),
                  ("frame_median", np.median, False), ("frame_sd_population_ddof0", np.std, False),
                  ("sequence_mean_median", np.median, True), ("sequence_mean_sd_population_ddof0", np.std, True)]
    result = []
    for aggregation, reduction, per_sample in reductions:
        row = {"rasterizer": rasterizer, "subset": subset, "method": method, "aggregation": aggregation,
               "frames": len(rows), "sequences": len({value["sample"] for value in rows}),
               "empty_mask_count": sum(value["empty_mask"] for value in rows),
               "severe_f1_lt_0_5_count": sum(value["severe_f1_lt_0_5"] for value in rows)}
        for metric in METRICS:
            values = grouped_values(rows, metric) if per_sample else np.asarray([value[metric] for value in rows], dtype=np.float64)
            row[metric] = float(reduction(values))
        if method == "R1_MINUS_R0":
            row.update(improved_f1_frames=sum(value["f1"] > 1e-12 for value in rows),
                       regressed_f1_frames=sum(value["f1"] < -1e-12 for value in rows),
                       unchanged_f1_frames=sum(abs(value["f1"]) <= 1e-12 for value in rows))
        result.append(row)
    return result


def pair_rows(rows):
    indexed = {(row["frame_id"], row["method"]): row for row in rows}
    result = []
    for stem in sorted({row["frame_id"] for row in rows}):
        baseline, treatment = indexed[(stem, "R0")], indexed[(stem, "R1")]
        result.append({"frame_id": stem, "sample": baseline["sample"], "geometry_warning": baseline["geometry_warning"],
                       "method": "R1_MINUS_R0", "rasterizer": baseline["rasterizer"],
                       **{metric: treatment[metric] - baseline[metric] for metric in METRICS},
                       "empty_mask": treatment["empty_mask"] - baseline["empty_mask"],
                       "severe_f1_lt_0_5": treatment["severe_f1_lt_0_5"] - baseline["severe_f1_lt_0_5"]})
    return result


def wide_rows(rows):
    indexed = {(row["frame_id"], row["method"]): row for row in rows}
    result = []
    for stem in sorted({row["frame_id"] for row in rows}):
        baseline, treatment = indexed[(stem, "R0")], indexed[(stem, "R1")]
        row = {"frame_id": stem, "sample": baseline["sample"], "rasterizer": baseline["rasterizer"],
               "geometry_warning": baseline["geometry_warning"], "main_gt_shape_ambiguous": baseline["main_gt_shape_ambiguous"],
               "difficulty_labels": baseline["difficulty_labels"], "gt_sha256": baseline["gt_sha256"],
               "r0_selected_id": baseline["selected_candidate_id"], "r1_selected_id": treatment["selected_candidate_id"],
               "selection_changed": int(baseline["selected_candidate_id"] != treatment["selected_candidate_id"]),
               "oracle_candidate_id": baseline["oracle_candidate_id"], "oracle_f1": baseline["oracle_f1"],
               "f1_effect": direction(treatment["f1"] - baseline["f1"])}
        for metric in METRICS + ["empty_mask", "severe_f1_lt_0_5", "tp", "fp", "fn", "tn", "precision", "recall"]:
            row[f"r0_{metric}"] = baseline[metric]
            row[f"r1_{metric}"] = treatment[metric]
            row[f"delta_{metric}"] = treatment[metric] - baseline[metric]
        result.append(row)
    return result


def main():
    for policy in ("base", "r1"):
        status = json.loads((WORK / "logs" / f"full_{policy}_status.json").read_text())
        if not status["verified"]:
            raise ValueError("Both full production runs must be verified")
    evaluate = metric_function()
    manifest = read_csv(WORK / "RASTERIZED_MASK_MANIFEST.csv")
    masks = {(row["frame_id"], row["rasterizer"]): row for row in manifest}
    selections = {(row["frame_id"], policy): row for policy in ("base", "r1")
                  for row in read_csv(WORK / "logs" / f"full_{policy}_selections.csv")}
    candidates = read_csv(OLD / "TEST32_CANDIDATE_PROVENANCE.csv")
    difficulties = {row["frame_id"]: row for row in read_csv(OLD / "DIFFICULTY_HUMAN_VERIFICATION.csv")}
    all_rows, oracle_rows = [], []
    for frame in frames():
        stem = frame["frame_id"]
        saved = difficulties[stem]
        labels = [code for code in DIFFICULTIES if saved[code] == "YES"]
        if any(saved[code] not in ("YES", "NO") for code in DIFFICULTIES):
            raise ValueError("Difficulty grouping must use existing manual cells")
        for rasterizer in ("opencv", "pillow"):
            gt_info = masks[(stem, rasterizer)]
            gt_path = local_path(gt_info["mask_path"])
            if sha256(gt_path) != gt_info["mask_sha256"]:
                raise ValueError("Rasterized GT bytes changed")
            ground = read_mask(gt_path)
            frame_oracles = []
            for candidate in sorted([row for row in candidates if row["frame_id"] == stem], key=lambda row: int(row["candidate_order"])):
                path = local_path(candidate["mask_path"])
                if sha256(path) != candidate["mask_sha256"]:
                    raise ValueError("Original candidate changed during evaluation")
                values = overlap(ground, read_mask(path))
                item = {"frame_id": stem, "rasterizer": rasterizer, "candidate_id": candidate["candidate_id"],
                        "candidate_order": int(candidate["candidate_order"]), "f1": values[0], "iou_fg": values[1],
                        "source_candidate_sha256": candidate["mask_sha256"], "gt_sha256": gt_info["mask_sha256"]}
                frame_oracles.append(item)
                oracle_rows.append(item)
            oracle = max(frame_oracles, key=lambda row: row["f1"])
            evaluation_cache = {}
            for policy, method in (("base", "R0"), ("r1", "R1")):
                choice = selections[(stem, policy)]
                selected_path = local_path(choice["selected_mask_path"])
                final_path = local_path(choice["final_mask_path"])
                if sha256(selected_path) != choice["selected_mask_sha256"] or sha256(final_path) != choice["output_mask_sha256"]:
                    raise ValueError("Production mask bytes changed")
                selected, final = read_mask(selected_path), read_mask(final_path)
                raw_f1, raw_iou = overlap(ground, selected)
                key = choice["selected_mask_pixels_sha256"]
                if key not in evaluation_cache:
                    evaluation_cache[key] = evaluate(ground, final)
                values = evaluation_cache[key]
                regret = oracle["f1"] - raw_f1
                if regret < -1e-12:
                    raise ValueError("Negative oracle regret indicates identity/metric error")
                all_rows.append({"frame_id": stem, "sample": frame["sample"], "method": method, "rasterizer": rasterizer,
                                 "geometry_warning": gt_info["contour_geometry_warning"],
                                 "main_gt_shape_ambiguous": int(bool(json.loads(gt_info["ambiguous_main_gt_shape_indices"]))),
                                 "difficulty_labels": ";".join(labels) or "none", "difficulty_record_status": saved["status"],
                                 "difficulty_is_archived_grouping_not_verified_truth": True,
                                 "selected_candidate_id": choice["selected_candidate_id"], "oracle_candidate_id": oracle["candidate_id"],
                                 "oracle_f1": oracle["f1"], "oracle_regret": regret, "raw_selected_f1": raw_f1,
                                 "raw_selected_iou_fg": raw_iou, "raw_equals_final": bool(np.array_equal(selected, final)),
                                 "empty_mask": int(not final.any()), "severe_f1_lt_0_5": int(values["f1"] < 0.5),
                                 "gt_sha256": gt_info["mask_sha256"], "source_candidate_sha256": choice["source_candidate_sha256"],
                                 "output_mask_sha256": choice["output_mask_sha256"], "output_mask_path": relative(final_path), **values})
        print(f"evaluated {stem}: both rasterizers, both frozen policies", flush=True)
    write_csv(WORK / "R0_R1_ALL_RASTER_RESULTS.csv", all_rows)
    write_csv(WORK / "ORACLE_ALL_CANDIDATES.csv", oracle_rows)
    summaries, sequences, paired, difficulty_rows = [], [], [], []
    for rasterizer in ("opencv", "pillow"):
        rows = [row for row in all_rows if row["rasterizer"] == rasterizer]
        wide = wide_rows(rows)
        write_csv(WORK / ("R0_R1_TEST32_RESULTS.csv" if rasterizer == "opencv" else "R0_R1_TEST32_SECONDARY_RESULTS.csv"), wide)
        differences = pair_rows(rows)
        paired.extend(differences)
        for subset in ("ALL32", "GEOMETRY_AMBIGUITY13", "OTHER19"):
            selected = rows + differences
            if subset != "ALL32":
                ambiguous = subset == "GEOMETRY_AMBIGUITY13"
                selected = [row for row in selected if (row["geometry_warning"] == "GEOMETRY_AMBIGUITY") == ambiguous]
            for method in ("R0", "R1", "R1_MINUS_R0"):
                group = [row for row in selected if row["method"] == method]
                if len(group) != {"ALL32": 32, "GEOMETRY_AMBIGUITY13": 13, "OTHER19": 19}[subset]:
                    raise ValueError("Predeclared geometry subset coverage mismatch")
                summaries.extend(summarize(group, rasterizer, subset, method))
        for sample in sorted({row["sample"] for row in rows}):
            for method in ("R0", "R1", "R1_MINUS_R0"):
                group = [row for row in rows + differences if row["sample"] == sample and row["method"] == method]
                summary = summarize(group, rasterizer, sample, method)
                for value in summary:
                    value["sample"] = sample
                sequences.extend(summary)
        for code in list(DIFFICULTIES) + ["none"]:
            group = [row for row in wide if code in row["difficulty_labels"].split(";")]
            if not group:
                continue
            identifiers = {row["frame_id"] for row in group}
            for aggregation in ("frame_equal_mean", "sample_equal_mean"):
                item = {"difficulty": code, "description": DIFFICULTIES.get(code, "No recorded difficulty"),
                        "rasterizer": rasterizer, "aggregation": aggregation, "frames": len(group),
                        "sequences": len({row["sample"] for row in group}), "overlapping_groups_allowed": True,
                        "record_quality": "Existing manual YES/NO; not independently Gate-approved difficulty truth",
                        "improved_frames": sum(row["delta_f1"] > 1e-12 for row in group),
                        "regressed_frames": sum(row["delta_f1"] < -1e-12 for row in group),
                        "unchanged_frames": sum(abs(row["delta_f1"]) <= 1e-12 for row in group),
                        "geometry_ambiguity_frames": sum(row["geometry_warning"] == "GEOMETRY_AMBIGUITY" for row in group),
                        "frame_ids": ";".join(sorted(identifiers))}
                for method, prefix in (("R0", "r0"), ("R1", "r1"), ("R1_MINUS_R0", "delta")):
                    method_rows = [row for row in rows + differences if row["method"] == method and row["frame_id"] in identifiers]
                    values = next(row for row in summarize(method_rows, rasterizer, code, method) if row["aggregation"] == aggregation)
                    for metric in METRICS:
                        item[f"{prefix}_{metric}"] = values[metric]
                    item[f"{prefix}_empty_count"] = values["empty_mask_count"]
                    item[f"{prefix}_severe_count"] = values["severe_f1_lt_0_5_count"]
                failure_rows = [row for row in rows if row["frame_id"] in identifiers and row["method"] == "R1"]
                item["major_failure_manifestation"] = "Severe contour mismatch(F1<0.5): " + ";".join(
                    f"{row['frame_id']}(FP={row['fp']},FN={row['fn']})" for row in failure_rows if row["severe_f1_lt_0_5"])
                if not any(row["severe_f1_lt_0_5"] for row in failure_rows):
                    item["major_failure_manifestation"] = "No F1<0.5; residual contour FP/FN and boundary mismatch (not a human identity diagnosis)"
                difficulty_rows.append(item)
    write_csv(WORK / "R0_R1_TEST32_SUMMARY.csv", summaries)
    write_csv(WORK / "R0_R1_TEST32_PER_SEQUENCE.csv", sequences)
    write_csv(WORK / "R0_R1_PAIRED_DIFFERENCES.csv", paired)
    write_csv(WORK / "DIFFICULTY_RESULTS.csv", difficulty_rows)
    stability_rows = []
    primary = {(row["frame_id"], row["method"]): row for row in all_rows if row["rasterizer"] == "opencv"}
    secondary = {(row["frame_id"], row["method"]): row for row in all_rows if row["rasterizer"] == "pillow"}
    for frame in frames():
        stem = frame["frame_id"]
        item = {"frame_id": stem, "geometry_warning": primary[(stem, "R0")]["geometry_warning"]}
        for metric in METRICS:
            for method in ("R0", "R1"):
                item[f"{method.lower()}_pillow_minus_opencv_{metric}"] = secondary[(stem, method)][metric] - primary[(stem, method)][metric]
            diff0 = primary[(stem, "R1")][metric] - primary[(stem, "R0")][metric]
            diff1 = secondary[(stem, "R1")][metric] - secondary[(stem, "R0")][metric]
            item[f"opencv_paired_{metric}"] = diff0
            item[f"pillow_paired_{metric}"] = diff1
            item[f"paired_direction_stable_{metric}"] = direction(diff0) == direction(diff1)
        stability_rows.append(item)
    write_csv(WORK / "METRIC_RASTERIZATION_STABILITY.csv", stability_rows)
    ranking_sensitive = any(not row[f"paired_direction_stable_{metric}"] for row in stability_rows for metric in ("f1", "iou_fg"))
    summary_map = {(row["rasterizer"], row["subset"], row["aggregation"]): row for row in summaries if row["method"] == "R1_MINUS_R0"}
    for subset in ("ALL32", "GEOMETRY_AMBIGUITY13", "OTHER19"):
        for aggregation in ("sample_equal_mean", "frame_equal_mean"):
            for metric in ("f1", "iou_fg"):
                if direction(summary_map[("opencv", subset, aggregation)][metric]) != direction(summary_map[("pillow", subset, aggregation)][metric]):
                    ranking_sensitive = True
    write_json(WORK / "logs/NUMERIC_EVALUATION_STATUS.json", {
        "frames_per_rasterizer": 32, "metric_rows": len(all_rows), "candidate_oracle_rows": len(oracle_rows),
        "geometry_frames": 13, "other_frames": 19, "rasterization_sensitivity": "SENSITIVE" if ranking_sensitive else "STABLE",
        "score_formula_or_weights_changed": False, "development_GT10_in_TEST32": False,
        "source_difficulty_sha256": sha256(OLD / "DIFFICULTY_HUMAN_VERIFICATION.csv"),
        "spec_sha256": sha256(WORK / "CONTOUR_RASTERIZATION_SPEC.md")})


if __name__ == "__main__":
    main()
