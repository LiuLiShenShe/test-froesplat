from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from fix_common import ALG, OLD, ROOT, WORK, manifest, production, read_csv, sha256, write_csv, write_json


def overlap_metrics(mask, gt):
    true_positive = np.logical_and(mask, gt).sum()
    denominator = mask.sum() + gt.sum()
    union = np.logical_or(mask, gt).sum()
    return float(2 * true_positive / denominator) if denominator else 1.0, float(true_positive / union) if union else 1.0


def main():
    module = production()
    benchmark = read_csv(ALG / "阶段十七C_候选排序修复开发/Phase17C_ranking_benchmark.csv")
    benchmark_index = {(row["sample"], row["frame"], int(row["instance_id"])): row for row in benchmark}
    provenance = read_csv(WORK / "CANDIDATE_METADATA_PROVENANCE.csv")
    provenance_index = {row["candidate_id"]: row for row in provenance}
    rows, selections = [], []
    maximum_error = 0.0
    for frame in manifest():
        dataset, stem = frame["dataset"], frame["stem"]
        source = ROOT / frame["candidate_source_root"]
        parameters = json.loads((source / "参数.json").read_text())
        args = argparse.Namespace(**parameters)
        args.ranking_policy = "base"
        weights = module.parse_key_values(args.score_weights, {})
        rgb = WORK / "inputs" / dataset / f"{stem}.jpg"
        image = Image.open(rgb).convert("RGB")
        with Image.open(ROOT / frame["gt_path"]) as handle:
            gt = np.asarray(handle.convert("L")) > 0
        candidates = module.load_existing_candidates(rgb, ["P6"], module.BUILTIN_PROMPTS, WORK / "candidates" / dataset,
                                                    candidate_metadata_dir=WORK / "metadata" / dataset, candidate_metadata_strict=True)
        candidate_rows = []
        for candidate in candidates:
            candidate_id = f"{stem}_{candidate.instance_id:02d}"
            archived = provenance_index[candidate_id]
            record = module.score_candidate(rgb, image, candidate, {}, weights, args)
            q_sam = 1 / (1 + math.exp(-6 * (candidate.sam_score - 0.5)))
            score_error = record.total_score - float(archived["historical_total"])
            maximum_error = max(maximum_error, abs(score_error))
            benchmark_row = benchmark_index[(frame["sample_id"], frame["frame"], candidate.instance_id)]
            if abs(record.q_contrast - float(archived["historical_q_contrast"])) > 1e-10:
                raise ValueError(f"Frozen non-SAM term mismatch: {candidate_id}")
            f1, iou = overlap_metrics(candidate.mask, gt)
            args.ranking_policy = "r1"
            r1_record = module.score_candidate(rgb, image, candidate, {}, weights, args)
            args.ranking_policy = "base"
            scoreless = copy.copy(candidate)
            scoreless.sam_score = 0.0
            zero_base = module.score_candidate(rgb, image, scoreless, {}, weights, args)
            args.ranking_policy = "r1"
            zero_r1 = module.score_candidate(rgb, image, scoreless, {}, weights, args)
            args.ranking_policy = "base"
            row = {"dataset": dataset, "sample": frame["sample_id"], "frame": frame["frame"], "candidate_id": candidate_id,
                   "prompt_id": "P6", "instance_id": candidate.instance_id, "sam_score": candidate.sam_score,
                   "q_sam": q_sam, "risk_penalty": record.leak_penalty, "box": json.dumps(candidate.box),
                   "mask_threshold": candidate.mask_threshold, "candidate_order": candidate.candidate_order,
                   "rgb_sha256": candidate.source_image_sha256, "mask_sha256": candidate.mask_sha256,
                   "historical_R0_primary": float(archived["historical_total"]),
                   "historical_offline_R1_primary": float(benchmark_row["total_score"]) - float(benchmark_row["q_contrast"]) / sum(weights.values()),
                   "phase20c_zero_score_R0_primary": zero_base.primary_score,
                   "phase20c_zero_score_R1_primary": zero_r1.primary_score,
                   "restored_R0_primary": record.primary_score, "restored_R1_primary": r1_record.primary_score,
                   "candidate_f1": f1, "candidate_iou": iou, "historical_total_difference": score_error,
                   "score_precision": "ARCHIVED_6_DECIMALS", "replay_tolerance": 1e-7,
                   "replay_status": "SCORE_EQUIVALENT" if abs(score_error) <= 1e-7 else "METADATA_UNVERIFIED"}
            rows.append(row)
            candidate_rows.append(row)
        oracle = max(value["candidate_f1"] for value in candidate_rows)
        policies = [("historical_R0", "historical_R0_primary"), ("phase17c_offline_R1", "historical_offline_R1_primary"),
                    ("phase20c_zero_R0", "phase20c_zero_score_R0_primary"), ("phase20c_zero_R1", "phase20c_zero_score_R1_primary"),
                    ("restored_R0", "restored_R0_primary"), ("restored_R1", "restored_R1_primary")]
        chosen = {}
        for run, field in policies:
            chosen[run] = max(candidate_rows, key=lambda value: value[field])
        for run, field in policies:
            selected = chosen[run]
            restored_policy = run == "restored_R1" or run == "restored_R0"
            baseline = chosen["phase20c_zero_R1" if run == "restored_R1" else "phase20c_zero_R0"]
            selections.append({"dataset": dataset, "sample": frame["sample_id"], "frame": frame["frame"], "run": run,
                               "selected_candidate_id": selected["candidate_id"], "primary_score": selected[field],
                               "risk_penalty": selected["risk_penalty"], "q_sam": selected["q_sam"] if "zero" not in run else 1 / (1 + math.exp(3)),
                               "f1": selected["candidate_f1"], "iou": selected["candidate_iou"], "oracle_regret": oracle - selected["candidate_f1"],
                               "phase20c_selection_changed": int(restored_policy and selected["candidate_id"] != baseline["candidate_id"]),
                               "reason": "RESTORED_CONFIDENCE_CHANGED_SELECTION" if restored_policy and selected["candidate_id"] != baseline["candidate_id"] else
                                         "SAME_SELECTION_CHANGED_SCORE" if restored_policy else "FROZEN_REFERENCE",
                               "replay_status": selected["replay_status"] if restored_policy else "REFERENCE_ONLY"})
        print(stem, chosen["restored_R0"]["candidate_id"], chosen["restored_R1"]["candidate_id"], flush=True)
    write_csv(WORK / "CANDIDATE_SCORE_REPLAY.csv", rows)
    write_csv(WORK / "R0_R1_SELECTION_REPLAY.csv", selections)
    write_json(WORK / "SCORE_REPLAY_AUDIT.json", {"candidates": len(rows), "frames": 31, "maximum_historical_total_error": maximum_error,
               "tolerance": 1e-7, "precision": "ARCHIVED_6_DECIMALS; NOT BYTE_IDENTICAL_UNROUNDED_TENSOR",
               "score_equivalent_candidates": sum(row["replay_status"] == "SCORE_EQUIVALENT" for row in rows),
               "restored_selection_changes": sum(row["phase20c_selection_changed"] for row in selections),
               "ranking_policy_sha256": sha256(Path(module.__file__).parent / "ranking_policy.py")})
    if maximum_error > 1e-7:
        raise ValueError("Historical scoring replay exceeds the predeclared archived-precision tolerance")


if __name__ == "__main__":
    main()
