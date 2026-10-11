from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fix_common import OLD, WORK, load_module, read_csv, sha256, write_csv


REQUIRED_LABELS = {"potted_plant", "plant", "pot"}


def annotation_masks(annotation, width, height):
    if annotation.get("imageWidth") != width or annotation.get("imageHeight") != height:
        raise ValueError("Annotation/RGB coordinate mismatch; resampling forbidden")
    masks = {label: np.zeros((height, width), dtype=np.uint8) for label in REQUIRED_LABELS | {"blue_cube"}}
    present = set()
    for shape in annotation["shapes"]:
        label = shape["label"]
        if label not in masks:
            raise ValueError(f"Unapproved label {label}")
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError("Only original-coordinate human polygons accepted")
        points = np.asarray(shape["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Invalid polygon")
        if (points[:, 0] < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] < 0).any() or (points[:, 1] >= height).any():
            raise ValueError("Out-of-frame polygon; manual correction required")
        cv2.fillPoly(masks[label], [np.rint(points).astype(np.int32)], 1)
        present.add(label)
    if not REQUIRED_LABELS <= present or any(not masks[label].any() for label in REQUIRED_LABELS):
        raise ValueError("Human target potted_plant/plant/pot polygons required")
    target = masks["potted_plant"].astype(bool) & ~masks["blue_cube"].astype(bool)
    return target, masks


def validate_review(row):
    required = ["target_identity_confirmed", "plant_confirmed", "pot_confirmed", "difficulty_manually_confirmed"]
    if row["approval"] != "APPROVED" or not row["reviewer"] or not row["second_reviewer"]:
        raise ValueError("Independent test blocked: two-person human approval missing")
    if row["reviewer"] == row["second_reviewer"] or any(row[field] != "YES" for field in required):
        raise ValueError("Independent test blocked: confirmation/second-review separation missing")
    if row["blue_cube_present"] not in ("YES", "NO") or not row["human_difficulty_labels"]:
        raise ValueError("Independent test blocked: cube/difficulty human decisions missing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, default=WORK / "annotation_package/ANNOTATION_REVIEW_QUEUE.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-prediction-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite independent evaluation output")
    frozen = read_csv(WORK / "TEST32_INPUT_AUDIT.csv")
    reviews = read_csv(args.review_queue)
    indexed = {row["challenge_id"]: row for row in reviews}
    if len(indexed) != 32 or set(indexed) != {row["challenge_id"] for row in frozen}:
        raise ValueError("Frozen TEST32 review coverage mismatch; no cherry-picking")
    for row in indexed.values():
        validate_review(row)
    metric_module = load_module("phase20d_fixed_metric_definition", OLD / "scripts/compute_phase20c_metrics.py")
    evaluate = metric_module.load_segmentation_metrics()
    results, paired = [], []
    for frame in frozen:
        row = indexed[frame["challenge_id"]]
        if row["rgb_sha256"] != frame["rgb_sha256"]:
            raise ValueError("Frozen review RGB identity mismatch")
        annotation_path = WORK / row["annotation_json"]
        annotation = json.loads(annotation_path.read_text())
        gt, label_masks = annotation_masks(annotation, int(frame["width"]), int(frame["height"]))
        if bool(label_masks["blue_cube"].any()) != (row["blue_cube_present"] == "YES"):
            raise ValueError("Blue cube presence disagrees with human review")
        path = args.prediction_dir / f"mask_{frame['sample']}_{frame['frame']}.png"
        with Image.open(path) as image:
            pred = np.asarray(image.convert("L")) > 0
        if pred.shape != gt.shape:
            raise ValueError("Prediction/RGB coordinate mismatch; resampling forbidden")
        values = evaluate(gt, pred)
        results.append({"challenge_id": frame["challenge_id"], "sample": frame["sample"], "frame": frame["frame"],
                        "annotation_sha256": sha256(annotation_path), "prediction_sha256": sha256(path),
                        "difficulty_human": row["human_difficulty_labels"], **values})
        if args.reference_prediction_dir is not None:
            reference_path = args.reference_prediction_dir / path.name
            with Image.open(reference_path) as image:
                reference = np.asarray(image.convert("L")) > 0
            if reference.shape != gt.shape:
                raise ValueError("Reference prediction coordinate mismatch")
            reference_values = evaluate(gt, reference)
            paired.append({"challenge_id": frame["challenge_id"], "sample": frame["sample"], "frame": frame["frame"],
                           **{metric: values[metric] - reference_values[metric] for metric in ["f1", "iou_fg", "hd95_px", "boundary_f1", "leakage_energy"]}})
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    paired_path = args.output.with_name(args.output.stem + "_paired.csv")
    if summary_path.exists() or paired_path.exists():
        raise FileExistsError("Refusing existing independent summary/paired output")
    write_csv(args.output, results)
    summaries = []
    for name, rows in [("absolute", results), ("paired", paired)]:
        if not rows:
            continue
        for aggregation in ["sample_equal_mean", "frame_level_mean" if name == "absolute" else "frame_level_paired_difference"]:
            summary = {"comparison": name, "aggregation": aggregation, "frames": len(rows),
                       "samples": len({row["sample"] for row in rows})}
            for metric in ["f1", "iou_fg", "hd95_px", "boundary_f1", "leakage_energy"]:
                by_sample = defaultdict(list)
                for row in rows:
                    by_sample[row["sample"]].append(row[metric])
                summary[metric] = float(np.mean([np.mean(values) for values in by_sample.values()])) if aggregation == "sample_equal_mean" else float(np.mean([row[metric] for row in rows]))
            summaries.append(summary)
    write_csv(summary_path, summaries)
    if paired:
        write_csv(paired_path, paired)


if __name__ == "__main__":
    main()
