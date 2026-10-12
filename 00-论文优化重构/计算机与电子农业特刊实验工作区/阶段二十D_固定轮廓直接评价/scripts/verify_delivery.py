from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from direct_common import DIFFICULTIES, METRICS, OLD, PROD, ROOT, SWITCHES, WORK, frames, local_path, metric_function, pixel_hash, protocol_module, read_csv, read_mask, relative, sha256, write_json
from evaluate_test32 import overlap, pair_rows, summarize, wide_rows
from rasterize_contours import rasterize
from run_frozen import frozen_parameters

REPORTS = ["PHASE20D_DIRECT_EVALUATION_REPORT.md", "CONTOUR_RASTERIZATION_SPEC.md", "RASTERIZED_MASK_MANIFEST.csv",
           "GEOMETRY_SENSITIVITY_REPORT.md", "R0_R1_TEST32_RESULTS.csv", "R0_R1_TEST32_SUMMARY.csv",
           "R0_R1_SWITCH_CASES.md", "DIFFICULTY_RESULTS.csv", "FOREGROUND_METHOD_DECISION.md", "PHASE20E_EXECUTION_READINESS.md"]


def assert_close(first, second, message):
    if not np.isclose(float(first), float(second), rtol=0, atol=1e-12):
        raise ValueError(f"{message}: {first} != {second}")


def verify():
    protocol = protocol_module()
    protocol.verify_original_sources(OLD, frames())
    for report in REPORTS:
        if not (WORK / report).is_file() or (WORK / report).stat().st_size == 0:
            raise ValueError(f"Missing required deliverable: {report}")
    sources = read_csv(WORK / "SOURCE_PROTECTION_MANIFEST.csv")
    for row in sources:
        path = local_path(row["path"])
        if not path.exists() or sha256(path) != row["sha256"] or relative(path.resolve()) != row["resolved_path"]:
            raise ValueError(f"Protected source changed: {path}")
    spec = sha256(WORK / "CONTOUR_RASTERIZATION_SPEC.md")
    geometry = read_csv(WORK / "CONTOUR_GEOMETRY_AUDIT.csv")
    affected = [row for row in geometry if row["geometry_warning"] == "GEOMETRY_AMBIGUITY"]
    if len(geometry) != 75 or len(affected) != 17 or len({row["frame_id"] for row in affected}) != 13:
        raise ValueError("Original geometry audit coverage mismatch")
    raster_manifest = read_csv(WORK / "RASTERIZED_MASK_MANIFEST.csv")
    if len(raster_manifest) != 64:
        raise ValueError("Missing primary/secondary masks")
    raster_index = {(row["frame_id"], row["rasterizer"]): row for row in raster_manifest}
    if len(raster_index) != 64:
        raise ValueError("Duplicate raster manifest identity")
    for row in raster_manifest:
        annotation_path, rgb_path, mask_path, mapping_path = [local_path(row[key]) for key in ("source_json", "rgb_path", "mask_path", "mapping_path")]
        for path, hash_field in ((annotation_path, "source_json_sha256"), (rgb_path, "rgb_sha256"),
                                 (mask_path, "mask_sha256"), (mapping_path, "mapping_sha256")):
            if sha256(path) != row[hash_field]:
                raise ValueError(f"Raster mapping/hash mismatch: {path}")
        annotation = json.loads(annotation_path.read_text())
        pixels = read_mask(mask_path)
        reproduced, vertices, counts = rasterize(annotation, row["rasterizer"])
        if not np.array_equal(pixels, reproduced) or not pixels.any() or int(pixels.sum()) != int(row["filled_area_px"]):
            raise ValueError("Deterministic raster reproduction failed")
        if pixel_hash(pixels.astype(np.uint8) * 255) != row["mask_pixels_sha256"]:
            raise ValueError("Raster decoded-pixel hash mismatch")
        mapping = json.loads(mapping_path.read_text())
        if mapping["all_source_shapes"] != annotation["shapes"] or mapping["evaluation_shapes"] != vertices:
            raise ValueError("Original vertex/type/label mapping incomplete")
    configurations = {}
    for policy in ("base", "r1"):
        status = json.loads((WORK / "logs" / f"full_{policy}_status.json").read_text())
        if not status["verified"] or status["returncode"]:
            raise ValueError("Production full run not verified")
        execution = json.loads((WORK / "logs" / f"full_{policy}_execution.json").read_text())
        for key, source in (("production_sha256", PROD / "生成RAP-FSAM3掩膜.py"), ("ranking_code_sha256", PROD / "ranking_policy.py"),
                            ("metadata_loader_sha256", PROD / "candidate_metadata.py")):
            if execution[key] != sha256(source):
                raise ValueError("Frozen production code changed")
        if execution["spec_sha256"] != spec:
            raise ValueError("Predeclared rasterization specification changed after execution")
        for row in read_csv(WORK / "logs" / f"full_{policy}_artifacts.csv"):
            if sha256(local_path(row["path"])) != row["sha256"]:
                raise ValueError("Production artifact changed")
        config = json.loads((WORK / "logs" / f"full_{policy}_runtime_parameters.json").read_text())
        expected = frozen_parameters(policy, Path(config["input_dir"]), Path(config["output_dir"]))
        if config != expected:
            raise ValueError("Non-policy runtime parameter changed")
        configurations[policy] = config
    differences = {key for key in configurations["base"] if configurations["base"][key] != configurations["r1"][key]}
    if differences != {"ranking_policy", "output_dir"}:
        raise ValueError("R0/R1 configs differ beyond policy/output path")
    for stem in SWITCHES:
        for policy in ("base", "r1"):
            status = json.loads((WORK / "logs" / f"smoke_{policy}_{stem}_status.json").read_text())
            if not status["verified"]:
                raise ValueError("Required smoke not passed")
    evaluate = metric_function()
    rows = read_csv(WORK / "R0_R1_ALL_RASTER_RESULTS.csv")
    if len(rows) != 128:
        raise ValueError("32×2×2metric coverage incomplete")
    candidate_scores = read_csv(WORK / "ORACLE_ALL_CANDIDATES.csv")
    if len(candidate_scores) != 142:
        raise ValueError("71×2oracle candidate coverage incomplete")
    numeric_rows = []
    cache = {}
    manual = {row["frame_id"]: row for row in read_csv(OLD / "DIFFICULTY_HUMAN_VERIFICATION.csv")}
    for row in rows:
        ground = read_mask(local_path(raster_index[(row["frame_id"], row["rasterizer"])]["mask_path"]))
        prediction = read_mask(local_path(row["output_mask_path"]))
        if sha256(local_path(row["output_mask_path"])) != row["output_mask_sha256"]:
            raise ValueError("Numeric table output mask hash mismatch")
        key = (row["gt_sha256"], row["output_mask_sha256"])
        if key not in cache:
            cache[key] = evaluate(ground, prediction)
        for metric, value in cache[key].items():
            assert_close(row[metric], value, "Full original metric recomputation")
        best = max([entry for entry in candidate_scores if entry["frame_id"] == row["frame_id"] and entry["rasterizer"] == row["rasterizer"]], key=lambda entry: float(entry["f1"]))
        selected = next(entry for entry in candidate_scores if entry["frame_id"] == row["frame_id"] and entry["rasterizer"] == row["rasterizer"] and entry["candidate_id"] == row["selected_candidate_id"])
        if best["candidate_id"] != row["oracle_candidate_id"]:
            raise ValueError("Oracle ID mismatch")
        assert_close(row["oracle_regret"], float(best["f1"]) - float(selected["f1"]), "Oracle regret definition")
        labels = ";".join(code for code in DIFFICULTIES if manual[row["frame_id"]][code] == "YES") or "none"
        if row["difficulty_labels"] != labels:
            raise ValueError("Difficulty record changed/reclassified")
        numeric_rows.append({**row, **{key: float(row[key]) for key in METRICS + ["tp", "fp", "fn", "tn", "precision", "recall", "raw_selected_f1", "raw_selected_iou_fg", "oracle_f1"]},
                             "empty_mask": int(row["empty_mask"]), "severe_f1_lt_0_5": int(row["severe_f1_lt_0_5"])})
    for rasterizer, path in (("opencv", WORK / "R0_R1_TEST32_RESULTS.csv"), ("pillow", WORK / "R0_R1_TEST32_SECONDARY_RESULTS.csv")):
        calculated = wide_rows([row for row in numeric_rows if row["rasterizer"] == rasterizer])
        saved = {row["frame_id"]: row for row in read_csv(path)}
        if len(saved) != 32:
            raise ValueError("Primary32frame table incomplete")
        for row in calculated:
            for key, value in row.items():
                if isinstance(value, (float, int)):
                    assert_close(saved[row["frame_id"]][key], value, "Wide paired table reconstruction")
                elif str(value) != saved[row["frame_id"]][key]:
                    raise ValueError("Wide table identity mismatch")
    summary = read_csv(WORK / "R0_R1_TEST32_SUMMARY.csv")
    if len(summary) != 108:
        raise ValueError("Full/subset summary coverage mismatch")
    for rasterizer in ("opencv", "pillow"):
        raster_rows = [row for row in numeric_rows if row["rasterizer"] == rasterizer]
        differences = pair_rows(raster_rows)
        for subset in ("ALL32", "GEOMETRY_AMBIGUITY13", "OTHER19"):
            selected = raster_rows + differences
            if subset != "ALL32":
                selected = [row for row in selected if (row["geometry_warning"] == "GEOMETRY_AMBIGUITY") == (subset == "GEOMETRY_AMBIGUITY13")]
            for method in ("R0", "R1", "R1_MINUS_R0"):
                calculated = summarize([row for row in selected if row["method"] == method], rasterizer, subset, method)
                for expected in calculated:
                    matches = [row for row in summary if all(row[key] == expected[key] for key in ("rasterizer", "subset", "method", "aggregation"))]
                    if len(matches) != 1:
                        raise ValueError("Summary coverage mismatch")
                    for key in METRICS + ["frames", "sequences", "empty_mask_count", "severe_f1_lt_0_5_count"]:
                        assert_close(matches[0][key], expected[key], "All summary rows recomputed")
    sequence_rows = read_csv(WORK / "R0_R1_TEST32_PER_SEQUENCE.csv")
    if len(sequence_rows) != 288:
        raise ValueError("Per-sequence summary coverage mismatch")
    for rasterizer in ("opencv", "pillow"):
        raster_rows = [row for row in numeric_rows if row["rasterizer"] == rasterizer]
        differences = pair_rows(raster_rows)
        for sample in sorted({row["sample"] for row in raster_rows}):
            for method in ("R0", "R1", "R1_MINUS_R0"):
                selected = [row for row in raster_rows + differences if row["sample"] == sample and row["method"] == method]
                for expected in summarize(selected, rasterizer, sample, method):
                    matches = [row for row in sequence_rows if row["sample"] == sample and all(row[key] == expected[key] for key in ("rasterizer", "method", "aggregation"))]
                    if len(matches) != 1:
                        raise ValueError("Per-sequence summary identity mismatch")
                    for key in METRICS + ["frames", "sequences", "empty_mask_count", "severe_f1_lt_0_5_count"]:
                        assert_close(matches[0][key], expected[key], "Per-sequence summaries recomputed")
    figures = read_csv(WORK / "figures/FIGURE_SOURCE_INDEX.csv")
    if len(figures) != 32 or {row["frame_id"] for row in figures} != {frame["frame_id"] for frame in frames()}:
        raise ValueError("Complete real32figure coverage missing")
    for row in figures:
        if sha256(local_path(row["figure_path"])) != row["figure_sha256"]:
            raise ValueError("Real figure hash changed")
    pilot = read_csv(WORK / "PHASE20E_VIEW_MANIFEST.csv")
    if len(pilot) != 532 or any(row["frozen_split"] != ("validation" if int(row["split_order_index"]) % 8 == 0 else "train") for row in pilot):
        raise ValueError("Phase20E full-view fixed split not reproducible")
    code = read_csv(WORK / "CODE_SHA256_MANIFEST.csv")
    for row in code:
        if sha256(local_path(row["path"])) != row["sha256"]:
            raise ValueError("Delivered code hash mismatch")
    return {"status": "PASS", "protected_sources": len(sources), "raster_masks_reproduced": len(raster_manifest),
            "full_metric_rows_recomputed": len(rows), "summary_rows_verified": len(summary), "sequence_rows_verified": len(sequence_rows), "oracle_candidates": len(candidate_scores),
            "real_frame_figures": len(figures), "phase20e_views_audited": len(pilot), "source_gate_unchanged": True,
            "original_json_rgb_candidates_unchanged": True, "dual_signatures_not_invented": True,
            "phase20e_training_started": False, "spec_sha256": spec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=WORK / "logs/DELIVERY_VERIFICATION.json")
    args = parser.parse_args()
    result = verify()
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
