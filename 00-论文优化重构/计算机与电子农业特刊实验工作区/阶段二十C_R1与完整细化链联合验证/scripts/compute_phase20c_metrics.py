#!/usr/bin/env python3
"""Compute frame, stage, selection, module, and paired Phase 20C metrics."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证"
B20_SCRIPTS = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十B_历史实验与证据修复/scripts"
P17C = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七C_候选排序修复开发"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def load_segmentation_metrics():
    sys.path.insert(0, str(B20_SCRIPTS))
    spec = importlib.util.spec_from_file_location("phase20b_segmentation", B20_SCRIPTS / "segmentation.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Phase 20B metric implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.evaluate_arrays


def safe_float(value: object) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_instance(path: Path, stem: str) -> int | None:
    prefix = f"mask_{stem}_"
    if not path.stem.startswith(prefix):
        return None
    try:
        return int(path.stem[len(prefix) :])
    except ValueError:
        return None


def stage_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "selected": output / f"选择后掩膜/mask_{stem}.png",
        "spnp": output / f"正负提示细化掩膜/mask_{stem}.png",
        "residual": output / f"残差修复掩膜/mask_{stem}.png",
        "gcfc": output / f"几何修正掩膜/mask_{stem}.png",
        "final": output / f"最终掩膜/mask_{stem}.png",
    }


def metric_values(evaluate, gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    values = evaluate(gt, pred)
    return {
        "f1": float(values["f1"]),
        "iou_fg": float(values["iou_fg"]),
        "miou": float(values["miou"]),
        "hd95": float(values["hd95_px"]),
        "boundary_f1": float(values["boundary_f1"]),
        "leakage": float(values["leakage_energy"]),
        "tp": int(values["tp"]),
        "fp": int(values["fp"]),
        "fn": int(values["fn"]),
        "tn": int(values["tn"]),
        "empty": int(not pred.any()),
        "severe": int(float(values["f1"]) < 0.5),
    }


def historical_replay() -> dict[tuple[str, str], dict[str, str]]:
    path = P17C / "Phase17C_production_replay.csv"
    rows = load_csv(path)
    return {(row["sample"], row["frame"]): row for row in rows}


def expected_id(row: dict[str, str], policy: str) -> str:
    if policy == "base":
        return row.get("baseline_selected", "")
    return row.get("replay_selected", row.get("offline_expected", ""))


def main() -> int:
    evaluate = load_segmentation_metrics()
    manifest = load_csv(WORK / "configs/PHASE20C_INPUT_MANIFEST.csv")
    replay = historical_replay()
    stage_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    group_configs: dict[str, dict[str, object]] = {}
    for dataset in ("Easy21", "GT10"):
        for group in ("V0", "V1", "V2", "V3"):
            config_path = WORK / "configs" / f"{dataset}_{group}.json"
            group_configs[f"{dataset}/{group}"] = json.loads(config_path.read_text(encoding="utf-8"))

    for record in manifest:
        dataset = record["dataset"]
        group_rows = []
        for group in ("V0", "V1", "V2", "V3"):
            output = WORK / "runs" / group / dataset
            paths = stage_paths(output, record["stem"])
            present = all(path.exists() for path in paths.values())
            availability_rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "sample_id": record["sample_id"],
                    "frame": record["frame"],
                    "output_dir": str(output),
                    "all_stage_masks_present": int(present),
                    "missing_stages": ";".join(stage for stage, path in paths.items() if not path.exists()),
                }
            )
            if not present:
                continue
            gt = read_mask(ROOT / record["gt_path"])
            stage_metrics: dict[str, dict[str, float | int]] = {}
            for stage, path in paths.items():
                pred = read_mask(path)
                values = metric_values(evaluate, gt, pred)
                stage_metrics[stage] = values
                stage_rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "sample_id": record["sample_id"],
                        "frame": record["frame"],
                        "stem": record["stem"],
                        "stage": stage,
                        **values,
                        "mask_sha256": sha256(path),
                        "mask_area": int(pred.sum()),
                    }
                )
            selected_path = paths["selected"]
            candidates = sorted((ROOT / record["candidate_alias_dir"]).glob(f"mask_{record['stem']}_*.png"))
            selected_instance = ""
            selected_match = False
            oracle_f1 = 0.0
            oracle_instance = ""
            for candidate in candidates:
                candidate_values = metric_values(evaluate, gt, read_mask(candidate))
                if candidate_values["f1"] > oracle_f1:
                    oracle_f1 = float(candidate_values["f1"])
                    oracle_instance = str(candidate_instance(candidate, record["stem"]))
                if np.array_equal(read_mask(candidate), read_mask(selected_path)):
                    selected_instance = str(candidate_instance(candidate, record["stem"]))
                    selected_match = True
            selected_id = f"{record['stem']}_{int(selected_instance):02d}" if selected_instance != "" else ""
            hist = replay.get((record["sample_id"], record["frame"]), {})
            policy = str(group_configs[f"{dataset}/{group}"]["ranking_policy"])
            expected = expected_id(hist, policy) if hist else ""
            selection_rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "sample_id": record["sample_id"],
                    "frame": record["frame"],
                    "selected_instance": selected_instance,
                    "selected_candidate_id": selected_id,
                    "oracle_instance": oracle_instance,
                    "oracle_f1": oracle_f1,
                    "selected_f1": stage_metrics["selected"]["f1"],
                    "oracle_regret": oracle_f1 - float(stage_metrics["selected"]["f1"]),
                    "selected_matches_raw_candidate": int(selected_match),
                    "historical_expected_id": expected,
                    "historical_selection_match": int(bool(expected) and selected_id == expected),
                    "historical_replay_available": int(bool(hist)),
                }
            )
            selected = read_mask(paths["selected"])
            previous = selected
            stage_acceptance = {}
            for stage in ("spnp", "residual", "gcfc", "final"):
                current = read_mask(paths[stage])
                changed = np.logical_xor(previous, current)
                stage_acceptance[stage] = int(changed.any())
                module_rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "sample_id": record["sample_id"],
                        "frame": record["frame"],
                        "module": stage,
                        "enabled": int(stage != "final" and group in ("V2", "V3")),
                        "called": 1,
                        "accepted_or_changed": int(changed.any()),
                        "changed_pixel_ratio": float(changed.sum() / max(current.size, 1)),
                        "iou_with_previous": float((previous & current).sum() / max((previous | current).sum(), 1)),
                        "output_area": int(current.sum()),
                        "output_sha256": sha256(paths[stage]),
                    }
                )
                previous = current
            frame_rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "sample_id": record["sample_id"],
                    "frame": record["frame"],
                    "stem": record["stem"],
                    "ranking_policy": policy,
                    "refinement": group in ("V2", "V3"),
                    "selected_f1": stage_metrics["selected"]["f1"],
                    "selected_iou_fg": stage_metrics["selected"]["iou_fg"],
                    "final_f1": stage_metrics["final"]["f1"],
                    "final_iou_fg": stage_metrics["final"]["iou_fg"],
                    "final_miou": stage_metrics["final"]["miou"],
                    "final_hd95": stage_metrics["final"]["hd95"],
                    "final_boundary_f1": stage_metrics["final"]["boundary_f1"],
                    "final_leakage": stage_metrics["final"]["leakage"],
                    "final_empty": stage_metrics["final"]["empty"],
                    "final_severe": stage_metrics["final"]["severe"],
                    "refinement_changed_pixel_ratio": float(np.logical_xor(selected, read_mask(paths["final"])).sum() / selected.size),
                    "refinement_changed": int(np.any(np.logical_xor(selected, read_mask(paths["final"]))),),
                }
            )

    write_csv(WORK / "metrics/availability.csv", availability_rows)
    write_csv(WORK / "metrics/stage_metrics.csv", stage_rows)
    write_csv(WORK / "metrics/frame_metrics.csv", frame_rows)
    write_csv(WORK / "metrics/selection_replay.csv", selection_rows)
    write_csv(WORK / "metrics/module_execution.csv", module_rows)

    metric_names = ["final_f1", "final_iou_fg", "final_miou", "final_hd95", "final_boundary_f1", "final_leakage"]
    summary_rows: list[dict[str, object]] = []
    frame_index = defaultdict(list)
    for row in frame_rows:
        frame_index[(row["dataset"], row["group"])].append(row)
    for dataset in ("Easy21", "GT10"):
        for group in ("V0", "V1", "V2", "V3"):
            rows = frame_index.get((dataset, group), [])
            if not rows:
                summary_rows.append({"dataset": dataset, "group": group, "status": "BLOCKED", "n_frames": 0})
                continue
            by_sample = defaultdict(list)
            for row in rows:
                by_sample[row["sample_id"]].append(row)
            out: dict[str, object] = {"dataset": dataset, "group": group, "status": "COMPLETE", "n_frames": len(rows), "n_samples": len(by_sample)}
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in rows], dtype=float)
                sample_means = np.asarray([np.mean([float(row[metric]) for row in sample_rows]) for sample_rows in by_sample.values()])
                out[f"{metric}_frame_mean"] = float(np.mean(values))
                out[f"{metric}_sample_equal_mean"] = float(np.mean(sample_means))
                out[f"{metric}_median"] = float(np.median(values))
                out[f"{metric}_sd_population"] = float(np.std(values))
            out["empty_masks"] = int(sum(int(row["final_empty"]) for row in rows))
            out["severe_errors"] = int(sum(int(row["final_severe"]) for row in rows))
            out["refinement_changed_frames"] = int(sum(int(row["refinement_changed"]) for row in rows))
            out["refinement_changed_pixel_ratio_mean"] = float(np.mean([float(row["refinement_changed_pixel_ratio"]) for row in rows]))
            summary_rows.append(out)
    write_csv(WORK / "metrics/summary.csv", summary_rows)

    effect_rows: list[dict[str, object]] = []
    effects = {"ranking_V1_minus_V0": ("V1", "V0"), "refinement_R0_V2_minus_V0": ("V2", "V0"), "refinement_R1_V3_minus_V1": ("V3", "V1")}
    for dataset in ("Easy21", "GT10"):
        for effect, (left, right) in effects.items():
            left_rows = {(row["sample_id"], row["frame"]): row for row in frame_index.get((dataset, left), [])}
            right_rows = {(row["sample_id"], row["frame"]): row for row in frame_index.get((dataset, right), [])}
            keys = sorted(set(left_rows) & set(right_rows))
            row: dict[str, object] = {"dataset": dataset, "effect": effect, "n_matched_frames": len(keys)}
            for metric in metric_names:
                diffs = [float(left_rows[key][metric]) - float(right_rows[key][metric]) for key in keys]
                row[f"{metric}_paired_mean"] = float(np.mean(diffs)) if diffs else ""
                row[f"{metric}_paired_median"] = float(np.median(diffs)) if diffs else ""
                row[f"{metric}_paired_sd"] = float(np.std(diffs)) if diffs else ""
            effect_rows.append(row)
        v0 = {(row["sample_id"], row["frame"]): row for row in frame_index.get((dataset, "V0"), [])}
        v1 = {(row["sample_id"], row["frame"]): row for row in frame_index.get((dataset, "V1"), [])}
        v2 = {(row["sample_id"], row["frame"]): row for row in frame_index.get((dataset, "V2"), [])}
        v3 = {(row["sample_id"], row["frame"]): row for row in frame_index.get((dataset, "V3"), [])}
        keys = sorted(set(v0) & set(v1) & set(v2) & set(v3))
        row = {"dataset": dataset, "effect": "interaction_(V3-V2)-(V1-V0)", "n_matched_frames": len(keys)}
        for metric in metric_names:
            diffs = [
                (float(v3[key][metric]) - float(v2[key][metric])) - (float(v1[key][metric]) - float(v0[key][metric]))
                for key in keys
            ]
            row[f"{metric}_paired_mean"] = float(np.mean(diffs)) if diffs else ""
            row[f"{metric}_paired_median"] = float(np.median(diffs)) if diffs else ""
            row[f"{metric}_paired_sd"] = float(np.std(diffs)) if diffs else ""
        effect_rows.append(row)
    write_csv(WORK / "metrics/paired_effects.csv", effect_rows)
    print(json.dumps({"frame_rows": len(frame_rows), "stage_rows": len(stage_rows), "selection_rows": len(selection_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
