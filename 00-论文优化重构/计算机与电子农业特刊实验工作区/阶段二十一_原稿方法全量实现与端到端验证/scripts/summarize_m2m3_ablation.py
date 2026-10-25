#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — collect the M0–M4 ablation into one auditable summary.

Reads only artifacts that already exist on disk:

  runs/<run_name>/<job_id>/eval/foreground_test.json   (rendered foreground metrics)
  runs/pipeline/<job_id>/run_manifest.json            (commands, git commit, masks)
  logs/pipeline/<job_id>/train.log                    (M2M3-GO round bookkeeping)
  runs/<run_name>/<job_id>/model/point_cloud/iteration_*/point_cloud.ply
                                                      (final Gaussian count)

No value is recomputed or inferred: a variant with a missing eval file is
reported with status MISSING rather than filled in from another run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
VARIANT_LABELS = {
    "C0": "full-scene 2DGS control (no foreground-object components)",
    "M0": "foreground-object 2DGS",
    "M1": "M0 + mask-guided pruning (legacy topology capacity)",
    "M2": "M0 + M2M3-GO utility pruning (no Floor40, no densify)",
    "M3": "M2 + Floor40 local capacity floor",
    "M4": "M3 + deficit-driven densification (complete M2M3-GO)",
}
METRIC_KEYS = [
    "psnr_fg", "ssim_fg", "lpips_fg", "outside_mask_non_black_ratio",
    "leakage_energy_ratio", "foreground_fraction",
]
M2M3_LINE = re.compile(
    r"M2M3-GO (\w+): pruned (\d+)/(\d+) \(candidates=(\d+)\), densified (\d+), "
    r"floor_blocked=(\d+), U_mean=([0-9.]+)")
POINTS_LINE = re.compile(r"Points=(\d+)")


def ply_vertex_count(path: Path) -> int | None:
    """Read only the PLY header; the body is never loaded."""
    try:
        with path.open("rb") as handle:
            head = handle.read(2048)
    except OSError:
        return None
    match = re.search(rb"element vertex (\d+)", head)
    return int(match.group(1)) if match else None


def latest_point_cloud(model_dir: Path) -> Path | None:
    root = model_dir / "point_cloud"
    if not root.is_dir():
        return None

    def iteration_of(path: Path) -> int:
        match = re.search(r"iteration_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    newest = max(candidates, key=iteration_of)
    ply = newest / "point_cloud.ply"
    return ply if ply.is_file() else None


def read_log_summary(path: Path) -> dict:
    """Extract the M2M3-GO round bookkeeping from the training log."""
    out = {"m2m3_rounds": 0, "pruned_total": 0, "densified_total": 0,
           "floor_blocked_total": 0, "init_points": None, "final_points": None}
    if not path.is_file():
        return out
    text = path.read_text(errors="replace").replace("\r", "\n")
    for match in M2M3_LINE.finditer(text):
        out["m2m3_rounds"] += 1
        out["pruned_total"] += int(match.group(2))
        out["densified_total"] += int(match.group(5))
        out["floor_blocked_total"] += int(match.group(6))
    init = re.search(r"Number of points at initialisation\s*:\s*(\d+)", text)
    if init:
        out["init_points"] = int(init.group(1))
    points = POINTS_LINE.findall(text)
    if points:
        out["final_points"] = int(points[-1])
    return out


def read_manifest(job_id: str, pipeline_root: Path) -> dict:
    path = pipeline_root / job_id / "run_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def stage(manifest: dict, name: str) -> dict:
    for item in manifest.get("stages", []):
        if item.get("stage") == name:
            return item
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="m2m3_ablation")
    ap.add_argument("--runs-root", default=str(PHASE / "runs"))
    ap.add_argument("--pipeline-root", default=str(PHASE / "runs/pipeline"))
    ap.add_argument("--log-root", default=str(PHASE / "logs/pipeline"))
    ap.add_argument("--out", default=str(PHASE / "metrics/m2m3_ablation_summary.csv"))
    ap.add_argument("--markdown", default=str(PHASE / "metrics/m2m3_ablation_summary.md"))
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    pipeline_root = Path(args.pipeline_root)
    log_root = Path(args.log_root)

    rows = []
    for job_dir in sorted((runs_root / args.run_name).glob("*")):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        variant = job_id.rsplit("_", 1)[-1]
        if variant not in VARIANT_LABELS:
            continue
        eval_json = job_dir / "eval" / "foreground_test.json"
        manifest = read_manifest(job_id, pipeline_root)
        train_stage = stage(manifest, "train")
        eval_stage = stage(manifest, "eval")
        log_info = read_log_summary(log_root / job_id / "train.log")
        ply = latest_point_cloud(job_dir / "model")
        row = {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "job_id": job_id,
            "status": "MISSING_EVAL",
            "git_commit": manifest.get("git", {}).get("commit", ""),
            "git_dirty": manifest.get("git", {}).get("dirty", ""),
            "train_returncode": train_stage.get("returncode", ""),
            "train_wall_sec": round(float(train_stage.get("wall_sec", 0.0)), 1),
            "eval_returncode": eval_stage.get("returncode", ""),
            "evaluated_views": "",
            "lpips_backend": "",
            "gaussians_init": log_info["init_points"] or "",
            "gaussians_final": ply_vertex_count(ply) if ply else "",
            "m2m3_rounds": log_info["m2m3_rounds"],
            "pruned_total": log_info["pruned_total"],
            "densified_total": log_info["densified_total"],
            "floor_blocked_total": log_info["floor_blocked_total"],
            "masks_sha256": (manifest.get("consumed_masks", {}).get("combined_sha256")
                             or manifest.get("consumed_masks", {}).get("sha256", "")),
            "eval_json": str(eval_json.relative_to(PHASE)) if eval_json.is_file() else "",
        }
        for key in METRIC_KEYS:
            row[key] = ""
        if eval_json.is_file():
            payload = json.loads(eval_json.read_text())
            summary = payload.get("summary", {})
            row["status"] = "OK" if train_stage.get("returncode") == 0 and eval_stage.get("returncode") == 0 else "PARTIAL"
            row["evaluated_views"] = summary.get("evaluated_views", "")
            row["lpips_backend"] = summary.get("lpips_backend", "") or ""
            for key in METRIC_KEYS:
                value = summary.get(key)
                row[key] = "" if value is None else float(value["mean"])
        rows.append(row)

    if not rows:
        print("no runs found under {}".format(runs_root / args.run_name), file=sys.stderr)
        return 1

    rows.sort(key=lambda item: item["variant"])
    columns = ["variant", "variant_label", "status", "evaluated_views", "psnr_fg", "ssim_fg",
               "lpips_fg", "lpips_backend", "outside_mask_non_black_ratio", "leakage_energy_ratio",
               "foreground_fraction", "gaussians_init", "gaussians_final", "m2m3_rounds",
               "pruned_total", "densified_total", "floor_blocked_total", "train_wall_sec",
               "train_returncode", "eval_returncode", "git_commit", "git_dirty",
               "masks_sha256", "job_id", "eval_json"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def fmt(value, digits=4):
        return "" if value == "" else "{:.{}f}".format(float(value), digits)

    lines = [
        "# M0–M4 M2M3-GO ablation (generated by summarize_m2m3_ablation.py)",
        "",
        "| " + " | ".join(["variant", "label", "status", "views", "PSNR_fg", "SSIM_fg",
                            "outside", "leakage", "Gaussians", "pruned", "densified",
                            "floor_blocked", "wall_s"]) + " |",
        "| " + " | ".join(["---"] * 13) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            row["variant"], row["variant_label"], row["status"],
            str(row["evaluated_views"]), fmt(row["psnr_fg"], 3), fmt(row["ssim_fg"], 4),
            fmt(row["outside_mask_non_black_ratio"], 4), fmt(row["leakage_energy_ratio"], 4),
            str(row["gaussians_final"]), str(row["pruned_total"]),
            str(row["densified_total"]), str(row["floor_blocked_total"]),
            fmt(row["train_wall_sec"], 0),
        ]) + " |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote {} and {}".format(out_path, args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
