#!/usr/bin/env python3
"""Render, score, and summarize the three-scene M2M3 pilot run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_EXP_ROOT = Path("/data/fj/F2DMAS/00-论文优化重构/m2m3实验")
DEFAULT_OUTPUT_ROOT = DEFAULT_EXP_ROOT / "实验输出"
DEFAULT_M2M3_REPO = Path("/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40")
DEFAULT_PYTHON = DEFAULT_M2M3_REPO / "venv/bin/python"
DEFAULT_METHODS = ("fg2dgs_baseline", "fg2dgs_m2m3", "fg2dgs_m2m3_floor40")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_run_tag(output_root: Path) -> str:
    runs_root = output_root / "runs"
    candidates = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {runs_root}")
    return sorted(candidates, key=lambda p: (p.name, p.stat().st_mtime))[-1].name


def parse_ply_vertex_count(path: Path) -> int | str:
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return ""


def latest_ply(model_dir: Path) -> tuple[int | str, Path | None]:
    candidates: list[tuple[int, Path]] = []
    for ply in model_dir.glob("point_cloud/iteration_*/point_cloud.ply"):
        try:
            iteration = int(ply.parent.name.split("_")[-1])
        except ValueError:
            continue
        candidates.append((iteration, ply))
    if not candidates:
        return "", None
    return sorted(candidates)[-1]


def parse_command_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.is_file():
        return metadata
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("# ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def parse_duration(value: str) -> int | str:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return ""


def format_seconds(value: Any) -> str:
    if value in ("", None):
        return ""
    seconds = int(round(float(value)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def parse_training_log(path: Path, started_at: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")

    progress_elapsed_sec: int | str = ""
    final_iter_per_sec: float | str = ""
    for match in re.finditer(
        r"30000/30000 \[(?P<elapsed>[0-9:]+)<00:00,\s*(?P<speed>[0-9.]+)it/s",
        text,
    ):
        progress_elapsed_sec = parse_duration(match.group("elapsed"))
        final_iter_per_sec = float(match.group("speed"))

    completed_at = ""
    wall_elapsed_sec: int | str = ""
    complete_match = re.search(r"Training complete\. \[(?P<stamp>\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]", text)
    if complete_match:
        completed_at = complete_match.group("stamp")
    if started_at and completed_at:
        try:
            start_dt = datetime.fromisoformat(started_at)
            finish_naive = datetime.strptime(f"{start_dt.year}/{completed_at}", "%Y/%d/%m %H:%M:%S")
            finish_dt = finish_naive.replace(tzinfo=start_dt.tzinfo)
            if finish_dt < start_dt:
                finish_dt += timedelta(days=1)
            wall_elapsed_sec = int((finish_dt - start_dt).total_seconds())
        except ValueError:
            wall_elapsed_sec = ""

    return {
        "train_completed_at": completed_at,
        "train_wall_elapsed_sec": wall_elapsed_sec,
        "train_progress_elapsed_sec": progress_elapsed_sec,
        "train_final_iter_per_sec": final_iter_per_sec,
    }


def run_logged(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return elapsed


def has_render_outputs(model_dir: Path, iteration: int) -> bool:
    render_dir = model_dir / "test" / f"ours_{iteration}" / "renders"
    gt_dir = model_dir / "test" / f"ours_{iteration}" / "gt"
    return render_dir.is_dir() and gt_dir.is_dir() and any(render_dir.iterdir()) and any(gt_dir.iterdir())


def has_metrics(model_dir: Path, iteration: int) -> bool:
    results = read_json(model_dir / "results.json")
    return f"ours_{iteration}" in results


def count_rendered_views(model_dir: Path, iteration: int) -> tuple[int | str, int | str]:
    render_dir = model_dir / "test" / f"ours_{iteration}" / "renders"
    gt_dir = model_dir / "test" / f"ours_{iteration}" / "gt"
    render_count = len([p for p in render_dir.iterdir() if p.is_file()]) if render_dir.is_dir() else ""
    gt_count = len([p for p in gt_dir.iterdir() if p.is_file()]) if gt_dir.is_dir() else ""
    return render_count, gt_count


def model_dirs_for(
    output_root: Path,
    run_tag: str,
    scenes: list[str] | None,
    methods: list[str] | None,
) -> list[Path]:
    run_root = output_root / "runs" / run_tag
    if not run_root.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_root}")

    scene_names = scenes or sorted(p.name for p in run_root.iterdir() if p.is_dir())
    method_names = methods or list(DEFAULT_METHODS)
    model_dirs: list[Path] = []
    for scene in scene_names:
        for method in method_names:
            model_dir = run_root / scene / method
            if model_dir.is_dir():
                model_dirs.append(model_dir)
    return model_dirs


def safe_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def pct_delta(value: Any, baseline: Any, invert: bool = False) -> float | str:
    value_f = safe_float(value)
    baseline_f = safe_float(baseline)
    if value_f is None or baseline_f in (None, 0.0):
        return ""
    delta = (value_f - baseline_f) / baseline_f * 100.0
    return -delta if invert else delta


def abs_delta(value: Any, baseline: Any) -> float | str:
    value_f = safe_float(value)
    baseline_f = safe_float(baseline)
    if value_f is None or baseline_f is None:
        return ""
    return value_f - baseline_f


def collect_row(model_dir: Path, output_root: Path, run_tag: str, iteration: int, runtimes: dict[str, Any]) -> dict[str, Any]:
    scene, method = model_dir.parent.name, model_dir.name
    key = f"{scene}/{method}"
    command_meta = parse_command_metadata(model_dir / "command.txt")
    log_path = output_root / "logs" / f"{run_tag}_{scene}_{method}.log"
    train_log = parse_training_log(log_path, command_meta.get("started_at", ""))

    final_iteration, ply_path = latest_ply(model_dir)
    if (not final_iteration or int(final_iteration) < iteration) and (model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").is_file():
        final_iteration = iteration
        ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    ply_bytes: int | str = ply_path.stat().st_size if ply_path else ""
    ply_mb = round(ply_bytes / 1_000_000, 3) if isinstance(ply_bytes, int) else ""
    point_count = parse_ply_vertex_count(ply_path) if ply_path else ""
    render_count, gt_count = count_rendered_views(model_dir, iteration)

    capacity = read_json(model_dir / "capacity_control" / "capacity_summary.json")
    foreground = read_json(model_dir / "foreground_init_pcd_report.json")
    pruning = read_json(model_dir / "pruning" / f"pruning_iter_{iteration}.json")
    results = read_json(model_dir / "results.json").get(f"ours_{iteration}", {})
    runtime = runtimes.get(key, {})

    return {
        "run_tag": run_tag,
        "scene": scene,
        "method": method,
        "model_dir": str(model_dir),
        "image_count": command_meta.get("image_count", ""),
        "mask_matched_gate_count": command_meta.get("mask_matched_gate_count", ""),
        "init_pcd_mode": command_meta.get("init_pcd_mode", ""),
        "started_at": command_meta.get("started_at", ""),
        **train_log,
        "train_wall_time": format_seconds(train_log.get("train_wall_elapsed_sec", "")),
        "train_progress_time": format_seconds(train_log.get("train_progress_elapsed_sec", "")),
        "render_elapsed_sec": runtime.get("render_elapsed_sec", ""),
        "render_time": format_seconds(runtime.get("render_elapsed_sec", "")),
        "metrics_elapsed_sec": runtime.get("metrics_elapsed_sec", ""),
        "metrics_time": format_seconds(runtime.get("metrics_elapsed_sec", "")),
        "final_iteration": final_iteration,
        "final_point_count": point_count,
        "final_ply_bytes": ply_bytes,
        "final_ply_mb": ply_mb,
        "test_render_count": render_count,
        "test_gt_count": gt_count,
        "psnr": results.get("PSNR", ""),
        "ssim": results.get("SSIM", ""),
        "lpips": results.get("LPIPS", ""),
        "capacity_mode": capacity.get("capacity_control_mode", "none"),
        "capacity_initial_count": capacity.get("initial_count", ""),
        "capacity_max_seen_count": capacity.get("max_seen_count", ""),
        "capacity_final_count": capacity.get("final_count", ""),
        "capacity_rounds": capacity.get("rounds", ""),
        "capacity_total_requested": capacity.get("total_requested", ""),
        "capacity_total_removed": capacity.get("total_removed", ""),
        "capacity_total_blocked_by_floor": capacity.get("total_blocked_by_floor", ""),
        "capacity_floor_active_rounds": capacity.get("floor_active_rounds", ""),
        "capacity_floor_ratio": capacity.get("capacity_floor_ratio", ""),
        "capacity_floor_reference": capacity.get("capacity_floor_reference", ""),
        "foreground_init_points_before": foreground.get("points_before", ""),
        "foreground_init_points_after": foreground.get("points_after", ""),
        "foreground_init_kept_ratio": foreground.get("kept_ratio", ""),
        "pruning_gaussians_before": pruning.get("gaussians_before", ""),
        "pruning_gaussians_after": pruning.get("gaussians_after", ""),
        "pruning_removed": pruning.get("removed", ""),
        "pruning_ratio": pruning.get("pruning_ratio", ""),
        "m2m3_requested_iter": pruning.get("m2m3_requested", ""),
        "m2m3_after_budget_iter": pruning.get("m2m3_after_budget", ""),
        "m2m3_budget_blocked_iter": pruning.get("m2m3_budget_blocked", ""),
        "floor_active_iter": pruning.get("floor_active", ""),
        "floor_count_iter": pruning.get("floor_count", ""),
        "floor_allowed_remove_iter": pruning.get("floor_allowed_remove", ""),
        "floor_blocked_iter": pruning.get("floor_blocked", ""),
    }


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {row["scene"]: row for row in rows if row["method"] == "fg2dgs_baseline"}
    for row in rows:
        baseline = baselines.get(row["scene"])
        if not baseline:
            continue
        row["point_reduction_pct_vs_baseline"] = pct_delta(
            row.get("final_point_count", ""), baseline.get("final_point_count", ""), invert=True
        )
        row["ply_size_reduction_pct_vs_baseline"] = pct_delta(
            row.get("final_ply_bytes", ""), baseline.get("final_ply_bytes", ""), invert=True
        )
        row["train_time_reduction_pct_vs_baseline"] = pct_delta(
            row.get("train_wall_elapsed_sec", ""), baseline.get("train_wall_elapsed_sec", ""), invert=True
        )
        row["psnr_delta_vs_baseline"] = abs_delta(row.get("psnr", ""), baseline.get("psnr", ""))
        row["ssim_delta_vs_baseline"] = abs_delta(row.get("ssim", ""), baseline.get("ssim", ""))
        row["lpips_delta_vs_baseline"] = abs_delta(row.get("lpips", ""), baseline.get("lpips", ""))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag",
        "scene",
        "method",
        "image_count",
        "mask_matched_gate_count",
        "init_pcd_mode",
        "final_iteration",
        "psnr",
        "ssim",
        "lpips",
        "final_point_count",
        "final_ply_mb",
        "final_ply_bytes",
        "point_reduction_pct_vs_baseline",
        "ply_size_reduction_pct_vs_baseline",
        "train_wall_time",
        "train_wall_elapsed_sec",
        "train_progress_time",
        "train_progress_elapsed_sec",
        "train_time_reduction_pct_vs_baseline",
        "train_final_iter_per_sec",
        "render_time",
        "render_elapsed_sec",
        "metrics_time",
        "metrics_elapsed_sec",
        "test_render_count",
        "test_gt_count",
        "psnr_delta_vs_baseline",
        "ssim_delta_vs_baseline",
        "lpips_delta_vs_baseline",
        "capacity_mode",
        "capacity_initial_count",
        "capacity_max_seen_count",
        "capacity_final_count",
        "capacity_rounds",
        "capacity_total_requested",
        "capacity_total_removed",
        "capacity_total_blocked_by_floor",
        "capacity_floor_active_rounds",
        "capacity_floor_ratio",
        "capacity_floor_reference",
        "foreground_init_points_before",
        "foreground_init_points_after",
        "foreground_init_kept_ratio",
        "pruning_gaussians_before",
        "pruning_gaussians_after",
        "pruning_removed",
        "pruning_ratio",
        "m2m3_requested_iter",
        "m2m3_after_budget_iter",
        "m2m3_budget_blocked_iter",
        "floor_active_iter",
        "floor_count_iter",
        "floor_allowed_remove_iter",
        "floor_blocked_iter",
        "started_at",
        "train_completed_at",
        "model_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    value_f = safe_float(value)
    if value_f is None:
        return ""
    return f"{value_f:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], run_tag: str) -> None:
    report_dir = path.parent
    output_root = report_dir.parent
    run_root = output_root / "runs" / run_tag
    log_root = output_root / "logs"
    csv_path = report_dir / f"full_evaluation_summary_{run_tag}.csv"
    runtime_path = report_dir / f"evaluation_runtime_{run_tag}.json"
    lines = [
        f"# M2M3 full evaluation summary: {run_tag}",
        "",
        "PSNR/SSIM higher is better; LPIPS lower is better. Reduction columns are relative to the scene baseline.",
        "",
        "## Output files",
        "",
        f"- Full run directory: `{run_root}`",
        f"- Training/evaluation logs: `{log_root}`",
        f"- Summary CSV: `{csv_path}`",
        f"- Summary Markdown: `{path}`",
        f"- Evaluation runtime JSON: `{runtime_path}`",
        f"- Capacity and point-cloud summary: `{report_dir / 'capacity_and_pointcloud_summary.csv'}`",
        f"- Per-model final PLY: `{run_root}/<SCENE>/<METHOD>/point_cloud/iteration_30000/point_cloud.ply`",
        f"- Per-model rendered test views: `{run_root}/<SCENE>/<METHOD>/test/ours_30000/{{renders,gt,vis}}`",
        f"- Per-model average metrics: `{run_root}/<SCENE>/<METHOD>/results.json`",
        f"- Per-model per-view metrics: `{run_root}/<SCENE>/<METHOD>/per_view.json`",
        f"- Per-model capacity reports: `{run_root}/<SCENE>/<METHOD>/capacity_control/`",
        f"- Per-model pruning reports: `{run_root}/<SCENE>/<METHOD>/pruning/`",
        "",
    ]
    for scene in sorted({row["scene"] for row in rows}):
        lines.append(f"## {scene}")
        lines.append("")
        lines.append(
            "| method | PSNR | SSIM | LPIPS | points | PLY MB | size red. | point red. | train | PSNR d | SSIM d | LPIPS d |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in [item for item in rows if item["scene"] == scene]:
            lines.append(
                "| {method} | {psnr} | {ssim} | {lpips} | {points} | {mb} | {size_red}% | {point_red}% | {train} | {psnr_d} | {ssim_d} | {lpips_d} |".format(
                    method=row["method"],
                    psnr=fmt(row.get("psnr"), 3),
                    ssim=fmt(row.get("ssim"), 4),
                    lpips=fmt(row.get("lpips"), 4),
                    points=row.get("final_point_count", ""),
                    mb=fmt(row.get("final_ply_mb"), 1),
                    size_red=fmt(row.get("ply_size_reduction_pct_vs_baseline"), 1),
                    point_red=fmt(row.get("point_reduction_pct_vs_baseline"), 1),
                    train=row.get("train_wall_time", ""),
                    psnr_d=fmt(row.get("psnr_delta_vs_baseline"), 3),
                    ssim_d=fmt(row.get("ssim_delta_vs_baseline"), 4),
                    lpips_d=fmt(row.get("lpips_delta_vs_baseline"), 4),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--m2m3-repo", type=Path, default=DEFAULT_M2M3_REPO)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--scenes", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    args = parser.parse_args()

    run_tag = args.run_tag or latest_run_tag(args.output_root)
    scenes = split_csv(args.scenes) if args.scenes else None
    methods = split_csv(args.methods) if args.methods else None
    model_dirs = model_dirs_for(args.output_root, run_tag, scenes, methods)
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found for run_tag={run_tag}")

    report_dir = args.output_root / "reports"
    runtime_path = report_dir / f"evaluation_runtime_{run_tag}.json"
    runtimes = read_json(runtime_path)
    env = os.environ.copy()

    for model_dir in model_dirs:
        scene, method = model_dir.parent.name, model_dir.name
        key = f"{scene}/{method}"
        runtimes.setdefault(key, {})
        print(f"[eval] {scene} {method}")

        if not args.skip_render and (args.force_render or not has_render_outputs(model_dir, args.iteration)):
            render_log = args.output_root / "logs" / f"{run_tag}_{scene}_{method}_render_iter{args.iteration}.log"
            command = [
                str(args.python_bin),
                "render.py",
                "-m",
                str(model_dir),
                "--iteration",
                str(args.iteration),
                "--skip_train",
                "--skip_mesh",
            ]
            elapsed = run_logged(command, args.m2m3_repo, render_log, env)
            if not has_render_outputs(model_dir, args.iteration):
                raise RuntimeError(f"Render finished but outputs are missing for {model_dir}")
            runtimes[key]["render_elapsed_sec"] = elapsed
            runtimes[key]["render_log"] = str(render_log)
            write_json(runtime_path, runtimes)
            print(f"  render: {format_seconds(elapsed)}")
        else:
            print("  render: skipped")

        if not args.skip_metrics and (args.force_metrics or not has_metrics(model_dir, args.iteration)):
            metrics_log = args.output_root / "logs" / f"{run_tag}_{scene}_{method}_metrics_iter{args.iteration}.log"
            metrics_script = Path(__file__).with_name("compute_render_metrics.py")
            command = [
                str(args.python_bin),
                str(metrics_script),
                "-m",
                str(model_dir),
                "--iteration",
                str(args.iteration),
                "--lpips-net",
                "vgg",
            ]
            elapsed = run_logged(command, args.m2m3_repo, metrics_log, env)
            if not has_metrics(model_dir, args.iteration):
                raise RuntimeError(f"Metrics finished but results.json has no ours_{args.iteration} for {model_dir}")
            runtimes[key]["metrics_elapsed_sec"] = elapsed
            runtimes[key]["metrics_log"] = str(metrics_log)
            write_json(runtime_path, runtimes)
            print(f"  metrics: {format_seconds(elapsed)}")
        else:
            print("  metrics: skipped")

    rows = [collect_row(model_dir, args.output_root, run_tag, args.iteration, runtimes) for model_dir in model_dirs]
    rows.sort(key=lambda row: (row["scene"], DEFAULT_METHODS.index(row["method"]) if row["method"] in DEFAULT_METHODS else 99))
    add_baseline_deltas(rows)

    csv_path = report_dir / f"full_evaluation_summary_{run_tag}.csv"
    md_path = report_dir / f"full_evaluation_summary_{run_tag}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, run_tag)
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
