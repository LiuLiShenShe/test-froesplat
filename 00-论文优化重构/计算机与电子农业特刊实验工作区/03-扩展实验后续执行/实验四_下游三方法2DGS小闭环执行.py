#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run experiment 4b: downstream 2DGS closed loop over three mask sources.

The experiment keeps the COLMAP scene and 2DGS/ForeSplat hyper-parameters fixed
within each sample, and only swaps the foreground mask source:

1. SAM3 single prompt
2. FSAM3-base
3. RAP-FSAM3-v2

Default mode is a dry run. Add ``--execute`` to actually launch training and
evaluation commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/fj/F2DMAS")
PAPER_ROOT = ROOT / "00-论文优化重构"
DATA_ROOT = PAPER_ROOT / "数据管理"
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent

REPO_PATH = ROOT / "2d-gaussian-splatting-main"
PYTHON_BIN = REPO_PATH / "venv/bin/python"
OUTPUT_ROOT = DATA_ROOT / "06-实验输出"
RESULT_ROOT = DATA_ROOT / "05-评测结果/S24_E4b_mask_source_2DGS_closed_loop"

FULL_METRICS_SCRIPT = (
    DATA_ROOT
    / "07-运行脚本与超参/S4-2DGS-baseline回归/scripts/evaluate_rendered_metrics.py"
)
FOREGROUND_METRICS_SCRIPT = (
    DATA_ROOT
    / "07-运行脚本与超参/S11-plant-only导出/scripts/evaluate_foreground_object_metrics.py"
)

DEFAULT_SAMPLES = ["KongQueZhuYu", "XianKeLai1", "CaoMei2"]
DEFAULT_STEPS = ["train", "render", "metrics", "foreground-metrics", "mesh"]

SOURCE_PATHS = {
    "KongQueZhuYu": DATA_ROOT / "02-位姿COLMAP/03-final_locked/KongQueZhuYu",
    "XianKeLai1": DATA_ROOT / "02-位姿COLMAP/04-sanitized_for_A6/XianKeLai1",
    "CaoMei2": DATA_ROOT / "02-位姿COLMAP/04-sanitized_for_A6/CaoMei2",
}


@dataclass(frozen=True)
class MaskSource:
    source_id: str
    label: str
    description: str
    default_templates: tuple[str, ...]


MASK_SOURCES = {
    "SAM3_single_prompt": MaskSource(
        source_id="SAM3_single_prompt",
        label="SAM3 single prompt",
        description="single-prompt SAM3 historical mask entry",
        default_templates=(
            str(DATA_ROOT / "03-分割Mask/02-sam_masks/{sample}"),
        ),
    ),
    "FSAM3_base": MaskSource(
        source_id="FSAM3_base",
        label="FSAM3-base",
        description="base FSAM3 mask entry; currently expected under the reserved FSAM3 directory",
        default_templates=(
            str(DATA_ROOT / "03-分割Mask/04-fsam3_masks/{sample}/最终掩膜"),
            str(DATA_ROOT / "03-分割Mask/04-fsam3_masks/{sample}"),
        ),
    ),
    "RAP_FSAM3_v2": MaskSource(
        source_id="RAP_FSAM3_v2",
        label="RAP-FSAM3-v2",
        description="final RAP-FSAM3-v2 masks; prefer the E4b full-sequence directory",
        default_templates=(
            str(DATA_ROOT / "03-分割Mask/05-RAP-FSAM3掩膜/E4b_downstream/{sample}/最终掩膜"),
            str(DATA_ROOT / "03-分割Mask/05-RAP-FSAM3掩膜/E4b_{sample}_RAP-FSAM3-v2/最终掩膜"),
            str(
                WORKSPACE_DIR
                / "06-实验一四样本代表集/E1_Representative4_PottedPlant_VFM/rap_runs/RAP-FSAM3-v2/{sample}/最终掩膜"
            ),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--mask-sources",
        nargs="+",
        default=list(MASK_SOURCES),
        choices=list(MASK_SOURCES),
    )
    parser.add_argument("--sam3-mask-root", type=str, default="")
    parser.add_argument("--fsam3-mask-root", type=str, default="")
    parser.add_argument("--rap-mask-root", type=str, default="")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--metrics-device", default="cuda:0")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only writes a dry-run manifest.")
    parser.add_argument("--steps", nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-timestamp-if-exists", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=17100)
    parser.add_argument("--mesh-depth-trunc", type=float, default=6.0)
    parser.add_argument("--mesh-voxel-size", type=float, default=0.02)
    parser.add_argument("--mesh-sdf-trunc", type=float, default=0.08)
    parser.add_argument("--mesh-num-cluster", type=int, default=50)
    return parser.parse_args()


def source_path_for(sample: str) -> Path:
    if sample in SOURCE_PATHS:
        return SOURCE_PATHS[sample]
    final_locked = DATA_ROOT / "02-位姿COLMAP/03-final_locked" / sample
    sanitized = DATA_ROOT / "02-位姿COLMAP/04-sanitized_for_A6" / sample
    return sanitized if sanitized.exists() else final_locked


def image_files(source_path: Path) -> list[Path]:
    image_dir = source_path / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing COLMAP image directory: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def ensure_scene(source_path: Path) -> None:
    required = [
        source_path / "images",
        source_path / "sparse/0/cameras.bin",
        source_path / "sparse/0/images.bin",
        source_path / "sparse/0/points3D.bin",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required COLMAP scene files:\n" + "\n".join(missing))


def fill_template(template: str, sample: str) -> Path:
    return Path(template.format(sample=sample))


def override_template(args: argparse.Namespace, source_id: str) -> str:
    if source_id == "SAM3_single_prompt":
        return args.sam3_mask_root
    if source_id == "FSAM3_base":
        return args.fsam3_mask_root
    if source_id == "RAP_FSAM3_v2":
        return args.rap_mask_root
    return ""


def template_to_sample_dir(template: str, sample: str) -> Path:
    root = Path(template.format(sample=sample)) if "{sample}" in template else Path(template) / sample
    return root


def resolve_raw_mask_dir(args: argparse.Namespace, source: MaskSource, sample: str) -> tuple[Path, list[Path]]:
    candidates: list[Path] = []
    override = override_template(args, source.source_id)
    if override:
        candidates.append(template_to_sample_dir(override, sample))
    candidates.extend(fill_template(template, sample) for template in source.default_templates)
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return candidates[0], candidates


def possible_mask_names(stem: str, image_name: str) -> list[str]:
    names: list[str] = []
    stems = [stem]
    bare_stems: list[str] = []
    if stem.startswith("crop_"):
        stripped = stem[len("crop_") :]
        stems.append(stripped)
        bare_stems.append(stripped)
    else:
        stems.append(f"crop_{stem}")
        bare_stems.append(stem)
    stems = list(dict.fromkeys(stems))
    bare_stems = list(dict.fromkeys(bare_stems))
    suffixes = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    for item in stems:
        for suffix in suffixes:
            names.append(f"mask_{item}{suffix}")
            names.append(f"masks/mask_{item}{suffix}")
    for item in bare_stems:
        for suffix in suffixes:
            names.append(f"{item}{suffix}")
            names.append(f"masks/{item}{suffix}")
    return list(dict.fromkeys(names))


def looks_like_image(path: Path) -> bool:
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return False
    return (
        header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith(b"\xff\xd8\xff")
        or header[:4].lower() in {b"ii*\x00", b"mm\x00*"}
    )


def find_mask(raw_mask_dir: Path, image: Path) -> Path | None:
    for name in possible_mask_names(image.stem, image.name):
        candidate = raw_mask_dir / name
        if candidate.exists() and looks_like_image(candidate):
            return candidate
    return None


def rebuild_prepared_masks(raw_mask_dir: Path, prepared_dir: Path, images: list[Path]) -> tuple[int, list[str]]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for existing in prepared_dir.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
        else:
            raise RuntimeError(f"Unexpected directory inside managed prepared mask dir: {existing}")

    missing: list[str] = []
    matched = 0
    for image in images:
        source_mask = find_mask(raw_mask_dir, image)
        if source_mask is None:
            missing.append(image.name)
            continue
        target = prepared_dir / f"mask_{image.stem}.png"
        target.symlink_to(source_mask.resolve())
        matched += 1
    return matched, missing


def inspect_masks(raw_mask_dir: Path, images: list[Path]) -> tuple[int, list[str]]:
    missing: list[str] = []
    matched = 0
    if not raw_mask_dir.exists():
        return 0, [image.name for image in images]
    for image in images:
        if find_mask(raw_mask_dir, image) is None:
            missing.append(image.name)
        else:
            matched += 1
    return matched, missing


def output_dir_for(args: argparse.Namespace, sample: str, source_id: str) -> Path:
    method_tag = f"E4b_{source_id}"
    output_dir = args.output_root / sample / method_tag
    if output_dir.exists() and args.append_timestamp_if_exists:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_root / sample / f"{method_tag}_{stamp}"
    return output_dir


def add_mask_and_foreground_switches(command: list[str], prepared_mask_dir: Path) -> None:
    switches: list[str] = [
        "--view_quality_mode",
        "none",
        "--raw_gate_mode",
        "none",
        "--mask_gate_mode",
        "none",
        "--geo_gate_mode",
        "none",
        "--mask_mode",
        "alpha",
        "--mask_dir",
        str(prepared_mask_dir),
        "--mask_pattern",
        "mask_{stem}.png",
        "--mask_threshold",
        "0.5",
        "--init_pcd_mode",
        "foreground_track",
        "--init_pcd_min_observations",
        "3",
        "--init_pcd_foreground_threshold",
        "0.9",
        "--init_pcd_dilate_mask_px",
        "0",
        "--init_pcd_max_cameras",
        "0",
        "--init_pcd_chunk_size",
        "200000",
    ]
    command.extend(switches)


def build_train_command(
    args: argparse.Namespace,
    source_path: Path,
    output_dir: Path,
    prepared_mask_dir: Path,
    port: int,
) -> list[str]:
    save_iterations = [value for value in (7000, args.iterations) if value <= args.iterations]
    command = [
        str(PYTHON_BIN),
        "train.py",
        "--source_path",
        str(source_path),
        "--model_path",
        str(output_dir),
    ]
    if args.eval:
        command.append("--eval")
    command.extend(["--resolution", str(args.resolution)])
    add_mask_and_foreground_switches(command, prepared_mask_dir)
    command.extend(
        [
            "--use_foreground_rgb_loss",
            "--lambda_fg_rgb",
            "1.0",
            "--foreground_bg_rgb_weight",
            "0.0",
            "--foreground_rgb_crop_padding",
            "12",
            "--use_mask_loss",
            "--use_bg_opacity_loss",
            "--lambda_mask",
            "0.08",
            "--lambda_bg",
            "0.02",
            "--mask_loss_type",
            "l1_dice",
            "--mask_ignore_boundary_px",
            "2",
            "--mask_loss_start_iter",
            "500",
            "--mask_loss_warmup_iters",
            "1500",
            "--pruning_mode",
            "mask",
            "--pruning_start_iter",
            "18000",
            "--pruning_interval",
            "3000",
            "--pruning_opacity_threshold",
            "0.005",
            "--pruning_brightness_threshold",
            "0.01",
            "--pruning_mask_threshold",
            "0.45",
            "--pruning_mask_max_views",
            "12",
            "--pruning_max_remove_ratio",
            "0.03",
            "--pruning_mask_score_weight",
            "3.0",
            "--save_pruning_report",
            "--ip",
            args.ip,
            "--port",
            str(port),
            "--iterations",
            str(args.iterations),
            "--test_iterations",
            "-1",
            "--save_iterations",
        ]
    )
    command.extend(str(value) for value in save_iterations)
    return command


def build_render_command(
    args: argparse.Namespace,
    output_dir: Path,
    prepared_mask_dir: Path,
) -> list[str]:
    command = [
        str(PYTHON_BIN),
        "render.py",
        "--model_path",
        str(output_dir),
        "--iteration",
        str(args.iterations),
        "--resolution",
        str(args.resolution),
    ]
    add_mask_and_foreground_switches(command, prepared_mask_dir)
    command.extend(["--skip_train", "--skip_mesh"])
    return command


def build_full_metrics_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        str(PYTHON_BIN),
        str(FULL_METRICS_SCRIPT),
        "--model_path",
        str(output_dir),
        "--device",
        args.metrics_device,
    ]
    if args.no_lpips:
        command.append("--no_lpips")
    return command


def build_foreground_metrics_command(
    args: argparse.Namespace,
    source_path: Path,
    output_dir: Path,
    prepared_mask_dir: Path,
) -> list[str]:
    command = [
        str(PYTHON_BIN),
        str(FOREGROUND_METRICS_SCRIPT),
        "--model-path",
        str(output_dir),
        "--source-path",
        str(source_path),
        "--mask-dir",
        str(prepared_mask_dir),
        "--mask-pattern",
        "mask_{stem}.png",
        "--mask-threshold",
        "127",
        "--method",
        f"ours_{args.iterations}",
        "--device",
        args.metrics_device,
    ]
    if args.no_lpips:
        command.append("--no-lpips")
    return command


def build_mesh_command(
    args: argparse.Namespace,
    output_dir: Path,
    prepared_mask_dir: Path,
) -> list[str]:
    command = [
        str(PYTHON_BIN),
        "render.py",
        "--model_path",
        str(output_dir),
        "--iteration",
        str(args.iterations),
        "--resolution",
        str(args.resolution),
    ]
    add_mask_and_foreground_switches(command, prepared_mask_dir)
    command.extend(
        [
            "--skip_train",
            "--skip_test",
            "--meshing_mode",
            "standard",
            "--depth_trunc",
            str(args.mesh_depth_trunc),
            "--voxel_size",
            str(args.mesh_voxel_size),
            "--sdf_trunc",
            str(args.mesh_sdf_trunc),
            "--num_cluster",
            str(args.mesh_num_cluster),
        ]
    )
    return command


def expected_output(args: argparse.Namespace, output_dir: Path, step: str) -> Path:
    method = f"ours_{args.iterations}"
    if step == "train":
        return output_dir / f"point_cloud/iteration_{args.iterations}/point_cloud.ply"
    if step == "render":
        return output_dir / "test" / method / "renders"
    if step == "metrics":
        return output_dir / "results.json"
    if step == "foreground-metrics":
        return output_dir / "foreground_object_results.json"
    if step == "mesh":
        return output_dir / "train" / method / "fuse_post.ply"
    raise KeyError(step)


def is_step_done(args: argparse.Namespace, output_dir: Path, step: str) -> bool:
    target = expected_output(args, output_dir, step)
    if target.is_dir():
        return any(target.iterdir())
    return target.exists()


def command_map(
    args: argparse.Namespace,
    source_path: Path,
    output_dir: Path,
    prepared_mask_dir: Path,
    port: int,
) -> dict[str, list[str]]:
    return {
        "train": build_train_command(args, source_path, output_dir, prepared_mask_dir, port),
        "render": build_render_command(args, output_dir, prepared_mask_dir),
        "metrics": build_full_metrics_command(args, output_dir),
        "foreground-metrics": build_foreground_metrics_command(args, source_path, output_dir, prepared_mask_dir),
        "mesh": build_mesh_command(args, output_dir, prepared_mask_dir),
    }


def write_command_file(output_dir: Path, commands: dict[str, list[str]]) -> None:
    lines: list[str] = []
    for step, command in commands.items():
        lines.append(f"[{step}]")
        lines.append(shlex.join(command))
        lines.append("")
    (output_dir / "command.txt").write_text("\n".join(lines), encoding="utf-8")


def write_config_file(
    args: argparse.Namespace,
    output_dir: Path,
    sample: str,
    source: MaskSource,
    source_path: Path,
    raw_mask_dir: Path,
    prepared_mask_dir: Path,
    image_count: int,
    missing: list[str],
) -> None:
    config = {
        "stage": "S24-E4b-mask-source-2DGS-closed-loop",
        "sample": sample,
        "mask_source_id": source.source_id,
        "mask_source_label": source.label,
        "source_path": str(source_path),
        "raw_mask_dir": str(raw_mask_dir),
        "prepared_mask_dir": str(prepared_mask_dir),
        "repo_path": str(REPO_PATH),
        "python": str(PYTHON_BIN),
        "iterations": args.iterations,
        "resolution": args.resolution,
        "eval": args.eval,
        "image_count": image_count,
        "missing_mask_count": len(missing),
        "missing_masks": missing[:200],
        "fixed_2dgs_setting": {
            "mask_mode": "alpha",
            "init_pcd_mode": "foreground_track",
            "use_foreground_rgb_loss": True,
            "use_mask_loss": True,
            "use_bg_opacity_loss": True,
            "pruning_mode": "mask",
            "lambda_mask": 0.08,
            "lambda_bg": 0.02,
            "mask_loss_type": "l1_dice",
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_command(name: str, command: list[str], log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n")
        log.write(f"# cwd: {REPO_PATH}\n")
        log.write(f"# command: {shlex.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_PATH),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{name}] {line}", end="")
        return_code = process.wait()
    return {
        "name": name,
        "return_code": return_code,
        "elapsed_seconds": round(time.time() - start, 3),
        "log": str(log_path),
    }


def selected_steps(raw_steps: list[str]) -> set[str]:
    if "all" in raw_steps:
        return set(DEFAULT_STEPS)
    known = set(DEFAULT_STEPS)
    unknown = [step for step in raw_steps if step not in known]
    if unknown:
        raise ValueError(f"Unknown step(s): {unknown}. Known: {sorted(known)}")
    return set(raw_steps)


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("PYTHONUNBUFFERED", "1")
    python_paths = [
        str(REPO_PATH),
        str(REPO_PATH / "submodules/diff-surfel-rasterization"),
        str(REPO_PATH / "submodules/simple-knn"),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "mask_source",
        "mask_source_id",
        "source_path",
        "raw_mask_dir",
        "prepared_mask_dir",
        "model_path",
        "image_count",
        "matched_mask_count",
        "missing_mask_count",
        "status",
        "steps_requested",
        "execute",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (path.with_suffix(".json")).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    steps = selected_steps(args.steps)
    args.result_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    env = command_env(args)
    run_records: list[dict[str, Any]] = []

    job_index = 0
    for sample in args.samples:
        source_path = source_path_for(sample)
        ensure_scene(source_path)
        images = image_files(source_path)

        for source_id in args.mask_sources:
            source = MASK_SOURCES[source_id]
            raw_mask_dir, candidates = resolve_raw_mask_dir(args, source, sample)
            output_dir = output_dir_for(args, sample, source.source_id)
            prepared_mask_dir = output_dir / "prepared_masks"
            matched, missing = inspect_masks(raw_mask_dir, images)
            status = "ready" if not missing and raw_mask_dir.exists() else "missing_masks"
            notes = "" if raw_mask_dir.exists() else "raw_mask_dir_not_found"

            if args.execute:
                output_dir.mkdir(parents=True, exist_ok=True)
                if missing or not raw_mask_dir.exists():
                    status = "skipped_missing_masks"
                else:
                    matched, missing = rebuild_prepared_masks(raw_mask_dir, prepared_mask_dir, images)
                    status = "prepared"

            manifest_row = {
                "sample": sample,
                "mask_source": source.label,
                "mask_source_id": source.source_id,
                "source_path": str(source_path),
                "raw_mask_dir": str(raw_mask_dir),
                "prepared_mask_dir": str(prepared_mask_dir),
                "model_path": str(output_dir),
                "image_count": len(images),
                "matched_mask_count": matched,
                "missing_mask_count": len(missing),
                "status": status,
                "steps_requested": ",".join(sorted(steps)),
                "execute": str(args.execute),
                "notes": notes,
                "candidate_mask_dirs": ";".join(str(path) for path in candidates),
            }
            manifest_rows.append(manifest_row)

            commands = command_map(
                args,
                source_path,
                output_dir,
                prepared_mask_dir,
                args.port_base + job_index,
            )
            job_index += 1

            print("\n===", sample, source.label, "===")
            print("source:", source_path)
            print("raw masks:", raw_mask_dir)
            print("matched masks:", matched, "/", len(images))
            if missing:
                print("missing masks:", len(missing), "(first 10)", ", ".join(missing[:10]))
            for step in DEFAULT_STEPS:
                if step not in steps:
                    continue
                if args.only_missing and is_step_done(args, output_dir, step):
                    print(f"[skip existing] {step}: {expected_output(args, output_dir, step)}")
                    continue
                print(f"[{step}] {shlex.join(commands[step])}")

            if not args.execute or status == "skipped_missing_masks":
                continue

            write_config_file(
                args,
                output_dir,
                sample,
                source,
                source_path,
                raw_mask_dir,
                prepared_mask_dir,
                len(images),
                missing,
            )
            write_command_file(output_dir, commands)
            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            for step in DEFAULT_STEPS:
                if step not in steps:
                    continue
                if args.only_missing and is_step_done(args, output_dir, step):
                    continue
                record = run_command(
                    f"{sample}_{source.source_id}_{step}",
                    commands[step],
                    logs_dir / f"{step}.log",
                    env,
                )
                run_records.append(record)
                if record["return_code"] != 0 and not args.continue_on_error:
                    write_manifest(args.result_root / "run_manifest.csv", manifest_rows)
                    (args.result_root / "run_records.json").write_text(
                        json.dumps(run_records, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    return int(record["return_code"])

    write_manifest(args.result_root / "run_manifest.csv", manifest_rows)
    (args.result_root / "run_records.json").write_text(
        json.dumps(run_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest written to {args.result_root / 'run_manifest.csv'}")
    if not args.execute:
        print("Dry run only. Add --execute to launch training/evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
