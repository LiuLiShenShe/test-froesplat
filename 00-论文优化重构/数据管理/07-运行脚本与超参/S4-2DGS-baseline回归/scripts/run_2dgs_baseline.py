#!/usr/bin/env python3
"""Run a traceable pure 2DGS baseline experiment from a JSON config."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_scene(source_path: Path) -> None:
    required = [
        source_path / "images",
        source_path / "sparse" / "0" / "cameras.bin",
        source_path / "sparse" / "0" / "images.bin",
        source_path / "sparse" / "0" / "points3D.bin",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required scene files:\n" + "\n".join(missing))


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def write_yaml_like(obj: Any, path: Path, indent: int = 0) -> None:
    lines: list[str] = []

    def emit(value: Any, level: int, key: str | None = None) -> None:
        prefix = " " * level
        if isinstance(value, dict):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                level += 2
                prefix = " " * level
            for k, v in value.items():
                emit(v, level, str(k))
        elif isinstance(value, list):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                level += 2
                prefix = " " * level
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    emit(item, level + 2)
                else:
                    lines.append(f"{prefix}- {yaml_scalar(item)}")
        else:
            if key is None:
                lines.append(f"{prefix}{yaml_scalar(value)}")
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(value)}")

    emit(obj, indent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_output_dir(config: dict[str, Any]) -> Path:
    output_root = Path(config["output_root"])
    sample = config["sample"]
    method_tag = config["method_tag"]
    output_dir = output_root / sample / method_tag
    if output_dir.exists():
        if config.get("append_timestamp_if_exists", False):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_root / sample / f"{method_tag}_{stamp}"
        elif not config.get("allow_existing_output", False):
            raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def add_common_model_args(command: list[str], config: dict[str, Any], output_dir: Path) -> list[str]:
    command.extend(["--source_path", config["source_path"]])
    command.extend(["--model_path", str(output_dir)])
    if config.get("eval", False):
        command.append("--eval")
    if "resolution" in config:
        command.extend(["--resolution", str(config["resolution"])])
    if config.get("quiet", False):
        command.append("--quiet")
    return command


def append_switches(command: list[str], switches: dict[str, Any], allowed: set[str]) -> None:
    for key, value in switches.items():
        if key not in allowed:
            continue
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.extend([f"--{key}", str(value)])


def build_train_command(config: dict[str, Any], output_dir: Path) -> list[str]:
    train = config["train"]
    cmd = [config["python"], "train.py"]
    add_common_model_args(cmd, config, output_dir)
    append_switches(
        cmd,
        config.get("module_switches", {}),
        {
            "view_quality_mode",
            "raw_gate_mode",
            "mask_gate_mode",
            "geo_gate_mode",
            "raw_gate_keep_ratio",
            "mask_gate_keep_ratio",
            "geo_gate_keep_ratio",
            "raw_gate_list",
            "mask_gate_list",
            "geo_gate_list",
            "mask_mode",
            "mask_dir",
            "mask_pattern",
            "mask_threshold",
            "init_pcd_mode",
            "init_pcd_min_observations",
            "init_pcd_foreground_threshold",
            "init_pcd_dilate_mask_px",
            "init_pcd_max_cameras",
            "init_pcd_chunk_size",
            "use_mask_loss",
            "use_bg_opacity_loss",
            "use_foreground_rgb_loss",
            "lambda_mask",
            "lambda_bg",
            "lambda_fg_rgb",
            "foreground_bg_rgb_weight",
            "foreground_rgb_crop_padding",
            "mask_loss_type",
            "mask_ignore_boundary_px",
            "mask_loss_start_iter",
            "mask_loss_warmup_iters",
            "view_weight_mode",
            "view_weight_list",
            "view_weight_min",
            "view_weight_max",
            "view_weight_default",
            "pruning_mode",
            "pruning_start_iter",
            "pruning_interval",
            "pruning_opacity_threshold",
            "pruning_brightness_threshold",
            "pruning_mask_threshold",
            "pruning_mask_max_views",
            "pruning_max_remove_ratio",
            "pruning_mask_score_weight",
            "save_pruning_report",
        },
    )
    if "ip" in config:
        cmd.extend(["--ip", str(config["ip"])])
    if "port" in config:
        cmd.extend(["--port", str(config["port"])])
    cmd.extend(["--iterations", str(train["iterations"])])
    if "test_iterations" in train:
        cmd.append("--test_iterations")
        cmd.extend(str(v) for v in train["test_iterations"])
    if "save_iterations" in train:
        cmd.append("--save_iterations")
        cmd.extend(str(v) for v in train["save_iterations"])
    if train.get("checkpoint_iterations"):
        cmd.append("--checkpoint_iterations")
        cmd.extend(str(v) for v in train["checkpoint_iterations"])
    return cmd


def build_render_command(config: dict[str, Any], output_dir: Path) -> list[str] | None:
    render = config.get("render", {})
    if not render.get("enabled", False):
        return None
    cmd = [config["python"], "render.py", "--model_path", str(output_dir)]
    if "iteration" in render:
        cmd.extend(["--iteration", str(render["iteration"])])
    if "resolution" in config:
        cmd.extend(["--resolution", str(config["resolution"])])
    render_switches = dict(config.get("module_switches", {}))
    render_switches.update(render.get("module_switch_overrides", {}))
    append_switches(
        cmd,
        render_switches,
        {
            "view_quality_mode",
            "raw_gate_mode",
            "mask_gate_mode",
            "geo_gate_mode",
            "raw_gate_list",
            "mask_gate_list",
            "geo_gate_list",
            "mask_mode",
            "mask_dir",
            "mask_pattern",
            "mask_threshold",
            "meshing_mode",
            "edge_truncation_scale",
            "boundary_shrink_ratio",
        },
    )
    if render.get("skip_train", False):
        cmd.append("--skip_train")
    if render.get("skip_test", False):
        cmd.append("--skip_test")
    if render.get("skip_mesh", False):
        cmd.append("--skip_mesh")
    if render.get("render_path", False):
        cmd.append("--render_path")
    for key in ("voxel_size", "depth_trunc", "sdf_trunc", "num_cluster", "mesh_res"):
        if key in render:
            cmd.extend([f"--{key}", str(render[key])])
    if render.get("unbounded", False):
        cmd.append("--unbounded")
    if config.get("quiet", False):
        cmd.append("--quiet")
    return cmd


def build_metrics_command(config: dict[str, Any], output_dir: Path) -> list[str] | None:
    metrics = config.get("metrics", {})
    if not metrics.get("enabled", False):
        return None
    script = metrics.get("script")
    if script:
        cmd = [config["python"], script, "--model_path", str(output_dir)]
        if metrics.get("no_lpips", False):
            cmd.append("--no_lpips")
        if "device" in metrics:
            cmd.extend(["--device", str(metrics["device"])])
        return cmd
    return [config["python"], "metrics.py", "--model_paths", str(output_dir)]


def stream_command(
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n")
        log.write(f"# cwd: {cwd}\n")
        log.write(f"# command: {shlex.join(command)}\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
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

    elapsed = round(time.time() - start, 3)
    return {
        "name": name,
        "command": command,
        "log": str(log_path),
        "return_code": return_code,
        "elapsed_seconds": elapsed,
    }


def write_run_files(config: dict[str, Any], output_dir: Path, commands: dict[str, list[str] | None]) -> None:
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_yaml_like(config, output_dir / "config.yaml")

    command_lines = []
    for name, command in commands.items():
        if command is not None:
            command_lines.append(f"[{name}]")
            command_lines.append(shlex.join(command))
            command_lines.append("")
    (output_dir / "command.txt").write_text("\n".join(command_lines), encoding="utf-8")

    module_switches = config.get("module_switches", {})
    uses_mask = module_switches.get("mask_mode", "none") != "none"
    uses_h_vqg = any(
        module_switches.get(key, "none") != "none"
        for key in ("view_quality_mode", "raw_gate_mode", "mask_gate_mode", "geo_gate_mode")
    )
    uses_view_weighting = module_switches.get("view_weight_mode", "none") != "none"
    uses_m3 = bool(module_switches.get("use_mask_loss", False)) or bool(module_switches.get("use_bg_opacity_loss", False))
    uses_m4 = module_switches.get("pruning_mode", "none") != "none"
    uses_m5 = module_switches.get("meshing_mode", "standard") != "standard"
    (output_dir / "baseline_guard.json").write_text(
        json.dumps(
            {
                "baseline_type": "pure_2dgs" if not any([uses_mask, uses_h_vqg, uses_view_weighting, uses_m3, uses_m4, uses_m5]) else "plant_aware_2dgs_variant",
                "uses_mask": uses_mask,
                "uses_fsam3": uses_mask,
                "uses_h_vqg": uses_h_vqg,
                "uses_view_weighting": uses_view_weighting,
                "uses_m3_mask_loss": uses_m3,
                "uses_m4_pruning": uses_m4,
                "uses_m5_edge_meshing": uses_m5,
                "module_switches": module_switches,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    repo_path = Path(config["repo_path"])
    source_path = Path(config["source_path"])
    ensure_scene(source_path)
    output_dir = build_output_dir(config)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = build_train_command(config, output_dir)
    render_cmd = build_render_command(config, output_dir)
    metrics_cmd = build_metrics_command(config, output_dir)
    commands = {"train": train_cmd, "render": render_cmd, "metrics": metrics_cmd}
    write_run_files(config, output_dir, commands)

    env = os.environ.copy()
    if "cuda_visible_devices" in config:
        env["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])
    env.setdefault("PYTHONUNBUFFERED", "1")
    python_paths = [
        str(repo_path),
        str(repo_path / "submodules" / "diff-surfel-rasterization"),
        str(repo_path / "submodules" / "simple-knn"),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    run_status: dict[str, Any] = {
        "sample": config["sample"],
        "method_tag": config["method_tag"],
        "output_dir": str(output_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
    }
    status_path = output_dir / "run_status.json"

    for name, command in commands.items():
        if command is None:
            continue
        result = stream_command(name, command, repo_path, env, logs_dir / f"{name}.log")
        run_status["steps"].append(result)
        run_status["updated_at"] = datetime.now().isoformat(timespec="seconds")
        status_path.write_text(json.dumps(run_status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if result["return_code"] != 0:
            run_status["status"] = "failed"
            status_path.write_text(json.dumps(run_status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return int(result["return_code"])

    run_status["status"] = "success"
    run_status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    status_path.write_text(json.dumps(run_status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRun finished: {output_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
