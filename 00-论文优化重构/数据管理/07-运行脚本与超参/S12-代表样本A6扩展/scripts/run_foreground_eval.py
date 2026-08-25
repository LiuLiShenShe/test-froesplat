#!/usr/bin/env python3
"""Run foreground-object evaluation for S12 A6 configs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/fj/F2DMAS")
EVAL_SCRIPT = (
    ROOT
    / "00-论文优化重构"
    / "数据管理"
    / "07-运行脚本与超参"
    / "S11-plant-only导出"
    / "scripts"
    / "evaluate_foreground_object_metrics.py"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mask_threshold_to_u8(value: Any) -> int:
    threshold = float(value)
    if threshold <= 1.0:
        return int(round(threshold * 255))
    return int(round(threshold))


def model_path_for(config: dict[str, Any]) -> Path:
    output_root = Path(config["output_root"])
    sample = config["sample"]
    method_tag = config["method_tag"]
    direct = output_root / sample / method_tag
    if direct.exists():
        return direct
    candidates = sorted((output_root / sample).glob(f"{method_tag}_*"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No output directory found for {sample}/{method_tag}")


def run_eval(config_path: Path, no_lpips: bool) -> int:
    config = read_json(config_path)
    switches = config.get("module_switches", {})
    repo_path = Path(config["repo_path"])
    model_path = model_path_for(config)
    log_dir = model_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics_foreground_object.log"

    command = [
        config["python"],
        str(EVAL_SCRIPT),
        "--model-path",
        str(model_path),
        "--source-path",
        config["source_path"],
        "--mask-dir",
        switches["mask_dir"],
        "--mask-pattern",
        switches.get("mask_pattern", "mask_{stem}.png"),
        "--mask-threshold",
        str(mask_threshold_to_u8(switches.get("mask_threshold", 0.5))),
        "--method",
        "ours_30000",
        "--device",
        str(config.get("metrics", {}).get("device", "cuda:0")),
    ]
    if no_lpips:
        command.append("--no-lpips")

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

    with log_path.open("w", encoding="utf-8") as log:
        log.write("# foreground-object eval\n")
        log.write(f"# config: {config_path}\n")
        log.write(f"# cwd: {repo_path}\n")
        log.write(f"# command: {shlex.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(repo_path),
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
            print(line, end="")
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()

    for config_path in args.configs:
        rc = run_eval(config_path, args.no_lpips)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
