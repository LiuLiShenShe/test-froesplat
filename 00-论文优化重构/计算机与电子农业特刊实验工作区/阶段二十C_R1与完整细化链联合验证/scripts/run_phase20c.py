#!/usr/bin/env python3
"""Run one frozen Phase 20C production configuration and preserve evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id(config: dict[str, object], smoke_stem: str | None) -> tuple[Path, str]:
    base = Path(str(config["output_dir"]))
    if smoke_stem:
        output = WORK / "runs/smoke" / str(config["group"]) / f"{config['dataset']}_{smoke_stem}"
        return output, f"{config['dataset']}_{config['group']}_{smoke_stem}"
    return base, f"{config['dataset']}_{config['group']}"


def prepare_smoke_input(config: dict[str, object], stem: str) -> Path:
    source_dir = Path(str(config["input_dir"]))
    source_images = sorted(source_dir.glob(f"{stem}.jpg"))
    if len(source_images) != 1:
        raise FileNotFoundError(f"Expected one smoke input named {stem}.jpg under {source_dir}")
    smoke_dir = WORK / "configs/smoke" / str(config["dataset"]) / str(config["group"]) / stem / "input"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    destination = smoke_dir / source_images[0].name
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source_images[0].resolve():
            raise RuntimeError(f"Smoke input collision: {destination}")
    else:
        destination.symlink_to(source_images[0].resolve())
    return smoke_dir


def write_hash_manifest(output: Path) -> None:
    rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"):
        rows.append({"path": str(path.relative_to(output)), "size": path.stat().st_size, "sha256": sha256(path)})
    (output / "artifact_hashes.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Easy21", "GT10"], required=True)
    parser.add_argument("--group", choices=["V0", "V1", "V2", "V3"], required=True)
    parser.add_argument("--smoke-stem", default=None)
    parser.add_argument("--allow-nonempty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = WORK / "configs" / f"{args.dataset}_{args.group}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output, run_name = run_id(config, args.smoke_stem)
    if output.exists() and any(output.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"Refusing to mix with existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in config["command"]]
    if args.smoke_stem:
        smoke_input = prepare_smoke_input(config, args.smoke_stem)
        command[command.index("--input_dir") + 1] = str(smoke_input)
        command[command.index("--output_dir") + 1] = str(output)
    command_text = " ".join(subprocess.list2cmdline([item]) for item in command)
    (output / "execution_command.txt").write_text(command_text + "\n", encoding="utf-8")
    environment = {
        "cwd": str(ROOT),
        "python": sys.executable,
        "python_version": sys.version,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "config": str(config_path.relative_to(ROOT)),
        "smoke_stem": args.smoke_stem,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv,noheader"], capture_output=True, text=True)
    environment["nvidia_smi"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    (output / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_path = output / "run.log"
    runtime_compat = Path(__file__).resolve().parent / "runtime_compat"
    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = str(runtime_compat) + os.pathsep + process_env.get("PYTHONPATH", "")
    environment["runtime_compat"] = str(runtime_compat)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {command_text}\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, env=process_env, stdout=log, stderr=subprocess.STDOUT, text=True)
    status = {"run_name": run_name, "returncode": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL"}
    (output / "runner_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result.returncode == 0:
        write_hash_manifest(output)
    print(json.dumps({**status, "output": str(output)}, ensure_ascii=False))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
