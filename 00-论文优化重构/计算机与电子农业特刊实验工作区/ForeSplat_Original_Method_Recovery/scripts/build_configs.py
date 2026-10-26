#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot every configuration this recovery experiment actually ran.

Nothing here is hand-transcribed: each JSON is either a byte-for-byte copy of a
run's own `参数.json` (the file the production script writes before it starts),
the flag list imported straight out of the ablation runner, or parsed out of the
2DGS argument parser.  If a config in `configs/` disagrees with a run on disk,
the run wins -- regenerate instead of editing by hand.

Usage:  python scripts/build_configs.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
CONFIGS = RECOVERY / "configs"
GS = ROOT / "2d-gaussian-splatting-m2m3-floor40"

SNAPSHOT_RUNS = {
    "R6_RAP_FULL_real.json": RECOVERY / "runs/R6_RAP_FULL_real",
    "KongQueZhuYu_RAP_FULL_210.json": RECOVERY / "runs/KongQueZhuYu_RAP_FULL_210",
    "KongQueZhuYu_C_R1_baseline_downstream.json":
        RECOVERY / "runs/downstream/KongQueZhuYu_C_R1_baseline",
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dev_rungs() -> dict:
    mod = load_module(RECOVERY / "scripts/run_dev_ablation.py", "dev_ablation_cfg")
    rungs = {"R6_R5_GCFC": mod.MULTI + mod.GATE + ["--ranking_policy", "r1"]
             + mod.SPNP_POINTS + mod.REPAIR + ["--corrective_geometry_backend",
                                              "colmap_tracks_gaussian"]}
    for name, extra in mod.CONFIGS:
        rungs[name] = mod.COMMON + extra
    return {
        "dev_set": mod.SCENES,
        "excluded_frozen_eval_scene": "KongQueZhuYu",
        "candidate_pool": "produced once per scene by R6_R5_GCFC (real SAM3), "
                          "reused by every other rung via --reuse_existing_candidates",
        "gt_convention_primary": "potted_plant",
        "python": mod.PYTHON,
        "production_script": str(mod.PROD),
        "rungs": rungs,
    }


def m2m3_defaults() -> dict:
    """Defaults of every `--m2m3_go_*` flag, parsed out of the 2DGS argument block."""
    text = (GS / "arguments/__init__.py").read_text(encoding="utf-8")
    out = {}
    for match in re.finditer(
            r'self\.(m2m3_go_\w+)\s*=\s*([^#\n]+?)\s*(?:#\s*(.*))?$',
            text, flags=re.MULTILINE):
        name, raw, comment = match.group(1), match.group(2).strip(), match.group(3)
        value: object = raw
        try:
            value = json.loads(raw.replace("'", '"'))
        except (ValueError, json.JSONDecodeError):
            pass
        out[f"--{name.replace('_', '-')}"] = value
        if comment:
            out[f"# {name}"] = comment.strip()
    return out


def main() -> int:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    written = []

    for name, run_dir in SNAPSHOT_RUNS.items():
        src = run_dir / "参数.json"
        if not src.exists():
            print(f"[warn] missing {src}")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        data["_snapshot_of"] = str(src.relative_to(ROOT))
        (CONFIGS / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(name)

    record = RECOVERY / "runs/downstream/KongQueZhuYu_C_R1_baseline/run_record.json"
    if record.exists():
        data = json.loads(record.read_text(encoding="utf-8"))
        data["_snapshot_of"] = str(record.relative_to(ROOT))
        (CONFIGS / "downstream_recipe_actual.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append("downstream_recipe_actual.json")

    (CONFIGS / "dev_ablation_rungs.json").write_text(
        json.dumps(dev_rungs(), indent=2, ensure_ascii=False), encoding="utf-8")
    written.append("dev_ablation_rungs.json")

    (CONFIGS / "m2m3_go_flags.json").write_text(
        json.dumps(m2m3_defaults(), indent=2, ensure_ascii=False), encoding="utf-8")
    written.append("m2m3_go_flags.json")

    print(json.dumps({"git_commit": git("rev-parse", "HEAD"),
                      "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
