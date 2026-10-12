from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from direct_common import PROD, ROOT, WORK, relative, sha256, write_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=WORK)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(WORK):
        raise ValueError("Snapshot output must stay within this new stage")
    files = sorted((WORK / "scripts").glob("*.py")) + sorted((WORK / "tests").glob("*.py"))
    files += [WORK / "CONTOUR_RASTERIZATION_SPEC.md", WORK / "IMPLEMENTATION_PLAN.md"]
    rows = [{"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
    write_csv(output / "CODE_SHA256_MANIFEST.csv", rows)
    distributions = ["numpy", "opencv-python", "pillow", "scipy", "pytest", "torch", "matplotlib", "requests"]
    versions = {name: importlib.metadata.version(name) for name in distributions}
    execution = json.loads((WORK / "logs/full_r1_execution.json").read_text())
    environment = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
        "repository_commit_before_delivery": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "production_device": "cpu",
        "production_code_sha256": {name: sha256(PROD / name) for name in ("生成RAP-FSAM3掩膜.py", "ranking_policy.py", "candidate_metadata.py")},
        "full_run_execution_log": relative(WORK / "logs/full_r1_execution.json"),
        "full_run_spec_sha256": execution["spec_sha256"],
        "new_sam_inference": False,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "not set; deterministic explicit orders and tie rules"),
        "credentials": "No credentials captured; optional external vision reads AGNES_API_KEY only",
    }
    write_json(output / "logs/RUNTIME_ENVIRONMENT.json", environment)
    print(f"Archived {len(rows)} delivered source hashes and actual runtime versions", flush=True)


if __name__ == "__main__":
    main()
