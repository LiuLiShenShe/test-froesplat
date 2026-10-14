from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

from common import E0, MATRIX, ROOT, STAGE, now, read_csv, sha256, write_csv, write_json


def main():
    configs = STAGE / "configs"
    records = []
    for sample, group in MATRIX:
        source = E0 / f"reports/{sample}_{group}_TRAIN_CONFIG.json"
        destination = configs / source.name
        shutil.copyfile(source, destination)
        assert sha256(destination) == sha256(source)
        records.append({"sample": sample, "group": group,
                        "config_path": str(destination.relative_to(ROOT)), "config_sha256": sha256(source),
                        "frozen_config_path": str(source.relative_to(ROOT)), "source_config_commit": "3e909f93"})
    write_csv(STAGE / "configs/CONFIG_INDEX.csv", records)
    manifests = ["MATCHED_2DGS_DATASET_MANIFEST.csv", "TRAIN_VALIDATION_SPLIT.csv", "CAMERA_PROVENANCE_MATRIX.csv",
                 "FULL_VIEW_IDENTITY_MANIFEST.csv", "reports/EVALUATION_REFERENCE_MANIFEST.csv",
                 "reports/PREDECLARED_SPLIT_COMPARISON.csv", "reports/MANAGED_ARTIFACT_HASHES.csv",
                 "reports/FROZEN_HARDWARE_IDENTITY.json", "reports/FINAL_ACCEPTANCE.json"]
    write_csv(STAGE / "configs/FROZEN_MANIFEST_INDEX.csv", [
        {"path": str((E0 / path).relative_to(ROOT)), "sha256": sha256(E0 / path),
         "bytes": (E0 / path).stat().st_size, "source_commit": "3e909f93"} for path in manifests])
    versions = {}
    for package in ("torch", "torchvision", "lpips", "tensorboard", "matplotlib", "numpy", "Pillow", "plyfile"):
        versions[package] = importlib.metadata.version(package)
    write_json(STAGE / "reports/ENVIRONMENT.json", {
        "timestamp": now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "versions": versions,
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "3e909f93"], text=True).strip(),
        "initial_branch": "main", "initial_worktree": "clean",
        "remote": subprocess.check_output(["git", "remote", "-v"], text=True),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
                                                "--format=csv"], text=True),
        "context7_library": "/pytorch/pytorch", "documentation_retrieved": True,
        "bocha_search": "HTTP_403_UNAVAILABLE", "RTK": "referenced file absent; not fabricated"})
    print("Frozen config copies and provenance indexed", len(records))


if __name__ == "__main__":
    main()
