from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORK = Path(__file__).resolve().parents[1]
GS = ROOT / "2d-gaussian-splatting-main"
sys.dont_write_bytecode = True
sys.path.insert(0, str(GS / "submodules/simple-knn"))
sys.path.insert(0, str(GS / "submodules/diff-surfel-rasterization"))
sys.path.insert(0, str(GS))


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Phase20E1 only; no optimization without explicit execution flag")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--execute-optimization", action="store_true")
    args = parser.parse_args()
    path = WORK / f"reports/{args.sample}_{args.group}_TRAIN_CONFIG.json"
    config = json.loads(path.read_text())
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if not args.execute_optimization:
        print("DRY INSPECTION ONLY; NO OPTIMIZER")
        return
    preflight = json.loads((WORK / f"reports/{args.sample}_PREFLIGHT.json").read_text())
    if preflight["status"] != "PASS" or os.environ.get("CUDA_VISIBLE_DEVICES") != config["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("Frozen loader/GPU acceptance gate not satisfied")
    hardware = json.loads((WORK / "reports/FROZEN_HARDWARE_IDENTITY.json").read_text())
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", config["CUDA_VISIBLE_DEVICES"], "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if gpu_uuid != hardware["training_gpu_uuid"]:
        raise ValueError("Physical training GPU identity changed")
    gate = json.loads((WORK / "reports/FINAL_ACCEPTANCE.json").read_text())
    verification = json.loads((WORK / "logs/DELIVERY_VERIFICATION.json").read_text())
    if verification["status"] != "PASS" or verification["optimizer_steps"] != 0:
        raise ValueError("Phase20E0 delivery validation not complete")
    if not gate["scenes"].get(args.sample, {}).get("ready", False):
        raise ValueError("Scene not Phase20E1 ready")
    if args.group not in gate["scenes"][args.sample]["distinct_training_groups"]:
        raise ValueError("Duplicate identical treatment is intentionally excluded")
    config_hashes = json.loads((WORK / "reports/FROZEN_TRAIN_CONFIG_HASHES.json").read_text())
    if hash_file(path) != config_hashes[str(path.relative_to(ROOT))]:
        raise ValueError("Frozen train configuration changed")
    with (WORK / "MATCHED_2DGS_DATASET_MANIFEST.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["sample"] == args.sample and row["group"] == args.group]
    if len(rows) != gate["scenes"][args.sample]["registered"]:
        raise ValueError("Frozen dataset manifest coverage changed")
    for row in rows:
        for field, hash_field in [("loader_image_path", "image_sha256"), ("mask_path", "mask_sha256"), ("common_rgb_path", "common_rgb_sha256")]:
            if hash_file(Path(row[field])) != row[hash_field]:
                raise ValueError(f"Dataset artifact changed: {row[field]}")
    with (WORK / "CAMERA_PROVENANCE_MATRIX.csv").open(newline="", encoding="utf-8-sig") as handle:
        camera_rows = [row for row in csv.DictReader(handle) if row["sample"] == args.sample]
    for row in camera_rows:
        for filename, field in [("cameras.bin", "cameras_sha256"), ("images.bin", "images_sha256"), ("points3D.bin", "points_sha256")]:
            if hash_file(Path(rows[0]["sparse_bundle"]) / filename) != row[field]:
                raise ValueError("Common camera bundle changed")
    initialization_path = Path(rows[0]["sparse_bundle"]) / "points3D.ply"
    with (WORK / "reports/MANAGED_ARTIFACT_HASHES.csv").open(newline="", encoding="utf-8-sig") as handle:
        initialization_rows = [row for row in csv.DictReader(handle) if row["path"] == str(initialization_path)]
    if len(initialization_rows) != 1 or hash_file(initialization_path) != initialization_rows[0]["sha256"]:
        raise ValueError("Consumed sparse initialization PLY changed")
    with (WORK / "reports/SOURCE_PROTECTION.csv").open(newline="", encoding="utf-8-sig") as handle:
        code_rows = [row for row in csv.DictReader(handle) if row["kind"] == "PRODUCTION_CODE"]
    for row in code_rows:
        if hash_file(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Production training source changed: {row['path']}")
    dataset = argparse.Namespace(**config["model"])
    optimization = argparse.Namespace(**config["optimization"])
    pipeline = argparse.Namespace(**config["pipeline"])
    output = Path(dataset.model_path)
    if output.exists():
        raise ValueError(f"Refusing to overwrite training output: {output}")
    from utils.general_utils import safe_state
    from train import training
    safe_state(False)
    training(dataset, optimization, pipeline, [7000, 30000], [7000, 30000], [], None)


if __name__ == "__main__":
    main()
