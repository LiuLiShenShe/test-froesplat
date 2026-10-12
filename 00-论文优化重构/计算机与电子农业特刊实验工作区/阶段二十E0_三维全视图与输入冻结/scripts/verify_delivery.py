from __future__ import annotations

import json
import argparse
import subprocess
from pathlib import Path

import numpy as np

from phase20e_common import DATA, EXPECTED, GS, PROD, ROOT, SAMPLES, WORK, binary_mask, pixel_hash, read_csv, sha256, write_csv, write_json


def protect_sources():
    errors = []
    sources = read_csv(WORK / "reports/SOURCE_PROTECTION.csv")
    stats = []
    for index, row in enumerate(sources):
        path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append({"type": "SOURCE_CHANGED", "path": str(path)})
        if path.is_file():
            stat = path.stat()
            stats.append({"path": row["path"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns})
        if index % 200 == 0:
            print("PROTECT", index + 1, "/", len(sources), flush=True)
    write_json(WORK / "logs/SOURCE_PROTECTION_FINAL.json", {"status": "PASS" if not errors else "FAIL", "protected_files": len(sources), "errors": errors})
    write_csv(WORK / "reports/SOURCE_VERIFICATION_STAT.csv", stats)
    if errors:
        raise ValueError("Protected historical sources changed")
    return sources


def protect_artifacts():
    hashes, stats = [], []
    for index, path in enumerate(sorted(DATA.rglob("*"))):
        if path.is_file():
            stat = path.stat()
            hashes.append({"path": str(path), "bytes": stat.st_size, "sha256": sha256(path),
                           "symlink": path.is_symlink(), "resolved_path": str(path.resolve())})
            stats.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns})
        if index % 500 == 0:
            print("ARTIFACT", index, flush=True)
    write_csv(WORK / "reports/MANAGED_ARTIFACT_HASHES.csv", hashes)
    write_csv(WORK / "reports/MANAGED_ARTIFACT_VERIFICATION_STAT.csv", stats)
    return hashes


def run(sources_only=False, artifacts_only=False):
    if sources_only:
        protect_sources()
        return
    if artifacts_only:
        hashes = protect_artifacts()
        print("ARTIFACT_HASH_FREEZE", len(hashes), "PASS", flush=True)
        return
    errors = []
    if (WORK / "reports/SOURCE_VERIFICATION_STAT.csv").exists():
        for row in read_csv(WORK / "reports/SOURCE_VERIFICATION_STAT.csv"):
            stat = (ROOT / row["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (int(row["size"]), int(row["mtime_ns"]), int(row["ctime_ns"])):
                errors.append({"type": "SOURCE_MUTATED_AFTER_FULL_SHA256_CHECK", "path": row["path"]})
        sources = read_csv(WORK / "reports/SOURCE_PROTECTION.csv")
    else:
        sources = protect_sources()
    identity = read_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv")
    candidates = read_csv(WORK / "FULL_VIEW_CANDIDATE_MANIFEST.csv")
    selection = read_csv(WORK / "FULL_VIEW_R0_R1_SELECTION.csv")
    masks = read_csv(WORK / "FULL_VIEW_R0_R1_MASK_MANIFEST.csv")
    datasets = read_csv(WORK / "MATCHED_2DGS_DATASET_MANIFEST.csv")
    acceptance = json.loads((WORK / "reports/FINAL_ACCEPTANCE.json").read_text())
    for sample in SAMPLES:
        frames = {row["actual_raw_frame_id"] for row in identity if row["sample"] == sample}
        current = {row["frame_id"] for row in candidates if row["sample"] == sample}
        if acceptance["scenes"][sample]["ready"] and current != frames:
            errors.append({"type": "CANDIDATE_COVERAGE", "sample": sample})
        for row in [row for row in selection if row["sample"] == sample]:
            for policy in ("R0", "R1"):
                candidate_id = row[f"{policy}_candidate_id"]
                source = next(item for item in candidates if item["candidate_id"] == candidate_id)
                output = next(item for item in masks if item["frame_id"] == row["frame_id"] and item["policy"] == policy)
                if not np.array_equal(binary_mask(source["mask_path"]), binary_mask(output["mask_path"])):
                    errors.append({"type": "SELECTED_MASK_IDENTITY", "frame": row["frame_id"], "policy": policy})
    if (WORK / "reports/MANAGED_ARTIFACT_VERIFICATION_STAT.csv").exists():
        hashes = read_csv(WORK / "reports/MANAGED_ARTIFACT_HASHES.csv")
        for row in read_csv(WORK / "reports/MANAGED_ARTIFACT_VERIFICATION_STAT.csv"):
            path = Path(row["path"])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (int(row["size"]), int(row["mtime_ns"]), int(row["ctime_ns"])):
                errors.append({"type": "MANAGED_ARTIFACT_CHANGED_AFTER_SHA256", "path": str(path)})
        if {str(path) for path in DATA.rglob("*") if path.is_file()} != {row["path"] for row in hashes}:
            errors.append({"type": "MANAGED_FILE_COVERAGE_CHANGED"})
    else:
        hashes = protect_artifacts()
    training_files = [row["path"] for row in hashes if Path(row["path"]).suffix in (".pth", ".ckpt") or "iteration_" in row["path"]]
    if training_files or (DATA / "phase20e1_outputs").exists():
        errors.append({"type": "UNAUTHORIZED_TRAINING_ARTIFACT", "paths": training_files})
    differences = subprocess.check_output(["git", "diff", "--name-only", "-z", "f53bf8f3"], cwd=ROOT).decode().split("\0")
    differences = [path for path in differences if path]
    outside = [path for path in differences if not path.startswith(str(WORK.relative_to(ROOT)) + "/")]
    if outside:
        errors.append({"type": "OUT_OF_SCOPE_TRACKED_CHANGE", "paths": outside})
    source_code = {name: sha256(PROD / name) for name in EXPECTED}
    if source_code != EXPECTED:
        errors.append({"type": "FROZEN_PRODUCTION_SHA256_CHANGED"})
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "protected_sources": len(sources),
              "registered_views": len(identity), "candidate_instances": len(candidates), "selection_views": len(selection),
              "dataset_views": len(datasets), "managed_artifacts": len(hashes), "production_sha256": source_code,
              "formal_2dgs_training": False, "optimizer_steps": 0, "checkpoints_created": 0,
              "mesh_reconstruction": False, "outside_stage_changes": outside}
    write_json(WORK / "logs/DELIVERY_VERIFICATION.json", result)
    write_csv(WORK / "reports/STAGE_SOURCE_SHA256.csv", [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in sorted((WORK / "scripts").glob("*.py"))])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise ValueError("Delivery verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--artifacts-only", action="store_true")
    options = parser.parse_args()
    run(options.sources_only, options.artifacts_only)
