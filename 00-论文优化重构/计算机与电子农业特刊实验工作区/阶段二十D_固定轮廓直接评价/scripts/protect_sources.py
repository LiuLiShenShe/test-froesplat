from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from direct_common import FIX, OLD, PHASEC, PROD, ROOT, WORK, frames, local_path, read_csv, relative, sha256, write_csv, write_json


def source_paths():
    paths = set()
    for folder in (OLD, PROD):
        paths.update(path for path in folder.rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts)
    for entry in read_csv(OLD / "SOURCE_HASHES.csv"):
        paths.add(local_path(entry["path"]))
    for frame in frames():
        paths.add(local_path(frame["rgb_path"]))
    for name in ("GT32_SOURCE_ARCHIVE.csv", "GT32_LOCAL_BACKUP_INDEX.csv"):
        for entry in read_csv(OLD / "completion" / name):
            for key in ("source_json", "canonical_json", "backup_path", "local_backup"):
                if entry.get(key):
                    paths.add(local_path(entry[key]))
    for folder in (FIX, PHASEC):
        tracked = subprocess.check_output(["git", "ls-files", "-z", "--", relative(folder)], cwd=ROOT).decode().split("\0")
        paths.update(ROOT / value for value in tracked if value)
    for folder in (ROOT / "2d-gaussian-splatting-main", ROOT / "2d-gaussian-splatting-m2m3-floor40"):
        tracked = subprocess.check_output(["git", "ls-files", "-z", "--", relative(folder)], cwd=ROOT).decode().split("\0")
        paths.update(ROOT / value for value in tracked if value and Path(value).suffix not in (".png", ".jpg", ".jpeg"))
    return sorted(paths)


def capture():
    destination = WORK / "SOURCE_PROTECTION_MANIFEST.csv"
    if destination.exists():
        raise FileExistsError("Source baseline already captured")
    rows = [{"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path),
             "is_symlink": int(path.is_symlink()), "resolved_path": relative(path.resolve())} for path in source_paths()]
    write_csv(destination, rows)
    write_json(WORK / "logs/REPOSITORY_START.json", {
        "cwd": str(ROOT), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True),
        "initial_user_files": [], "protected_files": len(rows), "original_gate_unchanged": True,
        "user_statement": "User says32 RGB/32 manual JSON and human review are complete; no archived signatures inferred",
    })
    print(f"SOURCE_BASELINE_CAPTURED: {len(rows)} files", flush=True)


def verify(output):
    rows = read_csv(WORK / "SOURCE_PROTECTION_MANIFEST.csv")
    failures = []
    for row in rows:
        path = local_path(row["path"])
        if not path.exists() or sha256(path) != row["sha256"] or relative(path.resolve()) != row["resolved_path"]:
            failures.append(row["path"])
    changes = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines()
    outside = [name for name in changes if not (ROOT / name).is_relative_to(WORK)]
    payload = {"protected_files": len(rows), "hash_failures": failures, "tracked_changes_outside_stage": outside,
               "status": "PASS" if not failures and not outside else "FAIL"}
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if failures or outside:
        raise RuntimeError("Source protection verification failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["capture", "verify"])
    parser.add_argument("--output", type=Path, default=WORK / "logs/SOURCE_PROTECTION_FINAL.json")
    args = parser.parse_args()
    capture() if args.action == "capture" else verify(args.output)


if __name__ == "__main__":
    main()
