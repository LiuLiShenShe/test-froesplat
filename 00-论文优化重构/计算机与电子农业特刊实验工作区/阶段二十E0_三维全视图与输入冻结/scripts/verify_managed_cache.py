from __future__ import annotations

import json
from pathlib import Path

from phase20e_common import DATA, WORK, read_csv, sha256, write_json


def run():
    rows = read_csv(WORK / "reports/MANAGED_ARTIFACT_HASHES.csv")
    errors = []
    for index, row in enumerate(rows):
        path = Path(row["path"])
        if not path.is_file():
            errors.append({"path": str(path), "reason": "MISSING"})
        elif path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append({"path": str(path), "reason": "SIZE_OR_SHA256_CHANGED"})
        elif path.is_symlink() != (row["symlink"] == "True") or str(path.resolve()) != row["resolved_path"]:
            errors.append({"path": str(path), "reason": "LINK_IDENTITY_CHANGED"})
        if index % 500 == 0:
            print("VERIFY_MANAGED", index, "/", len(rows), flush=True)
    for sample in ("CaoMei2", "WanNianQing1", "HongZhang"):
        build = json.loads((WORK / f"reports/{sample}_BUILD.json").read_text())
        bundle = DATA / sample / "common/camera_bundle/sparse"
        for group in build["groups"]:
            path = DATA / sample / group / "sparse"
            if not path.is_symlink() or path.resolve() != bundle.resolve():
                errors.append({"path": str(path), "reason": "COMMON_CAMERA_DIRECTORY_SYMLINK_CHANGED"})
    result = {"status": "PASS" if not errors else "FAIL", "managed_files": len(rows), "errors": errors,
              "read_only_managed_data": True, "formal_training": False}
    write_json(WORK / "logs/MANAGED_CACHE_RESTORE_VERIFICATION.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise ValueError("Managed cache must be restored exactly before use")


if __name__ == "__main__":
    run()
