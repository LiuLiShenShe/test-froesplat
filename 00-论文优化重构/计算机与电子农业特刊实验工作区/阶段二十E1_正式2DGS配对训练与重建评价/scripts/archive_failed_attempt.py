from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import DATA, STAGE, config_for, now, sha256, write_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    run_id = f"{args.sample}_{args.group}"
    runtime_path = STAGE / f"reports/{run_id}_RUNTIME.json"
    process_path = STAGE / f"reports/{run_id}_PROCESS.json"
    runtime = json.loads(runtime_path.read_text())
    process = json.loads(process_path.read_text())
    assert runtime["exit_code"] != 0 and process["exit_code"] != 0
    assert runtime["observed_iteration"] < 30000
    output = Path(config_for(args.sample, args.group)["model"]["model_path"])
    external = DATA / f"phase20e1_failed_attempts/{run_id}_attempt_01"
    archived = STAGE / f"logs/failed_attempts/{run_id}_attempt_01"
    assert not external.exists() and not archived.exists()
    external.parent.mkdir(parents=True, exist_ok=True)
    archived.mkdir(parents=True)
    if output.exists():
        shutil.move(str(output), str(external))
    records = []
    for folder in ("reports", "logs", "metrics"):
        for path in sorted((STAGE / folder).glob(f"{run_id}_*")):
            if not path.is_file():
                continue
            digest = sha256(path)
            destination = archived / path.name
            shutil.move(str(path), str(destination))
            assert sha256(destination) == digest
            records.append({"original": str(path), "preserved": str(destination), "sha256": digest,
                            "bytes": destination.stat().st_size})
    write_csv(archived / "PRESERVED_FILES.csv", records)
    write_json(archived / "FAILED_ATTEMPT_INDEX.json", {
        "timestamp": now(), "sample": args.sample, "group": args.group, "attempt": 1,
        "reason": runtime["exception"], "last_iteration": runtime["observed_iteration"],
        "failed_wall_seconds": runtime["process_wall_seconds"], "external_preserved_output": str(external),
        "original_output": str(output), "retry": "FULL_SAME_CONFIG_SEED_0; not checkpoint resume",
        "excluded_from_complete_model_averages": True})
    print("Full failure preserved; same-config output path released", run_id)


if __name__ == "__main__":
    main()
