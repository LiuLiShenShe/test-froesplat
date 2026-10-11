from __future__ import annotations

import json
import shutil
from pathlib import Path

from fix_common import OLD, WORK, read_csv, relative, sha256, write_csv, write_json


FILES = ["参数.json", "提示词评分.csv", "提示词选择.csv", "运行日志.json", "失败汇总.json", "几何修正提示.csv"]


def main():
    inventory = []
    parameter_audit = []
    for status_path in sorted((WORK / "execution_records").glob("*_status.json")):
        status = json.loads(status_path.read_text())
        if status["returncode"] != 0:
            continue
        output = Path(status["output"])
        run_name = status_path.stem.removesuffix("_status")
        for name in FILES:
            source = output / name
            if not source.exists():
                continue
            destination = WORK / "small_run_outputs" / run_name / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            inventory.append({"run": run_name, "source": str(source), "archived_path": relative(destination),
                              "sha256": sha256(destination), "size": destination.stat().st_size})
        if status["smoke"]:
            continue
        group, dataset = status["group"], status["dataset"]
        actual = json.loads((output / "参数.json").read_text())
        historical = json.loads((OLD / "runs/V3/Easy21/参数.json").read_text())
        allowed = {"input_dir", "output_dir", "device", "candidate_source_dir", "candidate_metadata_dir", "candidate_metadata_strict",
                   "ranking_policy", "use_spnp_refinement", "spnp_backend", "use_residual_repair", "thin_repair_mode",
                   "use_corrective_geometry", "corrective_geometry_backend", "colmap_dir"}
        for key in sorted(set(actual) | set(historical)):
            old_value, value = historical.get(key), actual.get(key)
            expected_change = key in allowed
            unchanged = old_value == value
            parameter_audit.append({"dataset": dataset, "group": group, "parameter": key,
                                    "historical_V3": json.dumps(old_value, ensure_ascii=False),
                                    "actual": json.dumps(value, ensure_ascii=False),
                                    "factor_or_path_change_allowed": int(expected_change), "unchanged": int(unchanged),
                                    "status": "PASS" if expected_change or unchanged else "FAIL"})
            if not unchanged and not expected_change:
                raise ValueError(f"Unfrozen nonfactor parameter: {dataset}/{group}/{key}")
    write_csv(WORK / "SMALL_RUN_OUTPUT_INVENTORY.csv", inventory)
    write_csv(WORK / "FROZEN_PARAMETER_AUDIT.csv", parameter_audit)
    write_json(WORK / "ARCHIVE_AUDIT.json", {"small_files": len(inventory), "parameter_rows": len(parameter_audit),
               "unexpected_nonfactor_changes": sum(row["status"] == "FAIL" for row in parameter_audit)})


if __name__ == "__main__":
    main()
