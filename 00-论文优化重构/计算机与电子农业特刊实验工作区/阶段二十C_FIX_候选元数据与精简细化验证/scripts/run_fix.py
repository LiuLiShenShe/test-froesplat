from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from fix_common import OLD, PROD, ROOT, WORK, link, manifest, relative, sha256, write_csv, write_json


def run(dataset, group, smoke):
    folder = "V4_V5_CONFIGS" if group in ("V4", "V5") else "REPLAY_CONFIGS"
    config_path = WORK / folder / f"{dataset}_{group}.json"
    config = json.loads(config_path.read_text())
    command = list(config["command"])
    output = Path(config["output_dir"]) if not smoke else WORK / "runs/smoke" / group / f"{dataset}_{smoke}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        input_dir = WORK / "smoke_inputs" / dataset / group / smoke
        link(Path(config["input_dir"]) / f"{smoke}.jpg", input_dir / f"{smoke}.jpg")
        command[command.index("--input_dir") + 1] = str(input_dir)
        command[command.index("--output_dir") + 1] = str(output)
    if not smoke and group in ("V4", "V5"):
        for smoke_dataset, stem in [("Easy21", "CaoMei1_0000"), ("GT10", "DouBanLv2_0039")]:
            status_path = WORK / "runs/smoke" / group / f"{smoke_dataset}_{stem}" / "runner_status.json"
            if not status_path.exists() or not json.loads(status_path.read_text()).get("smoke_validated", False):
                raise RuntimeError(f"Two-frame smoke gate not passed: {status_path}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = str(OLD / "scripts/runtime_compat") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OMP_NUM_THREADS"] = "4"
    log_path = WORK / "logs" / f"{group}_{dataset}_{smoke or 'all'}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {"command": command, "cwd": str(ROOT), "config_sha256": sha256(config_path),
                "production_sha256": sha256(PROD / "生成RAP-FSAM3掩膜.py"),
                "metadata_loader_sha256": sha256(PROD / "candidate_metadata.py"),
                "gpu": subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv"], capture_output=True, text=True).stdout,
                "CUDA_VISIBLE_DEVICES": "0", "runtime_compat": environment["PYTHONPATH"],
                "config": relative(config_path), "smoke": smoke, "output": str(output)}
    write_json(output / "execution.json", snapshot)
    write_json(WORK / "execution_records" / f"{group}_{dataset}_{smoke or 'all'}.json", snapshot)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    status = {"dataset": dataset, "group": group, "smoke": smoke or "", "returncode": result.returncode,
              "wall_sec": time.monotonic() - started, "output": str(output), "log": relative(log_path)}
    if smoke and result.returncode == 0:
        import numpy as np
        from PIL import Image
        frame = next(value for value in manifest() if value["stem"] == smoke and value["dataset"] == dataset)
        selected_path = output / "选择后掩膜" / f"mask_{smoke}.png"
        final_path = output / "最终掩膜" / f"mask_{smoke}.png"
        selected = np.asarray(Image.open(selected_path).convert("L")) > 0
        final = np.asarray(Image.open(final_path).convert("L")) > 0
        gt = np.asarray(Image.open(ROOT / frame["gt_path"]).convert("L")) > 0
        f1 = float(2 * np.logical_and(final, gt).sum() / max(final.sum() + gt.sum(), 1))
        matched = [path.name for path in (ROOT / frame["candidate_alias_dir"]).glob(f"mask_{smoke}_*.png")
                   if np.array_equal(selected, np.asarray(Image.open(path).convert("L")) > 0)]
        expected = "01" if smoke == "DouBanLv2_0039" and group == "V4" else "00"
        valid = final.shape == gt.shape and final.any() and len(matched) == 1 and matched[0].endswith(f"_{expected}.png")
        if group == "V4":
            valid = valid and f1 >= 0.5
        status.update({"smoke_validated": bool(valid), "selected_candidate_mask": matched,
                       "smoke_final_f1": f1, "final_nonempty": bool(final.any())})
        if not valid:
            status["returncode"] = 1
    write_json(output / "runner_status.json", status)
    write_json(WORK / "execution_records" / f"{group}_{dataset}_{smoke or 'all'}_status.json", status)
    if result.returncode == 0:
        hashes = [{"path": relative(path), "size": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(output.rglob("*")) if path.is_file()]
        write_csv(WORK / "artifact_hashes" / f"{group}_{dataset}_{smoke or 'all'}.csv", hashes)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status["returncode"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Easy21", "GT10"], required=True)
    parser.add_argument("--group", choices=["V0", "V1", "V4", "V5"], required=True)
    parser.add_argument("--smoke", default=None)
    args = parser.parse_args()
    raise SystemExit(run(args.dataset, args.group, args.smoke))


if __name__ == "__main__":
    main()
