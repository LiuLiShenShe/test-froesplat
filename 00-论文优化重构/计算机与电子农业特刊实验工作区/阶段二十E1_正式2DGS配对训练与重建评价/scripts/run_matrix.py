from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from common import MATRIX, PYTHON, ROOT, SAMPLES, STAGE, now, write_json


def run(command, log_path=None):
    print("EXECUTE", " ".join(command), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("xb") as handle:
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-run")
    parser.add_argument("--stop-after", choices=SAMPLES)
    parser.add_argument("--start-at", choices=SAMPLES)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    status = {"started": now(), "status": "RUNNING", "training_failures": [], "evaluation_failures": []}
    status_path = STAGE / "reports/MATRIX_PROGRESS.json"
    if args.wait_for_run:
        path = STAGE / f"reports/{args.wait_for_run}_PROCESS.json"
        print("WAITING_FOR_ACTUAL_RUNNING_EXPERIMENT", args.wait_for_run, flush=True)
        while not path.exists():
            time.sleep(30)
    for sample in SAMPLES:
        if args.start_at and SAMPLES.index(sample) < SAMPLES.index(args.start_at):
            continue
        for matrix_sample, group in MATRIX:
            if matrix_sample != sample:
                continue
            run_id = f"{sample}_{group}"
            path = STAGE / f"reports/{run_id}_PROCESS.json"
            if path.exists():
                result = json.loads(path.read_text())["exit_code"]
            else:
                status["current"] = {"sample": sample, "group": group, "operation": "optimization"}
                write_json(status_path, status)
                result = run([str(PYTHON), str(STAGE / "scripts/run_training.py"), "--sample", sample, "--group", group])
            if result:
                status["training_failures"].append(run_id)
                status["status"] = "PAUSED_FOR_PIPELINE_REMEDIATION"
                write_json(status_path, status)
                raise SystemExit(1)
        for matrix_sample, group in MATRIX:
            if matrix_sample != sample:
                continue
            run_id = f"{sample}_{group}"
            if run_id in status["training_failures"]:
                continue
            status["current"] = {"sample": sample, "group": group, "operation": "checkpoint reload and shared-reference held-out evaluation"}
            write_json(status_path, status)
            validation = STAGE / f"reports/{run_id}_VALIDATION.json"
            if validation.exists():
                result = 0 if json.loads(validation.read_text())["status"] == "PASS" else 1
            else:
                result = run([str(PYTHON), str(STAGE / "scripts/evaluate.py"), "--sample", sample, "--group", group],
                             STAGE / f"logs/{run_id}_EVALUATION.txt")
            if result:
                status["evaluation_failures"].append(run_id)
            write_json(status_path, status)
        if sample == "WanNianQing1" and status["evaluation_failures"]:
            status["status"] = "PAUSED_FOR_PIPELINE_REMEDIATION"
            write_json(status_path, status)
            raise SystemExit(1)
        if not status["evaluation_failures"]:
            result = run([str(PYTHON), str(STAGE / "scripts/summarize.py"), "--sample", sample])
            if result == 0:
                result = run([str(PYTHON), str(STAGE / "scripts/figures.py"), "--sample", sample])
            if result:
                status["status"] = "PAUSED_FOR_FIGURE_PIPELINE_REMEDIATION"
                write_json(status_path, status)
                raise SystemExit(1)
        if args.stop_after == sample:
            status.update({"status": "SCENE_CLOSED_LOOP_READY_FOR_DIRECT_VISUAL_CHECK", "scene_completed": sample})
            write_json(status_path, status)
            return
    status.update({"finished": now(), "status": "PASS" if not status["training_failures"] and not status["evaluation_failures"] else "PARTIAL"})
    write_json(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
