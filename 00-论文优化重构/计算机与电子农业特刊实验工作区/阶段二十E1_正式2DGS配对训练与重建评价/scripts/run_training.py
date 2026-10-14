from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from common import E0, MATRIX, PYTHON, ROOT, STAGE, config_for, now, sha256, write_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    assert (args.sample, args.group) in MATRIX
    run_id = f"{args.sample}_{args.group}"
    log = STAGE / f"logs/{run_id}_STDOUT_STDERR.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    assert not log.exists(), "Preserve every previous attempt"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "PYTHONDONTWRITEBYTECODE": "1"})
    command = [str(PYTHON), str(STAGE / "scripts/train_observed.py"), "--sample", args.sample,
               "--group", args.group]
    original = [str(PYTHON), str(E0 / "scripts/run_phase20e1.py"), "--sample", args.sample,
                "--group", args.group, "--execute-optimization"]
    launched = now()
    start = time.perf_counter()
    samples = []
    with log.open("xb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        print(f"START {run_id} PID={process.pid} LOG={log}", flush=True)
        audit_process = subprocess.Popen([str(PYTHON), str(STAGE / "scripts/resource_audit.py"),
                                          "--sample", args.sample, "--group", args.group, "--pid", str(process.pid)],
                                         cwd=ROOT, env=env)
        while process.poll() is None:
            result = subprocess.run(["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if result.returncode == 0:
                memory, utilization, temperature = result.stdout.strip().split(",")
                samples.append({"timestamp": now(), "elapsed_seconds": time.perf_counter() - start,
                                "gpu_memory_used_mib": float(memory), "utilization_pct": float(utilization),
                                "temperature_c": float(temperature), "training_pid": process.pid})
            time.sleep(2)
        audit_process.wait()
    runtime = {"sample": args.sample, "group": args.group, "start_time": launched, "end_time": now(),
               "process_wall_seconds": time.perf_counter() - start, "exit_code": process.returncode,
               "pid": process.pid, "observer_command": command, "unchanged_E0_execution": original,
               "environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")},
               "output_path": config_for(args.sample, args.group)["model"]["model_path"],
               "stdout_stderr_path": str(log), "stdout_stderr_sha256": sha256(log),
               "sampled_gpu_peak_mib": max((row["gpu_memory_used_mib"] for row in samples), default=None),
               "sampling_interval_seconds": 2}
    write_csv(STAGE / f"logs/{run_id}_GPU_SAMPLES.csv", samples)
    write_json(STAGE / f"reports/{run_id}_PROCESS.json", runtime)
    print(json.dumps(runtime, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
