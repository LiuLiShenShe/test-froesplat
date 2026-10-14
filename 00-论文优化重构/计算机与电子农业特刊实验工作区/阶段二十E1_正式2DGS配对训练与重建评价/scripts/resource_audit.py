from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

from common import STAGE, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--pid", required=True, type=int)
    args = parser.parse_args()
    path = STAGE / f"logs/{args.sample}_{args.group}_COMPETING_PROCESS_AUDIT.csv"
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "training_pid", "observed_pid", "process_memory_mib", "training_process", "GPU_UUID"])
        while Path(f"/proc/{args.pid}").exists():
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                uuid, pid, memory = [value.strip() for value in line.split(",")]
                if uuid != "GPU-6836f890-98b2-5891-07ef-675c25199c65":
                    continue
                writer.writerow([now(), args.pid, pid, memory, int(pid) == args.pid, uuid])
            handle.flush()
            time.sleep(10)
    print("Read-only competing-process audit completed", path)


if __name__ == "__main__":
    main()
