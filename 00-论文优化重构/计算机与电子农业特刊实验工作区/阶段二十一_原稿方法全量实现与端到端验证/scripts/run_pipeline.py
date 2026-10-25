#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — unified end-to-end ForeSplat runner with a run manifest.

One JSON job description drives the whole chain:

    fft   -> 02-FFT/blur_filter.py          (Eq. 1 frame filtering)
    rap   -> 生成RAP-FSAM3掩膜.py            (RAP-FSAM3 foreground priors)
    train -> 2d-gaussian-splatting-m2m3-floor40/train.py
             (foreground-object 2DGS + M2M3-GO)
    eval  -> train repo scripts/evaluate_foreground.py
             (PSNR_fg / SSIM_fg / leakage on the same masks)

Guarantees:
  * a failed or missing upstream stage marks every downstream stage BLOCKED —
    there is no silent fallback to a baseline mask or an untrained model;
  * the manifest records git commit + dirty state, config SHA256, interpreter,
    the exact command, wall time, log path, return code, and the SHA256 of the
    masks that training actually consumed;
  * every artifact is written under the job's output dir; nothing is
    overwritten unless --force is given.

Job schema (all keys optional except the stages you enable)::

    {
      "job_id": "wannianqing1_c_r1_m4",
      "stages": ["fft", "rap", "train", "eval"],
      "fft":   {"input_dir": "...", "output_dir": "...",
                "fft_formula": "paper", "blur_percentile": 25.0},
      "rap":   {"input_dir": "...", "output_dir": "...",
                "args": ["--prompt_list", "P1,P2,P3,P4,P5", ...]},
      "train": {"source_path": "...", "model_path": "...",
                "args": ["--mask_mode", "alpha", ...]},
      "eval":  {"mask_dir": "...", "output_json": "...",
                "args": ["--split", "test"]}
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/fj/F2DMAS")
PHASE = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十一_原稿方法全量实现与端到端验证"
FFT_SCRIPT = ROOT / "02-FFT/blur_filter.py"
RAP_DIR = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
RAP_SCRIPT = RAP_DIR / "生成RAP-FSAM3掩膜.py"
TRAIN_REPO = ROOT / "2d-gaussian-splatting-m2m3-floor40"
TRAIN_SCRIPT = TRAIN_REPO / "train.py"
EVAL_SCRIPT = TRAIN_REPO / "scripts/evaluate_foreground.py"
PYTHON = TRAIN_REPO / "venv/bin/python"

STAGE_ORDER = ["fft", "rap", "train", "eval"]


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


GIT_CACHE = Path("/tmp/phase21_git_state.json")
GIT_CACHE_TTL = 600
GIT_STATUS_TIMEOUT = int(os.environ.get("PHASE21_GIT_STATUS_TIMEOUT", "20"))


def _git_cache_read(commit: str) -> dict | None:
    try:
        payload = json.loads(GIT_CACHE.read_text())
    except Exception:
        return None
    if payload.get("commit") != commit:
        return None
    age = time.time() - float(payload.get("ts", 0))
    if age > 6 * 3600:
        return None
    return {"dirty": payload["dirty"], "age_sec": int(age)}


def _git_cache_write(commit: str, dirty: bool) -> None:
    try:
        GIT_CACHE.write_text(json.dumps({"commit": commit, "dirty": dirty,
                                         "ts": time.time()}))
    except Exception:
        pass


def git_state() -> dict:
    """Record the repository state without ever guessing.

    ``git status`` walks the whole worktree; on this host /data is a saturated
    mechanical disk and the scan has taken >20 min, which used to block a run
    before the first line of its log.  The commit/branch read only needs .git
    and stays cheap.  If the worktree scan times out, a recent verdict for the
    *same commit* is reused and labelled; if there is none, ``dirty`` is set to
    "unknown" rather than assuming a clean tree.
    """
    def run(args, timeout):
        try:
            done = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                  check=True, timeout=timeout)
            return True, done.stdout.strip()
        except Exception:
            return False, ""

    _, commit = run(["git", "rev-parse", "HEAD"], 30)
    _, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], 30)
    stale = _git_cache_read(commit) if commit else None
    if stale is not None and stale["age_sec"] <= GIT_CACHE_TTL:
        return {"commit": commit, "branch": branch, "dirty": stale["dirty"],
                "status_source": "cache({}s old)".format(stale["age_sec"])}
    ok, status = run(["git", "status", "--porcelain"], GIT_STATUS_TIMEOUT)
    state = {"commit": commit, "branch": branch}
    if ok:
        state["dirty"] = bool(status)
        state["status_source"] = "live"
        if commit:
            _git_cache_write(commit, state["dirty"])
        return state
    cached = _git_cache_read(commit) if commit else None
    if cached is not None:
        state["dirty"] = cached["dirty"]
        state["status_source"] = "cache({}s old)".format(cached["age_sec"])
    else:
        state["dirty"] = "unknown"
        state["status_source"] = "unavailable"
    return state


def mask_hashes(mask_dir: str, pattern: str) -> dict:
    root = Path(mask_dir)
    if not root.exists():
        return {"count": 0, "missing_dir": True, "combined_sha256": ""}
    masks = sorted(root.glob(pattern.replace("{stem}", "*")) or list(root.glob("*.png")))
    combined = hashlib.sha256()
    for path in masks:
        combined.update(path.name.encode("utf-8"))
        combined.update(path.read_bytes())
    return {"count": len(masks), "dir": str(root),
            "combined_sha256": combined.hexdigest(),
            "first": masks[0].name if masks else "", "last": masks[-1].name if masks else ""}


def build_command(stage: str, spec: dict, job_dir: Path) -> list[str]:
    if stage == "fft":
        cmd = [sys.executable, str(FFT_SCRIPT),
               "--input_dir", spec["input_dir"],
               "--output_dir", spec.get("output_dir", str(job_dir / "fft"))]
        for key in ("blur_percentile", "max_consecutive_drop", "fft_formula"):
            if key in spec:
                cmd += [f"--{key}", str(spec[key])]
    elif stage == "rap":
        cmd = [sys.executable, str(RAP_SCRIPT),
               "--input_dir", spec["input_dir"],
               "--output_dir", spec.get("output_dir", str(job_dir / "rap")),
               "--force"]
        cmd += [str(item) for item in spec.get("args", [])]
    elif stage == "train":
        cmd = [str(PYTHON), "-u", str(TRAIN_SCRIPT),
               "-s", spec["source_path"],
               "-m", spec.get("model_path", str(job_dir / "model")),
               "--iterations", str(spec.get("iterations", 30000))]
        cmd += [str(item) for item in spec.get("args", [])]
    elif stage == "eval":
        cmd = [str(PYTHON), "-u", str(EVAL_SCRIPT),
               "-m", spec["model_path"],
               "-s", spec["source_path"],
               "--iteration", str(spec.get("iteration", -1)),
               "--mask_dir", spec["mask_dir"],
               "--output_json", spec.get("output_json", str(job_dir / "eval" / "foreground.json"))]
        cmd += [str(item) for item in spec.get("args", [])]
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase21 end-to-end ForeSplat runner")
    ap.add_argument("--job", required=True, help="job JSON path")
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--output-root", default="",
                    help="override the run root (default: <phase>/runs/pipeline)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    job_path = Path(args.job).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_id = job.get("job_id") or job_path.stem
    run_root = Path(args.output_root) if args.output_root else PHASE / "runs" / "pipeline"
    job_dir = run_root / job_id
    log_dir = Path(args.log_dir) if args.log_dir else PHASE / "logs" / "pipeline" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / "run_manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing manifest: {manifest_path} "
                         "(pass --force to rerun)")

    stages = [stage for stage in STAGE_ORDER if stage in (job.get("stages") or [])]
    if not stages:
        raise SystemExit("job.stages is empty")

    manifest = {
        "phase": "21",
        "job_id": job_id,
        "job_config": str(job_path),
        "job_config_sha256": sha256_file(job_path),
        "git": git_state(),
        "python": str(PYTHON),
        "environment": {key: os.environ.get(key, "") for key in
                        ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")},
        "stages": [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "RUNNING",
    }

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    failed_upstream = ""
    exit_code = 0
    for stage in stages:
        spec = job.get(stage) or {}
        entry = {"stage": stage, "status": "PENDING"}
        manifest["stages"].append(entry)
        if failed_upstream:
            entry["status"] = "BLOCKED"
            entry["blocked_by"] = failed_upstream
            save()
            continue
        try:
            cmd = build_command(stage, spec, job_dir)
        except KeyError as exc:
            entry["status"] = "BLOCKED"
            entry["error"] = f"missing job field: {exc}"
            failed_upstream = stage
            save()
            continue
        entry["command"] = cmd
        if args.dry_run:
            entry["status"] = "DRY_RUN"
            save()
            continue
        log_path = log_dir / f"{stage}.log"
        entry["log"] = str(log_path)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, "command": cmd},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            result = subprocess.run(cmd, cwd=str(ROOT), stdout=handle,
                                    stderr=subprocess.STDOUT)
        entry["wall_sec"] = time.monotonic() - started
        entry["returncode"] = result.returncode
        entry["status"] = "OK" if result.returncode == 0 else "FAILED"
        if stage == "train":
            entry["mask_provenance"] = str(Path(spec.get("model_path", job_dir / "model"))
                                           / "mask_provenance.json")
        if stage == "eval":
            out = Path(spec.get("output_json", job_dir / "eval" / "foreground.json"))
            entry["metrics"] = str(out)
            entry["metrics_sha256"] = sha256_file(out)
        if result.returncode != 0:
            failed_upstream = stage
            exit_code = result.returncode
        save()

    # record the masks/priors that downstream stages actually consumed
    rap_spec = job.get("rap") or {}
    train_spec = job.get("train") or {}
    mask_dir = ""
    argv = train_spec.get("args", [])
    for idx, item in enumerate(argv):
        if item == "--mask_dir" and idx + 1 < len(argv):
            mask_dir = argv[idx + 1]
    if not mask_dir:
        mask_dir = rap_spec.get("output_dir", str(job_dir / "rap"))
        finals = Path(mask_dir) / "最终掩膜"
        if finals.exists():
            mask_dir = str(finals)
    manifest["consumed_masks"] = mask_hashes(mask_dir, "mask_*.png")
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["status"] = "FAILED" if exit_code else (
        "DRY_RUN" if args.dry_run else "OK")
    save()
    print(json.dumps({"job_id": job_id, "status": manifest["status"],
                      "manifest": str(manifest_path),
                      "stages": [{"stage": s["stage"], "status": s["status"]}
                                 for s in manifest["stages"]]}, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
