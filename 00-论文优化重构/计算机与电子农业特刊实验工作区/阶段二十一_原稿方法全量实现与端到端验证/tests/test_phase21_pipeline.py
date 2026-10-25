#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — unified runner contract tests (no GPU, no training).

  T1  dry-run builds the exact command per stage and writes no artifacts
  T2  a missing job field BLOCKS only the stages downstream of it
  T3  the manifest records git commit, config SHA256 and per-stage status
  T4  the manifest fingerprints the masks training would consume
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import tempfile

import pytest

PHASE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/"
             "阶段二十一_原稿方法全量实现与端到端验证")
RUNNER = PHASE / "scripts/run_pipeline.py"


def write_job(tmp_path, job):
    path = tmp_path / f"{job['job_id']}.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def run_runner(job_path, *extra, output_root=None):
    cmd = [sys.executable, str(RUNNER), "--job", str(job_path), *extra]
    if output_root is not None:
        cmd += ["--output-root", str(output_root)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_t1_dry_run_commands():
    job = {
        "job_id": "unit_dry_run",
        "stages": ["fft", "rap", "train", "eval"],
        "fft": {"input_dir": "/in", "output_dir": "/out", "fft_formula": "paper"},
        "rap": {"input_dir": "/in", "output_dir": "/out",
                "args": ["--prompt_list", "P1,P2"]},
        "train": {"source_path": "/src", "model_path": "/model", "iterations": 123,
                  "args": ["--mask_dir", "/masks"]},
        "eval": {"model_path": "/model", "source_path": "/src", "mask_dir": "/masks"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_job(Path(tmp), job)
        result = run_runner(path, "--dry-run", output_root=Path(tmp) / "runs")
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert [stage["status"] for stage in out["stages"]] == ["DRY_RUN"] * 4
        manifest = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
        commands = {entry["stage"]: entry["command"] for entry in manifest["stages"]}
        assert any(arg.endswith("blur_filter.py") for arg in commands["fft"])
        assert "--fft_formula" in commands["fft"] and "paper" in commands["fft"]
        assert any(arg.endswith("生成RAP-FSAM3掩膜.py") for arg in commands["rap"])
        assert "--prompt_list" in commands["rap"]
        assert commands["train"][commands["train"].index("--iterations") + 1] == "123"
        assert commands["train"][commands["train"].index("--mask_dir") + 1] == "/masks"
        assert any(arg.endswith("evaluate_foreground.py") for arg in commands["eval"])


def test_t2_missing_field_blocks_downstream():
    job = {
        "job_id": "unit_blocked",
        "stages": ["rap", "train", "eval"],
        "rap": {"input_dir": "/in", "output_dir": "/out"},
        # train missing source_path -> BLOCKED, eval must inherit the block
        "train": {"model_path": "/model"},
        "eval": {"model_path": "/model", "source_path": "/src", "mask_dir": "/masks"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_job(Path(tmp), job)
        result = run_runner(path, "--dry-run", output_root=Path(tmp) / "runs")
        out = json.loads(result.stdout)
        statuses = {stage["stage"]: stage["status"] for stage in out["stages"]}
        assert statuses["train"] == "BLOCKED"
        assert statuses["eval"] == "BLOCKED"


def test_t3_manifest_provenance():
    job = {"job_id": "unit_manifest", "stages": ["rap"],
           "rap": {"input_dir": "/in", "output_dir": "/out"}}
    with tempfile.TemporaryDirectory() as tmp:
        path = write_job(Path(tmp), job)
        result = run_runner(path, "--dry-run", output_root=Path(tmp) / "runs")
        out = json.loads(result.stdout)
        manifest = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
        assert len(manifest["job_config_sha256"]) == 64
        assert "commit" in manifest["git"] and "dirty" in manifest["git"]
        assert manifest["status"] == "DRY_RUN"
        assert manifest["stages"][0]["stage"] == "rap"


def test_t4_mask_fingerprint(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "mask_a.png").write_bytes(b"aa")
    (mask_dir / "mask_b.png").write_bytes(b"bb")
    job = {"job_id": "unit_masks", "stages": ["train"],
           "train": {"source_path": "/src", "model_path": str(tmp_path / "model"),
                     "args": ["--mask_dir", str(mask_dir)]}}
    path = write_job(tmp_path, job)
    result = run_runner(path, "--dry-run", output_root=tmp_path / "runs")
    out = json.loads(result.stdout)
    manifest = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
    assert manifest["consumed_masks"]["count"] == 2
    assert len(manifest["consumed_masks"]["combined_sha256"]) == 64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
