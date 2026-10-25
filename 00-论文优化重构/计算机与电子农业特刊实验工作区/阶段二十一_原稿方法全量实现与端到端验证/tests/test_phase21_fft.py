#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — FFT manuscript-formula parity tests.

Verifies the implementation of manuscript Eq. (1):

    Q_FFT(I) = || W_H .* log(1+|F(I)|) ||_1 / (|| log(1+|F(I)|) ||_1 + eps)

  T1 radial high-pass W_H is 0 at the centre and 1 in the corners (monotone)
  T2 a constant image has near-zero high-frequency saliency
  T3 a sharp texture has higher Q_FFT than its Gaussian-blurred version
  T4 paper-mode smart_filter uses the first-quartile threshold and keeps
     continuity
  T5 process_folder writes a real log with both legacy and paper scores
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

FFT_SCRIPT = Path("/data/fj/F2DMAS/02-FFT/blur_filter.py")


def load_fft_module():
    name = "blur_filter_phase21"
    spec = importlib.util.spec_from_file_location(name, FFT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bf():
    return load_fft_module()


def test_t1_radial_weight(bf):
    weight = bf.radial_high_pass_weight((101, 101), 0.10, 0.10)
    assert weight[50, 50] == pytest.approx(0.0, abs=1e-6)
    assert weight[0, 0] > 0.99
    # Inside the radial transition band the weight increases with radius.
    assert weight[40, 50] > weight[43, 50] > weight[45, 50] >= 0.0


def test_t2_constant_image(bf):
    constant = np.full((128, 128), 128, dtype=np.uint8)
    score = bf.fft_blur_score_paper(constant)
    assert 0.0 <= score < 0.05


def test_t3_sharp_beats_blurred(bf):
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    sharp = cv2.resize(base, (512, 512), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    q_sharp = bf.fft_blur_score_paper(sharp)
    q_blurred = bf.fft_blur_score_paper(blurred)
    assert q_sharp > q_blurred
    assert 0.0 <= q_blurred <= q_sharp <= 1.0


def test_t4_paper_threshold_and_continuity(bf):
    scores = [
        {"index": i, "file": f"{i}.png", "fft": 0.5, "fft_paper": q, "laplacian": 100.0}
        for i, q in enumerate([0.50, 0.10, 0.09, 0.08, 0.11, 0.12, 0.13, 0.14])
    ]
    keep, combined, threshold, blurry = bf.smart_filter(
        scores, blur_threshold_percentile=25.0, max_consecutive_drop=2,
        fft_formula="paper")
    assert len(keep) == len(set(keep))
    assert keep[0] == 0 and keep[-1] == len(scores) - 1
    assert threshold == pytest.approx(np.percentile(combined, 25.0))
    # no run of >2 consecutive drops
    dropped = [i for i in range(len(scores)) if i not in keep]
    for start in dropped:
        run = 0
        for i in range(start, len(scores)):
            if i in dropped:
                run += 1
            else:
                break
        assert run <= 2


def test_t5_process_folder_real_log(bf, tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    rng = np.random.default_rng(11)
    for i in range(6):
        if i % 2 == 0:
            image = cv2.GaussianBlur(
                rng.integers(0, 256, size=(64, 64), dtype=np.uint8), (15, 15), 0)
        else:
            image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(input_dir / f"{i:04d}.png"), image)
    log = bf.process_folder(str(input_dir), str(output_dir), blur_percentile=25.0,
                            max_consecutive_drop=2, copy_mode="copy",
                            fft_formula="paper")
    assert log is not None
    saved = json.loads((output_dir / "filter_log.json").read_text(encoding="utf-8"))
    assert saved["fft_formula"] == "paper"
    assert saved["total_frames"] == 6
    assert len(saved["per_frame"]) == 6
    assert all("fft_paper" in row for row in saved["per_frame"])
    assert saved["kept_frames"] == len(list(output_dir.glob("*.png")))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
