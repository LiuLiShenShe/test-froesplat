#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — foreground reconstruction metric tests.

  T1 perfect foreground render: high PSNR, SSIM ~ 1, zero leakage
  T2 background leakage raises leakage energy and outside non-black ratio
  T3 boundary-ignore band excludes boundary errors
  T4 PSNR matches the analytic value for a constant error
  T5 metric dict contains the manuscript criteria
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import pytest

TRAINER = Path("/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40")
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from utils import foreground_metrics as fm  # noqa: E402


def make_case(size=32):
    mask = torch.zeros(1, size, size)
    mask[:, 8:24, 8:24] = 1.0
    target = torch.zeros(3, size, size)
    target[:, 8:24, 8:24] = torch.tensor([0.2, 0.7, 0.3]).view(3, 1, 1)
    return mask, target


def test_t1_perfect_render():
    mask, target = make_case()
    metrics = fm.compute_foreground_metrics(target, target, mask, boundary_ignore=0)
    assert metrics["psnr_fg"] > 60 or math.isinf(metrics["psnr_fg"])
    assert metrics["ssim_fg"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["outside_mask_non_black_ratio"] == pytest.approx(0.0)
    assert metrics["leakage_energy_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_t2_background_leakage():
    mask, target = make_case()
    render = target.clone()
    render[:, :8, :] = 0.8  # bright background leak
    outside = fm.outside_mask_non_black_ratio(render, mask)
    leakage = fm.leakage_energy_ratio(render, target, mask)
    assert outside > 0.0
    assert leakage > 0.0
    clean = fm.outside_mask_non_black_ratio(target, mask)
    assert outside > clean


def test_t3_boundary_ignore():
    mask, target = make_case()
    render = target.clone()
    # corrupt only the 1-px boundary ring
    ring = mask * fm.boundary_band(mask, 1).float()
    render = torch.where(ring > 0.5, torch.ones_like(render), render)
    with_ring = fm.foreground_psnr(render, target, mask, boundary_ignore=0)
    without_ring = fm.foreground_psnr(render, target, mask, boundary_ignore=2)
    assert without_ring > with_ring


def test_t4_analytic_psnr():
    mask, target = make_case()
    error = 0.1
    render = (target + error).clamp(0, 1)
    psnr = fm.foreground_psnr(render, target, mask, boundary_ignore=0)
    assert psnr == pytest.approx(10.0 * math.log10(1.0 / (error ** 2)), rel=1e-4)


def test_t5_metric_keys():
    mask, target = make_case()
    metrics = fm.compute_foreground_metrics(target, target, mask)
    for key in ("psnr_fg", "ssim_fg", "outside_mask_non_black_ratio",
                "leakage_energy_ratio", "foreground_fraction"):
        assert key in metrics


def test_t6_lpips_backend_resolves_or_reports_absence():
    """LPIPS must either run on a real backend or report NaN -- never a guess.

    The vendored `lpipsPyTorch` copy in this workspace is corrupted on disk, so
    this also exercises the pip-`lpips` fallback path.
    """
    name = fm.lpips_backend_name()
    mask, target = make_case()
    value = fm.foreground_lpips(target, target, mask, boundary_ignore=0)
    if name is None:
        assert math.isnan(value)
    else:
        assert name in {"lpipsPyTorch"} or name.startswith("lpips-pip")
        # identical images must score ~0 under any LPIPS implementation
        assert value == pytest.approx(0.0, abs=5e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
