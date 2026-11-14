#!/usr/bin/env python3
"""Fast, CPU-only, low-I/O native GGGS Gaussian cleaner."""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path

from dataclasses import asdict
from fast_ply import CleanupConfig, cleanup_ply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="native GGGS point_cloud.ply")
    parser.add_argument("--output", type=Path, default=None, help="cleaned_gaussians.ply")
    parser.add_argument("--report", type=Path, default=None, help="cleanup_report.json")
    parser.add_argument("--config", type=Path, help="optional CleanupConfig JSON; explicit CLI values override it")
    parser.add_argument("--io-limit-mib", type=float)
    parser.add_argument("--memory-budget-gib", type=float)
    parser.add_argument("--cell-scale", type=float)
    parser.add_argument("--min-local-support", type=int)
    parser.add_argument("--block-rows", type=int)
    parser.add_argument("--stage", choices=["attributes","appearance","isolation","joint"], default=None, help="Research ablation; use joint for normal cleanup")
    args = parser.parse_args()
    options = asdict(CleanupConfig())
    if args.config:
        options.update(json.loads(args.config.read_text()))
    for key in ["io_limit_mib", "memory_budget_gib", "cell_scale", "min_local_support", "block_rows", "stage"]:
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    config = CleanupConfig(**options)
    out = args.output or args.input.with_name("cleaned_gaussians.ply")
    report = args.report or out.with_name("cleanup_report.json")
    result = cleanup_ply(args.input, out, report,
                         config)
    print(json.dumps({k: result[k] for k in ("original_gaussians", "remaining_gaussians",
                                              "removed_gaussians", "removed_fraction", "timing", "io")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
