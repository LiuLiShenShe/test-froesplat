#!/usr/bin/env python3
"""Launch or document a ForeSplat training run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Sequence root containing images, masks and cameras.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true", help="Write a run manifest instead of launching a 2DGS backend.")
    parser.add_argument("--backend-command", default="", help="Optional external 2DGS training command.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.demo and not args.backend_command:
        raise SystemExit(
            "Full training requires an external 2DGS backend. "
            "Pass --backend-command or use --demo to write a reproducibility manifest."
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "demo" if args.demo else "external_backend",
        "data_root": str(args.data),
        "config": str(args.config),
        "train_iterations": config.get("experiment", {}).get("train_iterations"),
        "foreground_initialisation": config.get("foreground_initialisation", {}),
        "losses": config.get("losses", {}),
        "view_quality_weighting": config.get("view_quality_weighting", {}),
        "m2m3_go": config.get("m2m3_go", {}),
        "mask_guided_pruning": config.get("mask_guided_pruning", {}),
        "backend_command": args.backend_command,
        "expected_outputs": {
            "checkpoint": "checkpoint.pth",
            "renders": "renders/",
            "metrics": "rendering_metrics.csv",
        },
        "note": "Demo mode records the training interface and does not optimise Gaussian primitives.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote run manifest to {args.output / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
