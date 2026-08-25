#!/usr/bin/env python3
"""Write a command manifest for launching an external ForeSplat/2DGS backend."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default_foresplat.yaml"))
    parser.add_argument("--backend-command", required=True, help="External backend command to run in your full environment.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "config": str(args.config),
        "backend_command": args.backend_command,
        "train_iterations": config.get("experiment", {}).get("train_iterations"),
        "losses": config.get("losses"),
        "foreground_initialisation": config.get("foreground_initialisation"),
        "note": "This repository does not bundle the external 2DGS backend.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote training manifest to {args.output}")


if __name__ == "__main__":
    main()
