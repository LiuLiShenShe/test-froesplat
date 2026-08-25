#!/usr/bin/env python3
"""Write a command manifest for external foreground mesh extraction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default_foresplat.yaml"))
    parser.add_argument("--backend-command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "backend_command": args.backend_command,
        "mesh_extraction": config.get("mesh_extraction"),
        "note": "Use this manifest with the full external mesh extraction implementation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote mesh extraction manifest to {args.output}")


if __name__ == "__main__":
    main()
