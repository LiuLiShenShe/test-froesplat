#!/usr/bin/env python3
"""Inspect table-level experiment configuration templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON summary path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = {
        "table": config.get("table"),
        "description": config.get("description"),
        "dataset": config.get("dataset"),
        "metrics": config.get("metrics") or config.get("agreement_metrics") or config.get("shared_settings", {}).get("foreground_metrics"),
        "variants_or_methods": config.get("variants") or config.get("methods") or config.get("traits"),
        "note": "This script validates the table configuration template. Exact paper-number reproduction requires the full curated dataset.",
    }
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote table summary to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
