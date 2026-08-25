#!/usr/bin/env python3
"""Summarise numeric columns in a real manuscript source table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.table.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {args.table}")

    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            parsed = maybe_float(value)
            if parsed is not None:
                numeric.setdefault(key, []).append(parsed)

    summary = {
        "table": str(args.table),
        "rows": len(rows),
        "columns": list(rows[0].keys()),
        "numeric_summary": {
            key: {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for key, values in numeric.items()
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
