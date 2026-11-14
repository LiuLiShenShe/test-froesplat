#!/usr/bin/env python3
"""Invoke ForeSplat's automatic cleaning node on an already trained GGGS scene."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ForeSplat_AutoGaussianCleaner.integration import clean_trained_gggs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['C5', 'C5-B', 'C5-C', 'C5-D'], default='C5')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(clean_trained_gggs(args.scene, args.output, method=args.method,
        config=args.config, resume=args.resume), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
