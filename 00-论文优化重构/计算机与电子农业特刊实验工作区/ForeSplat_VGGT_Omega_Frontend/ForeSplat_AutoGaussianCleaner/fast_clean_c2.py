#!/usr/bin/env python3
"""CPU-only frozen C2_raw, with one final PLY and a lightweight report."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.fast_ply.c2 import C2Config, cleanup_c2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--scene', type=Path, help='Read the existing scene model field')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--io-limit-mib', type=float, default=20.)
    p.add_argument('--memory-budget-gib', type=float, default=8.)
    p.add_argument('--historical-ids', type=Path)
    p.add_argument('--expected-retained', type=int)
    a = p.parse_args()
    source = a.input if a.input else Path(json.loads(a.scene.read_text())['model'])
    report = cleanup_c2(source, a.output, C2Config(a.io_limit_mib, a.memory_budget_gib),
                        a.historical_ids, a.expected_retained,
                        progress=lambda message: print(message, flush=True))
    print(json.dumps({k: report[k] for k in ['original_gaussians', 'remaining_gaussians', 'timing', 'io', 'resource']}, indent=2))


if __name__ == '__main__':
    main()
