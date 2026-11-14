#!/usr/bin/env python3
"""Read-only native GGGS input validation; never render or tune heldout plants."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from plyfile import PlyData
from scipy.special import expit
from ForeSplat_AutoGaussianCleaner.auto_clean_batch import model_ready
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import validate_native_input


def audit(job, spec):
    path = Path(spec['model']).resolve()
    record = dict(plant=job['plant'], group=job['group'], model=str(path),
                  status='pending_native_30K_input', gaussians='', field_count='',
                  input_sha256='', error='')
    if not model_ready(job, spec):
        return record
    try:
        if 'iteration_30000' not in path.parts:
            raise ValueError('Expected final 30K GGGS input')
        rows = PlyData.read(str(path), mmap='r')['vertex'].data
        validate_native_input(rows)
        names = rows.dtype.names
        if {k for k in names if k.startswith('f_dc_')} != {f'f_dc_{i}' for i in range(3)}:
            raise ValueError('Incomplete SH DC')
        if {k for k in names if k.startswith('f_rest_')} != {f'f_rest_{i}' for i in range(45)}:
            raise ValueError('This experiment requires complete native SH degree 3')
        if any(k.startswith('sg_color_') for k in names):
            raise ValueError('SG lobes cannot be exported equivalently to the SH-only viewer')
        for start in range(0, len(rows), 250000):
            block = rows[start:start+250000]
            q = np.column_stack([block[f'rot_{i}'] for i in range(4)]).astype(np.float64)
            if np.any(np.linalg.norm(q, axis=1) < 1e-10):
                raise ValueError('Invalid zero quaternion')
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                sigma = np.exp(np.column_stack([block[f'scale_{i}'] for i in range(3)]))
            if np.any(sigma <= 0) or not np.isfinite(sigma).all():
                raise ValueError('Encoded scales do not decode to finite positive native float32 scales')
            sigma = sigma.astype(np.float64)
            effective = np.hypot(sigma, block['filter_3D'].astype(np.float64)[:, None])
            alpha = expit(block['opacity'].astype(np.float64)) * (sigma/effective).prod(1)
            if not np.isfinite(effective).all() or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
                raise ValueError('Invalid native effective covariance/opacity encoding')
        record.update(status='validated_original_native_30K', gaussians=len(rows),
                      field_count=len(names), input_sha256=sha(path))
    except (OSError, ValueError, KeyError) as error:
        record.update(status='invalid_native_input', error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = [audit(job, json.loads(Path(job['scene']).read_text())) for job in manifest['plants']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output.with_suffix('.json'), dict(
        expected=len(rows), validated=sum(row['status']=='validated_original_native_30K' for row in rows),
        pending=sum(row['status']=='pending_native_30K_input' for row in rows),
        invalid=sum(row['status']=='invalid_native_input' for row in rows),
        checks='Final iteration, native 3D filter, all three log scales, quaternion, SH3/SG0, finite decoded float32 scales and effective filter-aware alpha; complete native input SHA256',
        scope='Read-only format and reconstruction integrity. No heldout rendering, image inspection, colour feature statistics, cleanup, thresholds or independent test metrics.',
        source_sha256=sha(Path(__file__)), results=rows))
    print(json.dumps({key:sum(row['status']==key for row in rows) for key in sorted({r['status'] for r in rows})}), flush=True)
    if any(row['status']=='invalid_native_input' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
