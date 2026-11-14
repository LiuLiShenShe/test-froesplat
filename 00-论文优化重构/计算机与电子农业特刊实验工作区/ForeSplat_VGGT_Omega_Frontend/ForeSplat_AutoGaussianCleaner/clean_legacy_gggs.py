#!/usr/bin/env python3
"""Independent faithful/normalized V2 or V3 pruning of an actual native GGGS PLY."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply,sha,write_json
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features,rules
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.artifacts import export


def main():
    p=argparse.ArgumentParser(description='V2/V3 rules on native GGGS; no reconstruction or quality certification')
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--variant',choices=['C1','C2'],required=True)
    p.add_argument('--thresholds',choices=['original','adaptive'],required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();source=a.input.resolve();out=a.output.resolve()
    if source.parent==out:raise ValueError('Output must be independent of the input checkpoint directory')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new output directory; original checkpoints and previous results are preserved')
    begin=time.perf_counter();rows=read_ply(source)
    required=['x','y','z','opacity','filter_3D']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]
    if not set(required).issubset(rows.dtype.names):raise ValueError('Native GGGS schema requires three log scales, quaternion, logit opacity, SH and filter_3D')
    if any(k.startswith('sg_sharpness_') for k in rows.dtype.names):raise ValueError('This verified viewer baking path requires sg_degree=0')
    if not all(np.isfinite(rows[k]).all() for k in rows.dtype.names):raise ValueError('Nonfinite native Gaussian parameters')
    original_hash=sha(source)
    removed,bits,parameters=rules(features(rows),2 if a.variant=='C1' else 3,a.thresholds=='adaptive')
    retained=np.flatnonzero(~removed);export(out,rows,rows[~removed],retained)
    np.savez(out/'deletion_reasons.npz',original_ids=np.arange(len(rows)),removed=removed,reason_bits=bits)
    write_json(out/'RULES_COMPLETE.json',dict(method=a.variant,thresholds=a.thresholds,input_ply=str(source),
        input_sha256=original_hash,native_sha256=sha(out/'cleaned_gaussians.ply'),original_gaussians=len(rows),
        cleaned_gaussians=len(retained),removed_gaussians=int(removed.sum()),seconds=time.perf_counter()-begin,
        parameters=parameters,outcome='uncertified_appearance_rule_pruning',
        provenance_requirement='Input must be a trained GGGS model; compatible PLY columns alone do not establish backend provenance',
        evaluation_required='Native multiview evaluation is still required; RULES_COMPLETE does not certify leaf preservation or cleanup success'))
    assert sha(source)==original_hash
    print(a.variant,a.thresholds,'removed',int(removed.sum()),'native',out/'cleaned_gaussians.ply')


if __name__=='__main__':main()
