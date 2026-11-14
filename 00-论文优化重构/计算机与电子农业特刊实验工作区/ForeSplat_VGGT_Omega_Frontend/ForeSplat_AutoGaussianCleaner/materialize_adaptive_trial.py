#!/usr/bin/env python3
"""Reconstruct a fully native GGGS candidate from compact parameter evidence."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply,save_ply


def main():
    p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=json.loads(a.trial.read_text());rows=read_ply(meta['original_model'])
    for path in meta['previous_accepted_states']+[meta['parameters']]:
        with np.load(path) as z:rows[z['original_ids']]=z['new']
    ids=np.load(meta['retained_ids']);save_ply(a.output,rows[ids]);print(a.output.resolve(),len(ids))


if __name__=='__main__':main()
