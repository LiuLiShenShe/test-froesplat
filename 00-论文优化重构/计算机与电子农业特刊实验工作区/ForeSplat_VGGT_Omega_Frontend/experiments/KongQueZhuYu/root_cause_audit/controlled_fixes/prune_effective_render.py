#!/usr/bin/env python3
"""Prepared isolated effective-opacity deletion control. Run only when scheduled.

Uses a historical checkpoint through an audit-owned symlink, removes only rows
with effective opacity<0.05 by default, records keep mask/indices/parameter
hashes, renders three common training views, and never trains or saves a PLY.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
from pathlib import Path
import sys

AUDIT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('E1','E5'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threshold',type=float,default=.05)
    parser.add_argument('--source-ids',type=int,nargs='+',default=[0,95,189])
    args=parser.parse_args()
    sys.path.insert(0,str(AUDIT/'scripts'))
    import render_checkpoint_views
    sys.argv=[str(Path(render_checkpoint_views.__file__)),'--experiment',args.experiment,
        '--output',str(args.output),'--effective-opacity-threshold',str(args.threshold),
        '--source-ids',*[str(i) for i in args.source_ids]]
    render_checkpoint_views.main()


if __name__=='__main__':main()
