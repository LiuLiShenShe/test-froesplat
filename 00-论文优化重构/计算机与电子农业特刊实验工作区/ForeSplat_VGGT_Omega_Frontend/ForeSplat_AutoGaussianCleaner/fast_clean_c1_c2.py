#!/usr/bin/env python3
"""CPU-only, shared C1_raw/C2_raw low-I/O cleanup of a single scene."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.fast_ply.c1_c2 import cleanup_pair,C2Config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse',type=Path);p.add_argument('--historical-ids',type=Path)
    p.add_argument('--memory-budget-gib',type=float,default=8.)
    a=p.parse_args();spec=json.loads(a.scene.read_text())
    result=cleanup_pair(spec['model'],a.output,C2Config(memory_budget_gib=a.memory_budget_gib),
        historical_ids=json.loads(a.historical_ids.read_text()) if a.historical_ids else None,
        reuse=json.loads(a.reuse.read_text()) if a.reuse else None,
        progress=lambda msg:print(msg,flush=True))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
