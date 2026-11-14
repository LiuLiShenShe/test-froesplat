"""Serial three-scene resource validation; never launches a 19-plant batch."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-root',type=Path,required=True,help='new independent result directory')
    p.add_argument('--plants',nargs='+',choices=['KongQueZhuYu','ChangShouHua1','WangWenCao1'],
                   default=['KongQueZhuYu','ChangShouHua1','WangWenCao1'])
    a=p.parse_args()
    for plant in a.plants:
        spec=json.loads((ROOT/'configs/scenes'/(plant+'.json')).read_text())
        out=a.result_root/plant
        print('Starting',plant,flush=True)
        subprocess.run([sys.executable,str(ROOT/'fast_clean_gggs.py'),'--input',spec['model'],
                        '--output',str(out/'cleaned_gaussians.ply'),'--config',str(ROOT/'configs/fast_ply.json'),
                        '--memory-budget-gib','2.5' if plant=='WangWenCao1' else '8'],cwd=ROOT,check=True)

if __name__=='__main__':main()
