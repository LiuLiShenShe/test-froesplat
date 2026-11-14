"""Explicit serial representative evaluation; never starts a 19-plant batch."""
from pathlib import Path
import argparse
import subprocess
import sys
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plants',nargs='+',choices=['KongQueZhuYu','ChangShouHua1','WangWenCao1','DouBanLv1'],
                   default=['ChangShouHua1','WangWenCao1','DouBanLv1'])
    a=p.parse_args()
    for plant in a.plants:
        folder=ROOT/'experiments/fast_ply_validation/v2'/plant
        scene=ROOT/'configs/scenes'/(plant+'.json')
        if plant=='KongQueZhuYu':scene=ROOT/'experiments/Historical_TwoStage_Cleaner/scene_configs/KongQueZhuYu.json'
        if not (folder/'cleanup_report.json').exists():
            spec=json.loads(scene.read_text())
            subprocess.run([sys.executable,str(ROOT/'fast_clean_gggs.py'),'--input',spec['model'],'--output',str(folder/'cleaned_gaussians.ply'),
                            '--config',str(ROOT/'configs/fast_ply.json')],cwd=ROOT,check=True)
        out=folder/'evaluation'
        if (out/'paired_quality.json').exists():
            print('Existing evaluation preserved:',plant,flush=True);continue
        command=[sys.executable,'-u','-m','fast_ply.evaluate','--scene',str(scene),'--output',str(out),
                 '--views','16' if plant=='KongQueZhuYu' else '8','--max-side','960']
        if plant!='KongQueZhuYu':command+=['--skip-ablations']
        print('Starting final paired evaluation:',plant,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
