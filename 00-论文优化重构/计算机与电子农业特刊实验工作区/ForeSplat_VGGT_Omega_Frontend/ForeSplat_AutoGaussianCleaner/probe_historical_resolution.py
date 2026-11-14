#!/usr/bin/env python3
"""Hold camera IDs/models fixed to isolate raster resolution in FG protection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.historical.evidence import camera_indices
from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import HistoricalEvaluator,quality_reasons
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();root=a.experiment.resolve()
    r=json.loads((root/'execution.json').read_text())['reference'];torch.set_num_threads(8)
    history=Path(r['scene']['baseline_models']['R5']).parent.parent/'additional_rounds/round_05/gaussians.ply'
    dest=root/'comparison/resolution_control';results=[]
    for size in [960,1920]:
        config={**r['config'],'evaluation_max_side':size}
        scene=Scene(r['scene'],config);indices=camera_indices(scene,24)
        old,_=scene.load_model(scene.spec['model']);new,_=scene.load_model(history)
        ev=HistoricalEvaluator(scene,dest/str(size),original_model=old)
        initial=ev.evaluate(old,dest/str(size)/'initial',indices,include_free=False)
        candidate=ev.evaluate(new,dest/str(size)/'round5',indices,include_free=False)
        row=dict(resolution=size,camera_names=[scene.views[i]['name'] for i in indices],
            initial=initial,candidate=candidate,reasons=quality_reasons(initial,candidate,config))
        results.append(row)
        print(size,row['reasons'],'dark',initial['mean']['supported_dark_foreground_mse'],candidate['mean']['supported_dark_foreground_mse'],flush=True)
        ev.close();del ev,scene,old,new;torch.cuda.empty_cache()
    write_json(dest/'RESOLUTION_CONTROL.json',results)


if __name__=='__main__':main()
