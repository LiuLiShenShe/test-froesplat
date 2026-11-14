#!/usr/bin/env python3
"""Post-cleaning historical camera comparison; historical proxies never enter C5 selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.artifacts import image_pair
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs',type=Path,nargs='+',required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();reference=json.loads((args.baseline/'cleanup_metrics.json').read_text())
    protocol=json.loads(Path(reference['scene']['historical_free_protocol']).read_text())
    torch.set_num_threads(8)
    for out in args.outputs:
        out=out.resolve();marker=json.loads((out/'COMPLETE.json').read_text())
        if (out/'INVALIDATED.json').exists():raise ValueError('Invalidated experiment')
        report=json.loads((out/'cleanup_metrics.json').read_text())
        if marker['input_sha256']!=reference['input_sha256'] or sha(out/'cleaned_gaussians.ply')!=marker['native_sha256']:
            raise ValueError('Changed or different original/native model')
        base_spec={k:v for k,v in reference['scene'].items() if not k.startswith('historical_')}
        specification=marker.get('scene') or report.get('scene')
        if specification is None:raise ValueError('Completed model lacks its recorded scene')
        result_spec={k:v for k,v in specification.items() if not k.startswith('historical_')}
        if base_spec!=result_spec:raise ValueError('Historical evaluation must use the identical scene data')
        scene=Scene(specification,report['config']);model,_=scene.load_model(out/'cleaned_gaussians.ply');records=[]
        for number,view in enumerate(protocol['views']):
            camera=argparse.Namespace(uid=number,image_name=view['name'],image_width=view['width'],image_height=view['height'],
                FoVx=2*np.arctan(view['width']/(2*view['Fx'])),FoVy=2*np.arctan(view['height']/(2*view['Fy'])))
            for key in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
                setattr(camera,key,torch.tensor(view[key],dtype=torch.float32,device='cuda'))
            with torch.no_grad():a=arrays(scene.render(camera,model))
            image_pair(out/'visualizations/historical_free',view['name'],a)
            with np.load(Path(reference['scene']['historical_free_silhouettes'])/(view['name']+'.npz')) as z:
                outside=ndimage.distance_transform_edt(~z['reference_mesh_silhouette'])>5
            gray=a['rgb']+(1-a['alpha'][...,None])*protocol['background']
            records.append(dict(name=view['name'],black_pixels=int((outside&(gray@[.2126,.7152,.0722]<.12)).sum()),
                background_alpha=float(a['alpha'][outside].sum())))
        write_json(out/'visualizations/historical_free_metrics.json',dict(views=records,
            black_pixels=sum(v['black_pixels'] for v in records),native_sha256=marker['native_sha256'],
            protocol=reference['scene']['historical_free_protocol'],protocol_sha256=sha(reference['scene']['historical_free_protocol']),
            reference='Fixed historical P3 silhouette proxy, post-cleaning evaluation only. No historical cameras, manual ROIs or primitive IDs were supplied to C5 selection.',
            source_sha256=sha(Path(__file__).resolve())))
        print(out,records,flush=True)

if __name__=='__main__':main()
