#!/usr/bin/env python3
"""Automatic evaluation crop from the historical worst thin-coverage camera."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy import ndimage
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--c5-output',type=Path)
    args=parser.parse_args();p=Path(__file__).resolve().parent/'experiments/GGGS_Cleaner_Comparison'
    view='crop_0035';d=p/'baselines/C3_pre_marker/quality_inspection'
    a=np.asarray(Image.open(d/f'{view}_3.png').convert('RGB'));lost=(a==[255,0,0]).all(2)
    labels,_=ndimage.label(lost);sizes=np.bincount(labels.ravel())[1:];largest=labels==sizes.argmax()+1
    ys,xs=np.nonzero(largest);pad=round(.25*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1))+8
    box=(max(0,int(xs.min()-pad)),max(0,int(ys.min()-pad)),min(a.shape[1],int(xs.max()+pad+1)),min(a.shape[0],int(ys.max()+pad+1)))
    paths=[('Target RGB',d/f'{view}_0.png'),('C0 GGGS',d/f'{view}_1.png'),
        ('C1 adaptive',p/'baselines/C1_adaptive/quality_inspection'/f'{view}_2.png'),
        ('C2 adaptive',p/'baselines/C2_adaptive/quality_inspection'/f'{view}_2.png'),('C3 pre-marker',d/f'{view}_2.png')]
    if args.c5_output:
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='1'
        import torch
        from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
        from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
        output=args.c5_output.resolve();marker=json.loads((output/'COMPLETE.json').read_text())
        if marker['native_sha256']!=sha(output/'cleaned_gaussians.ply'):raise ValueError('Changed C5 model')
        if marker['input_sha256']!=json.loads((p/'baselines/C0/COMPLETE.json').read_text())['input_sha256']:raise ValueError('Another GGGS input')
        scene=Scene(marker['scene'],marker['config']);index=next(i for i,v in enumerate(scene.views) if v['name']==view)
        camera,_,mask,*_=scene.get_view(index);model,_=scene.load_model(output/'cleaned_gaussians.ply')
        with torch.no_grad():render=arrays(scene.render(camera,model))
        ys,xs=np.nonzero(mask);pad=round(.06*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1));pad=max(8,pad)
        mask_box=(max(0,int(xs.min()-pad)),max(0,int(ys.min()-pad)),min(mask.shape[1],int(xs.max()+pad+1)),min(mask.shape[0],int(ys.max()+pad+1)))
        path=output/'quality_inspection'/f'{view}_leaf_case_native_white.png';path.parent.mkdir(exist_ok=True)
        Image.fromarray(np.uint8(np.clip(render['rgb']+1-render['alpha'][...,None],0,1)*255)).crop(mask_box).save(path)
        paths.append(('C5 '+json.loads((output/'cleanup_metrics.json').read_text())['method'],path))
    canvas=Image.new('RGB',(300*len(paths),352),'#e4e4e4');draw=ImageDraw.Draw(canvas)
    for i,(name,path) in enumerate(paths):
        image=ImageOps.contain(Image.open(path).convert('RGB').crop(box),(296,320))
        draw.text((i*300+5,6),name,fill='black');canvas.paste(image,(i*300+(300-image.width)//2,28+(320-image.height)//2))
    out=p/'comparisons/visualizations'/('DARK_CURLED_LEAF_LOSS_C0_C5.png' if args.c5_output else 'DARK_CURLED_LEAF_LOSS_C0_C3.png')
    out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out)
    camera_index=next(v['index'] for v in json.loads((d/'QUALITY_INSPECTION.json').read_text())['selected_views'] if v['name']==view)
    record=dict(view=view,camera_index=camera_index,largest_new_foreground_hole_pixels=int(sizes.max()),evaluation_crop=list(box),
        crop_selection='Automatic largest newly lost foreground alpha component in the C3 worst thin-coverage camera; evaluation only',
        direct_visual_observation='Target RGB shows a curled/drooping leaf with green upper edge and purple-brown lower surface; the original GGGS render is already obscured by a nearly black band, and pruning removes its alpha occupancy',
        interpretation='Alpha loss alone cannot prove valid leaf Gaussian deletion. Native evidence shows original luminance near zero versus target 0.193 and no high-confidence depth support here; cleanup exposes an existing reconstruction gap. This is separate from the blue marker.',
        source_images=[str(path.resolve()) for _,path in paths],visualization=str(out.resolve()),source_sha256=sha(Path(__file__).resolve()),
        review_method='Direct model vision on target photograph and native renders; not Agnes API or human signature',algorithms_using_this_evaluation_crop=False)
    write_json(p/'comparisons/DARK_CURLED_LEAF_LOSS_CASE.json',record);print(out.resolve())


if __name__=='__main__':main()
