#!/usr/bin/env python3
"""Actual native-render worst-view inspection; selected automatically from metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from scipy import ndimage
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene,camera
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.artifacts import image_pair


def selected_views(report,maximum=6):
    old={v['index']:v for v in report['initial']['views']}
    views=report['final']['views'];chosen=[]
    def dark(v):
        a,b=v.get('dark_foreground_mse'),old[v['index']].get('dark_foreground_mse')
        return a-b if a is not None and b is not None else 0.
    keys=[lambda v:v['fg_mse']-old[v['index']]['fg_mse'],
        lambda v:old[v['index']]['thin_coverage']-v['thin_coverage'],dark,
        lambda v:old[v['index']]['boundary_coverage']-v['boundary_coverage'],
        lambda v:v.get('new_depth_invalid_fraction',0),lambda v:v['depth_changed_fraction']]
    for key in keys:
        for view in sorted(views,key=lambda v:(-key(v),v['index'])):
            if view['index'] not in chosen:chosen.append(view['index']);break
        if len(chosen)>=maximum:break
    return chosen


def inspect(output,baseline=None):
    output=Path(output).resolve()
    if (output/'INVALIDATED.json').exists():raise ValueError('Invalidated experiment')
    complete=json.loads((output/'COMPLETE.json').read_text())
    report=json.loads((output/'cleanup_metrics.json').read_text())
    if 'initial' not in report:
        if baseline is None:raise ValueError('Historical/rule-only inspection requires the explicit matched C0 baseline')
        old=json.loads((Path(baseline)/'cleanup_metrics.json').read_text())
        if old['input_sha256']!=report['input_sha256'] or old['scene']!=report['scene']:
            raise ValueError('Inspection baseline belongs to another GGGS input/scene')
        report={**report,'initial':old['metrics'],'final':report['metrics']}
    if complete['native_sha256']!=sha(output/'cleaned_gaussians.ply'):raise ValueError('Changed final native model')
    scene=Scene(complete.get('scene') or report['scene'],complete.get('config') or report['config'])
    indices=selected_views(report)
    folder=output/'quality_inspection';folder.mkdir(exist_ok=True)
    rendered={}
    for label,path in [('original',scene.spec['model']),('cleaned',output/'cleaned_gaussians.ply')]:
        model,_=scene.load_model(path)
        for index in indices:
            c,*_=scene.get_view(index)
            with torch.no_grad():a=arrays(scene.render(c,model))
            rendered.setdefault(index,{})[label]=a
            image_pair(folder/label/'training',scene.views[index]['name'],a)
        definitions=json.loads((output/'visualizations/automatic_free_views.json').read_text())['definitions']
        for uid,view in enumerate(definitions):
            c=camera(np.asarray(view['K']),np.asarray(view['E']),view['width'],view['height'],view['name'],uid)
            with torch.no_grad():a=arrays(scene.render(c,model))
            image_pair(folder/label/'free',view['name'],a)
        del model;torch.cuda.empty_cache()
    width,height=1440,390
    canvas=Image.new('RGB',(width,height*len(indices)),'#e4e4e4');draw=ImageDraw.Draw(canvas)
    records=[]
    for row,index in enumerate(indices):
        _,target,mask,*_=scene.get_view(index);before=rendered[index]['original'];after=rendered[index]['cleaned']
        loss=mask&(before['alpha']>.5)&(after['alpha']<=.5)
        gain=mask&(before['alpha']<=.5)&(after['alpha']>.5)
        components,count=ndimage.label(loss)
        sizes=np.bincount(components.ravel())[1:]
        overlay=after['rgb']+(1-after['alpha'][...,None]);overlay=overlay.copy()
        overlay[loss]=[1.,0.,0.];overlay[gain]=[0.,.6,1.]
        ys,xs=np.nonzero(mask)
        if len(xs):
            pad=max(8,round(.06*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)))
            crop=(max(0,xs.min()-pad),max(0,ys.min()-pad),min(mask.shape[1],xs.max()+pad+1),min(mask.shape[0],ys.max()+pad+1))
        else:crop=(0,0,mask.shape[1],mask.shape[0])
        images=[target,before['rgb']+(1-before['alpha'][...,None]),after['rgb']+(1-after['alpha'][...,None]),overlay]
        titles=['Target RGB','C0 native white',f'{report["method"]} native white','Red: new FG alpha holes / blue: restored']
        for col,(title,value) in enumerate(zip(titles,images)):
            image=Image.fromarray(np.uint8(np.clip(value,0,1)*255)).crop(crop)
            fitted=ImageOps.contain(image,(356,352));x=col*360;y=row*height
            draw.text((x+4,y+4),title,fill='black')
            canvas.paste(fitted,(x+(360-fitted.width)//2,y+24+(352-fitted.height)//2))
            image.save(folder/f'{scene.views[index]["name"]}_{col}.png')
        draw.text((4,row*height+376),f'{scene.views[index]["name"]}: newly lost alpha pixels={int(loss.sum())}, FG mask={int(mask.sum())}',fill='black')
        records.append(dict(index=index,name=scene.views[index]['name'],new_fg_hole_pixels=int(loss.sum()),
            new_fg_hole_fraction=float(loss.sum()/max(mask.sum(),1)),largest_new_hole_pixels=int(sizes.max()) if len(sizes) else 0,
            largest_new_hole_fraction=float(sizes.max()/max(mask.sum(),1)) if len(sizes) else 0.,
            restored_fg_pixels=int(gain.sum()),native_resolution=[mask.shape[1],mask.shape[0]],automatic_mask_crop=[int(v) for v in crop]))
    contact=folder/'WORST_VIEW_CONTACT_SHEET.png';canvas.save(contact)
    result=dict(plant=scene.spec['plant'],method=report['method'],native_sha256=complete['native_sha256'],input_sha256=complete['input_sha256'],
        selected_views=records,contact_sheet=str(contact),source_sha256=sha(Path(__file__).resolve()),
        visual_review='pending_direct_image_inspection',
        selection='Distinct worst FG MSE, thin, dark MSE, boundary, depth holes and depth-change cameras; evaluation crops only, never cleaner ROIs',
        limitations='Target masks include retained scene foreground/calibration objects; alpha holes and target-dark regions are proxies, not leaf-vein semantic annotations')
    write_json(folder/'QUALITY_INSPECTION.json',result)
    print(scene.spec['plant'],contact,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outputs',type=Path,nargs='+',required=True)
    parser.add_argument('--baseline',type=Path,help='Explicit identical-input C0 for legacy/historical comparisons')
    args=parser.parse_args();torch.set_num_threads(8)
    for output in args.outputs:inspect(output,args.baseline)


if __name__=='__main__':main()
