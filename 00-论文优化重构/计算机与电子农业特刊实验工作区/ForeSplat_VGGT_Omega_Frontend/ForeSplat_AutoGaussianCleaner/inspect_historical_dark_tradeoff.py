#!/usr/bin/env python3
"""Export actual photo/error crops explaining historical foreground tradeoffs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from scipy import ndimage
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.historical.evidence import camera_indices
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();root=a.experiment.resolve()
    r=json.loads((root/'execution.json').read_text())['reference']
    scene=Scene(r['scene'],r['config']);torch.set_num_threads(8)
    dest=root/'comparison/dark_tradeoff';dest.mkdir(parents=True,exist_ok=True)
    historical=Path(scene.spec['baseline_models']['R5'])
    paths={'Original':Path(scene.spec['model']),
           'Historical_R5_marker':historical.parent.parent/'additional_rounds/round_05/gaussians.ply',
           'Historical_best':historical}
    selected=root/'stage1_generalized/cleaned_gaussians.ply'
    if not selected.exists():
        progress=json.loads((root/'stage1_generalized/progress.json').read_text())
        selected=root/'stage1_generalized/rounds'/f'{progress["round"]:02d}'/'accepted.ply'
    paths['Generalized']=selected
    models={label:scene.load_model(path)[0] for label,path in paths.items()}
    patches=[];records=[]
    for vi in camera_indices(scene,24):
        c,target,mask,depth,conf=scene.get_view(vi)
        with torch.no_grad():renders={label:arrays(scene.render(c,model)) for label,model in models.items()}
        old=((renders['Original']['rgb']-target)**2).mean(2)
        dark=mask&(target@[.2126,.7152,.0722]<.12)
        new=((renders['Historical_R5_marker']['rgb']-target)**2).mean(2)
        damage=dark&(new>old+.005)
        labels,count=ndimage.label(damage)
        for component,slices in enumerate(ndimage.find_objects(labels),1):
            if slices is None:continue
            roi=labels[slices]==component;area=int(roi.sum())
            if area<6:continue
            weight=float((new-old)[slices][roi].sum())
            y,x=slices;margin=20
            yy=slice(max(0,y.start-margin),min(mask.shape[0],y.stop+margin))
            xx=slice(max(0,x.start-margin),min(mask.shape[1],x.stop+margin))
            row=dict(view=c.image_name,index=vi,component=component,damaged_dark_pixels=area,weighted_error=weight,
                     crop=[xx.start,yy.start,xx.stop,yy.stop])
            records.append(row)
            imgs={'Photo_target':target,**{label:render['rgb'] for label,render in renders.items()}}
            if len(patches)<36 or weight>min(s[0] for s in patches):
                patches.append((weight,row,{label:im[yy,xx].copy() for label,im in imgs.items()}))
                patches=sorted(patches,key=lambda x:-x[0])[:36]
        print('dark tradeoff view',vi,flush=True)
    for rank,(_,row,imgs) in enumerate(patches):
        canvas=Image.new('RGB',(320*len(imgs),345),'white');draw=ImageDraw.Draw(canvas)
        for j,(label,img) in enumerate(imgs.items()):
            im=Image.fromarray(np.uint8(np.clip(img,0,1)*255));im.thumbnail((315,315))
            # Upscale small regions with nearest pixels so crop context remains
            # inspectable without inventing texture or a generated image.
            scale=min(315/im.width,315/im.height)
            if scale>1:im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
            canvas.paste(im,(j*320,25));draw.text((j*320+4,5),label,fill='black')
        path=dest/f'{rank:02d}_{row["view"]}.png';canvas.save(path);row['sheet']=str(path)
    write_json(dest/'DARK_TRADEOFF.json',dict(models={k:str(v) for k,v in paths.items()},
        ranked_patches=[r for _,r,_ in patches],all_components=records,
        role='Diagnostic visualization only; no coordinates or IDs are fed back as cleaning ROIs'))


if __name__=='__main__':main()
