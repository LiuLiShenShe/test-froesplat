"""Eight native leaf-texture windows selected from GT only, never from a model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'root_cause_audit/scripts'))
from render_evaluate_common import gradients,psnr_region
EXPS=['E1','E5','E7','E8','E9','E10'];IDS=[0,26,53,79,105,131,157,183]
OUT=ROOT/'render_comparison/visualizations'

def main():
    protocols=[];metrics=[]
    for sid in IDS:
        values={}
        for e in EXPS:
            with np.load(ROOT/'render_comparison'/e/'views'/f'crop_{sid:04d}'/'raw_render_outputs.npz') as z:
                values[e]=(z['render_rgb'],z['gt_rgb'],z['foreground_mask'])
        gt=values['E1'][1];fg=values['E1'][2];h,w=fg.shape
        yy=np.nonzero(fg)[0];upper=fg.copy();upper[int(yy.min()+.6*(yy.max()-yy.min())):]=False
        candidates=ndimage.binary_erosion(fg,iterations=12)&upper;fallback=not candidates.any()
        if fallback:candidates=upper
        gradient=gradients(gt);score=ndimage.uniform_filter(gradient,size=31)
        score[~candidates]=-np.inf;cy,cx=np.unravel_index(np.argmax(score),fg.shape)
        x0=int(max(0,min(cx-160,w-320)));y0=int(max(0,min(cy-160,h-320)))
        bounds=[x0,y0,x0+320,y0+320];selected=np.zeros_like(fg);selected[y0:y0+320,x0:x0+320]=True
        pixels=selected&fg;edges=pixels&~ndimage.binary_erosion(fg,iterations=5)
        panels=[gt]+[values[e][0] for e in EXPS];labels=['GT']+EXPS
        canvas=Image.new('RGB',(320*7,344),'white');draw=ImageDraw.Draw(canvas)
        for i,(a,label) in enumerate(zip(panels,labels)):
            crop=(np.clip(a[y0:y0+320,x0:x0+320],0,1)*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(crop),(320*i,24));draw.text((320*i+8,5),label,fill='black')
        canvas.save(OUT/f'source_{sid:04d}_leaf_texture_GT_selected_native320.png')
        protocols.append(dict(source_id=sid,bounds_native=bounds,side=320,foreground_pixels=int(pixels.sum()),fallback=fallback,
            selection='Maximum GT31pixel mean Sobel magnitude within original mask eroded12pixels and upper60percent foreground height; one GT-only window for all models; native pixels, no interpolation'))
        for e in EXPS:
            rgb,other_gt,other_mask=values[e];assert np.array_equal(gt,other_gt) and np.array_equal(fg,other_mask)
            error=(rgb-gt)**2;g=gradients(rgb)
            metrics.append(dict(experiment=e,source_id=sid,foreground_pixels=int(pixels.sum()),inner_edge_pixels=int(edges.sum()),
                fg_psnr=psnr_region(error,pixels),inner_edge5_psnr=psnr_region(error,edges),
                fg_gradient_retention=float(g[pixels].mean()/gradient[pixels].mean()),fg_gradient_mae=float(np.abs(g-gradient)[pixels].mean())))
    (OUT/'GT_texture_crop_protocol.json').write_text(json.dumps(dict(records=protocols,
        original_fixed_audit_windows_retained=True,reason='Some fixed audit windows contain almost only background at rotated views; GT-only interior windows supply meaningful leaf texture evidence in all eight views',
        unchanged_model_inputs_masks_and_checkpoints=True),indent=2)+'\n')
    with (ROOT/'render_comparison/GT_texture_crop_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    print('eight GT-selected native leaf texture windows saved; original masks/checkpoints unchanged')

if __name__=='__main__':main()
