"""Same metric code, library environment, GT and masks as E10."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['MPLBACKEND']='Agg'
import csv,hashlib,json,sys,time
import numpy as np
import torch
import lpips
from scipy import ndimage
from skimage.metrics import structural_similarity
from surface_protocol import ROOT,DUAL,write
sys.path.insert(0,str(DUAL.parent/'root_cause_audit/scripts'))
from render_evaluate_common import psnr_region,gradients

def main():
    out=ROOT/'E11_2DGS_Corrected';inp=ROOT/'surface_inputs/e11_2dgs_surface'
    if (out/'evaluation_complete.json').exists():return
    index=json.loads((inp/'render_index.json').read_text());info=json.loads((inp/'render_complete.json').read_text());rows=[]
    torch.set_num_threads(8);start=time.time();perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval();torch.cuda.reset_peak_memory_stats()
    for item in index:
        with np.load(item['path']) as d:
            rgb=d['render_raw'].clip(0,1);gt=d['gt_rgb'];fg=d['foreground_mask'];alpha=d['alpha']
        boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5);inner=fg&boundary;err=(rgb-gt)**2
        ssim,ssim_map=structural_similarity(gt,rgb,channel_axis=2,data_range=1.,win_size=7,gaussian_weights=False,use_sample_covariance=True,full=True)
        valid=np.ones_like(fg);valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
        with torch.no_grad():
            ta=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()*2-1;tb=torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda()*2-1;lp=float(perceptual(ta,tb))
        grad=gradients(rgb);gg=gradients(gt);ratio=float(grad[fg].mean()/gg[fg].mean());alpha50=alpha>.5
        r=dict(experiment='E11',source_id=item['source_id'],width=item['camera']['width'],height=item['camera']['height'],full_psnr=psnr_region(err,np.ones_like(fg)),fg_mask_psnr=psnr_region(err,fg),
            boundary_band5_psnr=psnr_region(err,boundary),inner_fg_edge5_psnr=psnr_region(err,inner),ssim=float(ssim),fg_mask_ssim_map=float(ssim_map.mean(2)[fg&valid].mean()),lpips_alex=lp,
            fg_gradient_retention=ratio,fg_gradient_ratio_abs_error=abs(ratio-1),edge_gradient_retention=float(grad[inner].mean()/gg[inner].mean()),fg_gradient_mae=float(np.abs(grad-gg)[fg].mean()),
            foreground_alpha_mean=float(alpha[fg].mean()),background_alpha_mean=float(alpha[~fg].mean()),background_alpha_gt05_fraction=float((alpha[~fg]>.5).mean()),alpha_mask05_iou=int((alpha50&fg).sum())/int((alpha50|fg).sum()),
            gaussian_count=info['model_count'],gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(fg.tobytes()).hexdigest())
        assert r['gt_sha256']==item['gt_sha256'] and r['mask_sha256']==item['mask_sha256'];rows.append(r)
        with (out/'per_view_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerows(rows)
        print('E11',r['source_id'],'PSNR',r['full_psnr'],'FG',r['fg_mask_psnr'],flush=True)
        del ta,tb,ssim_map
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(int,float)) and k!='source_id'}
    write(out/'evaluation_complete.json',dict(status='complete',views=208,summary=summary,elapsed_seconds=time.time()-start,physical_gpu=1,
        metric_source_sha256=hashlib.sha256((DUAL.parent/'root_cause_audit/scripts/render_evaluate_common.py').read_bytes()).hexdigest(),torch=torch.__version__,lpips='Alex full1080x1920',evaluation='Same E10 audit region/gradient functions and skimage win7/sample covariance; float RGB clip[0,1], no PNG or resizing'))
    comparisons=[]
    for exp in ['E8','E10','E11']:
        table=rows if exp=='E11' else list(csv.DictReader((DUAL/'render_comparison'/exp/'per_view_metrics.csv').open()))
        row=dict(experiment=exp,views=len(table))
        for k in summary:row[k]=float(np.mean([float(r[k]) for r in table]))
        if exp=='E11':
            meta=json.loads((ROOT/'E11_2DGS_Corrected/training_complete.json').read_text())
            row['training_wall_seconds']=meta['training_wall_seconds'];row['peak_allocated_gib']=meta['peak_allocated_gib']
        else:
            meta=next(r for r in csv.DictReader((DUAL/'reconstruction_comparison.csv').open()) if r['experiment']==exp)
            row['training_wall_seconds']=float(meta['full_training_command_wall_seconds']);row['peak_allocated_gib']=float(meta['peak_full_training_command_allocated_gib'])
        comparisons.append(row)
    path=ROOT/'comparisons/rendering_comparison.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)

if __name__=='__main__':main()
