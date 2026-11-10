"""Apply the exact historical E10/E11 floating rendering metric protocol to E14 stages."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,json,csv,time,hashlib
from pathlib import Path
import numpy as np,torch,lpips
from scipy import ndimage
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent.parent/'root_cause_audit/scripts'))
from render_evaluate_common import psnr_region,gradients
def cpu_metric_row(rgb,gt,fg,alpha,item,model_count,stage,lp):
    """Use the historical float metric formulas and retain actual reference hashes."""
    assert np.isfinite(rgb).all() and np.isfinite(gt).all() and np.isfinite(alpha).all()
    err=(rgb-gt)**2
    ssim,ssim_map=structural_similarity(gt,rgb,channel_axis=2,data_range=1.,win_size=7,gaussian_weights=False,use_sample_covariance=True,full=True)
    boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5);inner=fg&boundary
    valid=np.ones_like(fg);valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
    grad,gg=gradients(rgb),gradients(gt)
    ratio=float(grad[fg].mean()/gg[fg].mean());alpha50=alpha>.5
    r=dict(experiment='E14',stage=stage,source_id=item['source_id'],width=rgb.shape[1],height=rgb.shape[0],
        full_psnr=psnr_region(err,np.ones_like(fg)),fg_mask_psnr=psnr_region(err,fg),boundary_band5_psnr=psnr_region(err,boundary),inner_fg_edge5_psnr=psnr_region(err,inner),
        ssim=float(ssim),fg_mask_ssim_map=float(ssim_map.mean(2)[fg&valid].mean()),lpips_alex=lp,
        fg_gradient_retention=ratio,fg_gradient_ratio_abs_error=abs(ratio-1),edge_gradient_retention=float(grad[inner].mean()/gg[inner].mean()),fg_gradient_mae=float(np.abs(grad-gg)[fg].mean()),
        background_alpha_mean=float(alpha[~fg].mean()),background_alpha_gt05_fraction=float((alpha[~fg]>.5).mean()),foreground_alpha_mean=float(alpha[fg].mean()),
        alpha_mask05_iou=int((alpha50&fg).sum())/int((alpha50|fg).sum()),gaussian_count=model_count,
        gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(fg.tobytes()).hexdigest())
    assert r['gt_sha256']==item['gt_sha256'] and r['mask_sha256']==item['mask_sha256']
    assert all(np.isfinite(v) for v in r.values() if isinstance(v,(int,float))),('Nonfinite actual metric',stage,item['source_id'])
    return r
def main():
    if (ROOT/'evaluation_complete.json').exists() and json.loads((ROOT/'evaluation_complete.json').read_text())['status']=='complete':return
    torch.set_num_threads(8)
    started=time.time();net=None
    all_rows=[];summaries={};unavailable=[];reused=[]
    for stage in ['initial','cluster_before','cluster_after','final_30000','refined']:
        folder=ROOT/'stage_renders'/stage
        if not (folder/'render_complete.json').exists():
            unavailable.append(dict(stage=stage,reason='No completed native rendering stage'));continue
        info=json.loads((folder/'render_complete.json').read_text())
        if info.get('status')!='complete':
            unavailable.append(dict(stage=stage,reason=info.get('reason','Native render unavailable')));continue
        index=json.loads((folder/'render_index.json').read_text());assert len(index)==208
        # The recovery only adds the refined render. Preserve the already measured
        # immutable stages instead of repeating their full-resolution LPIPS pass.
        # Check actual identities, finite values and native model counts first.
        cached=folder/'per_view_metrics.csv'
        prior=json.loads((ROOT/'evaluation_complete.json').read_text()) if (ROOT/'evaluation_complete.json').is_file() else {}
        if cached.is_file() and stage in prior.get('stages',{}):
            with cached.open(newline='') as f: old=list(csv.DictReader(f))
            ids={int(item['source_id']):item for item in index}
            valid=len(old)==208 and {int(r['source_id']) for r in old}==set(ids)
            numeric=[k for k in prior['stages'][stage]]
            for r in old:
                item=ids.get(int(r['source_id']),{})
                valid=valid and r['experiment']=='E14' and r['stage']==stage
                valid=valid and r['gt_sha256']==item.get('gt_sha256') and r['mask_sha256']==item.get('mask_sha256')
                valid=valid and float(r['gaussian_count'])==info['model_count']
                valid=valid and all(np.isfinite(float(r[k])) for k in numeric)
            if valid:
                rows=[{k:(int(v) if k=='source_id' else float(v) if k in numeric else v) for k,v in r.items()} for r in old]
                all_rows.extend(rows)
                summaries[stage]={k:float(np.mean([r[k] for r in rows])) for k in numeric}
                reused.append(stage)
                print('E14 reused verified208 metric stage',stage,flush=True)
                continue
        if net is None:net=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
        rows=[]
        for item in index:
            with np.load(item['path']) as data:
                native_raw=data['render_raw'];assert np.isfinite(native_raw).all(),(stage,item['source_id'],'Nonfinite native RGB')
                rgb=native_raw.clip(0,1);gt=data['gt_rgb'];fg=data['foreground_mask'];alpha=data['alpha']
            assert np.isfinite(gt).all() and np.isfinite(alpha).all(),(stage,item['source_id'],'Nonfinite reference or alpha')
            with torch.no_grad():
                a=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()*2-1
                b=torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda()*2-1
                lp=float(net(a,b))
            r=cpu_metric_row(rgb,gt,fg,alpha,item,info['model_count'],stage,lp)
            rows.append(r);all_rows.append(r)
            with (folder/'per_view_metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerows(rows)
            print('E14 eval',stage,item['source_id'],r['full_psnr'],flush=True)
        summaries[stage]={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(float,int)) and k!='source_id'}
    with (ROOT/'all_stage_rendering_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    (ROOT/'evaluation_complete.json').write_text(json.dumps(dict(status='complete' if not unavailable else 'partial',views=208,stages=summaries,unavailable_stages=unavailable,elapsed_seconds=time.time()-started,reused_prior_208_metric_stages=reused,cached_metrics_scope='Previously completed stages retain actual measurements after208 ID, GT/RAP hash, finite-value and native model-count verification; elapsed time is this recovery pass, prior evaluation cost retained in actual command logs',protocol='Same E10/E11 region PSNR/gradient, skimage7/sample-covariance SSIM, AlexLPIPS float1080x1920; metricRGBclipped0..1 as historical208 evaluator; nativeRawNPZ unclipped, no resizing or PNG'),indent=2)+'\n')
if __name__=='__main__':main()

