"""Evaluate QGS using the historical E10/E11 float image metric protocol."""
import qgs_common as q
import csv,time,hashlib
import numpy as np
import torch,lpips
from scipy import ndimage
from skimage.metrics import structural_similarity
q.sys.path.insert(0,str(q.KONG/'root_cause_audit/scripts'))
from render_evaluate_common import psnr_region,gradients
def main():
 done=q.ROOT/'evaluation_complete.json'
 if done.exists():print(done.read_text());return
 rows=[];index=q.json.loads((q.ROOT/'surface_inputs/render_index.json').read_text())
 info=q.json.loads((q.ROOT/'surface_inputs/render_complete.json').read_text());start=time.time()
 torch.set_num_threads(8);perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval();torch.cuda.reset_peak_memory_stats()
 for item in index:
  with np.load(item['path']) as d:
   native_raw=d['render_raw'];assert np.isfinite(native_raw).all(),(item['source_id'],'Nonfinite native RGB; metrics are not valid')
   rgb=native_raw.clip(0,1);gt=d['gt_rgb'];fg=d['foreground_mask'];alpha=d['alpha']
  assert np.isfinite(gt).all() and np.isfinite(alpha).all(),(item['source_id'],'Nonfinite reference/alpha')
  boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5);inner=fg&boundary;err=(rgb-gt)**2
  score,ssim_map=structural_similarity(gt,rgb,channel_axis=2,data_range=1.,win_size=7,gaussian_weights=False,use_sample_covariance=True,full=True)
  valid=np.ones_like(fg);valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
  with torch.no_grad():
   ta=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()*2-1;tb=torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda()*2-1;lp=float(perceptual(ta,tb))
  grad=gradients(rgb);gg=gradients(gt);ratio=float(grad[fg].mean()/gg[fg].mean());alpha50=alpha>.5
  r=dict(experiment='E13',source_id=item['source_id'],width=item['camera']['width'],height=item['camera']['height'],
    full_psnr=psnr_region(err,np.ones_like(fg)),fg_mask_psnr=psnr_region(err,fg),boundary_band5_psnr=psnr_region(err,boundary),inner_fg_edge5_psnr=psnr_region(err,inner),
    ssim=float(score),fg_mask_ssim_map=float(ssim_map.mean(2)[fg&valid].mean()),lpips_alex=lp,
    fg_gradient_retention=ratio,fg_gradient_ratio_abs_error=abs(ratio-1),edge_gradient_retention=float(grad[inner].mean()/gg[inner].mean()),
    fg_gradient_mae=float(np.abs(grad-gg)[fg].mean()),foreground_alpha_mean=float(alpha[fg].mean()),background_alpha_mean=float(alpha[~fg].mean()),
    background_alpha_gt05_fraction=float((alpha[~fg]>.5).mean()),alpha_mask05_iou=int((alpha50&fg).sum())/int((alpha50|fg).sum()),
    gaussian_count=info['model_count'],gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(fg.tobytes()).hexdigest())
  assert r['gt_sha256']==item['gt_sha256'] and r['mask_sha256']==item['mask_sha256']
  assert all(np.isfinite(v) for k,v in r.items() if isinstance(v,(int,float))),('Nonfinite actual metric',item['source_id'])
  rows.append(r)
  with (q.ROOT/'per_view_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerows(rows)
  print('E13',r['source_id'],'PSNR',r['full_psnr'],'FG',r['fg_mask_psnr'],flush=True)
 assert len(rows)==208
 summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(int,float)) and k!='source_id'}
 q.write(done,dict(status='complete',views=208,summary=summary,elapsed_seconds=time.time()-start,physical_gpu=1,torch=torch.__version__,
     metric_source_sha256=hashlib.sha256((q.KONG/'root_cause_audit/scripts/render_evaluate_common.py').read_bytes()).hexdigest(),
     evaluation='Same E10/E11 region/gradient code, skimage win7/sample covariance and Alex LPIPS; original1080x1920 float RGB clip[0,1]'))
if __name__=='__main__':main()

