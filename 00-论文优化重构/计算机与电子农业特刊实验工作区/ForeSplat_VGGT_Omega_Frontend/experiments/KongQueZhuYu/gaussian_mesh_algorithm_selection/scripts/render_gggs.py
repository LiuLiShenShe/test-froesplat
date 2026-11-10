"""Unified208 native GGGS float evaluation, raw and EO corrected kept separate."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MPLBACKEND='Agg')
import argparse,csv,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity
from gggs_runtime import OUT,DUAL,OLD,setup,cameras,mask
sys.path.insert(0,str(OUT.parent/'root_cause_audit/scripts'))
from render_evaluate_common import psnr_region,gradients
sys.path.insert(0,str(OLD/'scripts'))
from surface_protocol import save_npz_fast,write
p=argparse.ArgumentParser();p.add_argument('--ply',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--experiment',required=True);p.add_argument('--exposure',type=Path);p.add_argument('--repo',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--method',choices=['GGGS','2DGS'],default='GGGS');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
if (a.out/'render_complete.json').exists():raise SystemExit(0)
if a.method=='GGGS':
    GaussianModel,native=setup(a.repo);model=GaussianModel(3,0)
else:
    sys.path.insert(0,str(a.repo));from scene import GaussianModel;import gaussian_renderer as native;model=GaussianModel(3)
model.load_ply(str(a.ply));cams,refs=cameras(a.reference)
from utils.general_utils import PILtoTorch
import lpips
perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval();torch.set_num_threads(8)
pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,depth_ratio=0.0)
bg=torch.zeros(3,device='cuda');ds=DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected'
truth={int(r['source_id']):r for r in csv.DictReader((DUAL/'render_comparison/E10/per_view_metrics.csv').open())}
matrices={}
if a.exposure:
    with np.load(a.exposure,allow_pickle=True) as e:matrices={str(n):m for n,m in zip(e['camera_names'],e['matrices'])}
    assert len(matrices)==208
def scores(rgb,gt,fg,alpha):
    rgb=np.clip(rgb,0,1);err=(rgb-gt)**2;allpixels=np.ones_like(fg);boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5);inner=fg&boundary
    score,ss=structural_similarity(gt,rgb,channel_axis=2,data_range=1.,win_size=7,gaussian_weights=False,use_sample_covariance=True,full=True)
    valid=np.ones_like(fg);valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
    with torch.no_grad():lp=float(perceptual(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()*2-1,torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda()*2-1))
    grad=gradients(rgb);gtgrad=gradients(gt);ratio=float(grad[fg].mean()/gtgrad[fg].mean());alpha50=alpha>.5
    return dict(full_psnr=psnr_region(err,allpixels),fg_mask_psnr=psnr_region(err,fg),boundary_band5_psnr=psnr_region(err,boundary),inner_fg_edge5_psnr=psnr_region(err,inner),ssim=float(score),fg_mask_ssim_map=float(ss.mean(2)[fg&valid].mean()),lpips_alex=lp,fg_gradient_retention=ratio,fg_gradient_ratio_abs_error=abs(ratio-1),edge_gradient_retention=float(grad[inner].mean()/gtgrad[inner].mean()),fg_gradient_mae=float(np.abs(grad-gtgrad)[fg].mean()),foreground_alpha_mean=float(alpha[fg].mean()),background_alpha_mean=float(alpha[~fg].mean()),background_alpha_gt05_fraction=float((alpha[~fg]>.5).mean()),alpha_mask05_iou=int((alpha50&fg).sum())/int((alpha50|fg).sum()),gaussian_count=len(model.get_xyz))
start=time.time();torch.cuda.reset_peak_memory_stats();rows=[];index=[]
for c,r in zip(cams,refs):
    sid=int(c.image_name.split('_')[-1]);dest=a.out/'rendered'/f'{c.image_name}.npz';dest.parent.mkdir(exist_ok=True)
    paths=list((ds/'images').glob(c.image_name+'.*'));assert len(paths)==1
    with Image.open(paths[0]) as image:gt=PILtoTorch(image,(c.image_width,c.image_height)).numpy().transpose(1,2,0).astype(np.float32)
    fg=mask(c)
    assert hashlib.sha256(gt.tobytes()).hexdigest()==truth[sid]['gt_sha256']
    assert hashlib.sha256(fg.tobytes()).hexdigest()==truth[sid]['mask_sha256']
    with torch.no_grad():pkg=native.render(c,model,pipe,bg,0.) if a.method=='GGGS' else native.render(c,model,pipe,bg)
    raw=pkg['render'].cpu().numpy().transpose(1,2,0).astype(np.float32);alpha=pkg['mask' if a.method=='GGGS' else 'rend_alpha'].cpu().numpy().squeeze();depth=pkg['median_depth' if a.method=='GGGS' else 'surf_depth'].cpu().numpy().squeeze();normal=pkg['normal' if a.method=='GGGS' else 'rend_normal'].cpu().numpy().transpose(1,2,0)
    extra={}
    for stage in ['raw','corrected'] if matrices else ['raw']:
        image=raw
        if stage=='corrected':
            m=matrices[c.image_name];image=raw@m[:,:3].T+m[:,3];extra['render_corrected']=image.astype(np.float32)
        rows.append(dict(experiment=a.experiment,stage=stage,source_id=sid,width=c.image_width,height=c.image_height,gt_sha256=truth[sid]['gt_sha256'],mask_sha256=truth[sid]['mask_sha256'],**scores(image,gt,fg,alpha)))
    save_npz_fast(dest,render_raw=raw,gt_rgb=gt,foreground_mask=fg,alpha=alpha,depth_surface=depth,native_normal=normal,**extra)
    index.append(dict(source_id=sid,path=str(dest),camera=dict(r,source_id=sid),gt_sha256=truth[sid]['gt_sha256'],mask_sha256=truth[sid]['mask_sha256']))
    if sid in [0,26,53,79,105,131,157,183]:
        d=a.out/'views'/f'{sid:04d}';d.mkdir(parents=True,exist_ok=True)
        for name,img in [('RGB',raw),('Normal',normal*.5+.5),('Alpha',np.repeat(alpha[...,None],3,2))]+([('RGB_corrected',extra['render_corrected'])] if extra else []):
            Image.fromarray((np.clip(img,0,1)*255).round().astype(np.uint8)).save(d/f'{name}.png')
    with (a.out/'per_view_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.experiment,sid,rows[-1]['full_psnr'],flush=True);del pkg
write(a.out/'camera_runtime.json',dict(records=refs));write(a.out/'render_index.json',index)
write(a.out/'render_complete.json',dict(status='complete',experiment=a.experiment,views=208,ply=str(a.ply.resolve()),wall_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,raw_and_corrected_separate=bool(matrices),metric_protocol='same E10 native float/trueFG/SSIMwin7/samplecov/AlexLPIPS/Rec709Sobel8; RGB clipped0..1 for metrics',hashes_match_E10=True))

