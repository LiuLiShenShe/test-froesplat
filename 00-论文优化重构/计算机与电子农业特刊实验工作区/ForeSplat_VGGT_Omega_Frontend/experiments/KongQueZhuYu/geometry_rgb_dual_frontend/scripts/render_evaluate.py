"""All208 native-float metrics; fixed E6 views plus root-audit views saved."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['MPLBACKEND']='Agg'
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT.parent/'root_cause_audit';OMEGA=ROOT.parents[2]
sys.path.insert(0,str(AUDIT/'scripts'))
from render_evaluate_common import psnr_region,gradients
from render_checkpoint_views import camera_record,json_write
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint-root',type=Path,required=True);a=p.parse_args()
OUT=ROOT/'render_comparison'/a.experiment
if (OUT/'render_complete.json').exists():raise SystemExit(0)
OUT.mkdir(parents=True,exist_ok=True);runtime=OUT/'runtime_checkpoint';linked=runtime/'point_cloud/iteration_30000';linked.mkdir(parents=True,exist_ok=True)
checkpoint=a.checkpoint_root/'point_cloud/iteration_30000/point_cloud.ply'
if not (linked/'point_cloud.ply').exists():(linked/'point_cloud.ply').symlink_to(checkpoint)
sys.path.insert(0,str(a.repo))
from utils.cuda_preload import preload_cuda_ext_deps
preload_cuda_ext_deps()
from scene import Scene,GaussianModel
import gaussian_renderer as native
import lpips
torch.set_num_threads(8);begin=time.perf_counter()
ns=argparse.Namespace(sh_degree=3,sg_degree=0,source_path=str(a.dataset),model_path=str(runtime),images='images',dataset='',resolution=2,white_background=False,data_device='cpu',eval=False,use_decoupled_appearance=0,disable_filter3D=False,kernel_size=0.,multi_view_num=8,multi_view_max_angle=30,multi_view_min_dis=.01,multi_view_max_dis=1.5)
pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
gaussians=GaussianModel(3,0);scene=Scene(ns,gaussians,load_iteration=30000,shuffle=False);cams=scene.getTrainCameras();assert len(cams)==208
json_write(OUT/'camera_runtime.json',dict(records=[camera_record(c) for c in cams]))
perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval();torch.cuda.reset_peak_memory_stats()
bg=torch.zeros(3,device='cuda');rows=[];ids=[i for i in range(210) if i not in (44,206)]
representative=sorted(set([ids[i] for i in range(0,208,26)]+[0,19,38,57,76,95,114,133,152,170,189,209]))
for cam in cams:
    sid=int(cam.image_name.split('_')[-1]);t=time.perf_counter()
    with torch.no_grad():pkg=native.render(cam,gaussians,pipe,bg,0.)
    raw=pkg['render'].cpu().numpy().transpose(1,2,0).astype(np.float32);rgb=raw.clip(0,1);gt=cam.original_image[:3].cpu().numpy().transpose(1,2,0).astype(np.float32)
    fg=np.asarray(Image.open(a.dataset/'masks'/f'{cam.image_name}.png').convert('L').resize((cam.image_width,cam.image_height),Image.Resampling.NEAREST))>127
    alpha=pkg['mask'].cpu().numpy().squeeze();depth=pkg['median_depth'].cpu().numpy().squeeze();normal=pkg['normal'].cpu().numpy().transpose(1,2,0)
    boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5);inner=fg&boundary;all_pixels=np.ones_like(fg)
    err=(rgb-gt)**2;score,ssim_map=structural_similarity(gt,rgb,channel_axis=2,data_range=1.,win_size=7,gaussian_weights=False,use_sample_covariance=True,full=True)
    ss=ssim_map.mean(2);valid=np.ones_like(fg);valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
    with torch.no_grad():
        ta=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()*2-1;tb=torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda()*2-1;lp=float(perceptual(ta,tb))
    grad=gradients(rgb);gtgrad=gradients(gt);alpha50=alpha>.5
    fg_grad_ratio=float(grad[fg].mean()/gtgrad[fg].mean())
    row=dict(experiment=a.experiment,source_id=sid,width=cam.image_width,height=cam.image_height,full_psnr=psnr_region(err,all_pixels),fg_mask_psnr=psnr_region(err,fg),boundary_band5_psnr=psnr_region(err,boundary),inner_fg_edge5_psnr=psnr_region(err,inner),
        ssim=float(score),fg_mask_ssim_map=float(ss[fg&valid].mean()),lpips_alex=lp,fg_gradient_retention=fg_grad_ratio,fg_gradient_ratio_abs_error=abs(fg_grad_ratio-1),edge_gradient_retention=float(grad[inner].mean()/gtgrad[inner].mean()),fg_gradient_mae=float(np.abs(grad-gtgrad)[fg].mean()),
        foreground_alpha_mean=float(alpha[fg].mean()),background_alpha_mean=float(alpha[~fg].mean()),background_alpha_gt05_fraction=float((alpha[~fg]>.5).mean()),alpha_mask05_iou=int((alpha50&fg).sum())/int((alpha50|fg).sum()),
        gaussian_count=int(gaussians.get_xyz.shape[0]),gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(fg.tobytes()).hexdigest(),render_seconds=time.perf_counter()-t)
    rows.append(row)
    if sid in representative:
        d=OUT/'views'/f'crop_{sid:04d}';d.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(d/'raw_render_outputs.npz',render_raw=raw,render_rgb=rgb,gt_rgb=gt,foreground_mask=fg,alpha=alpha,depth_median=depth,native_normal=normal,normal_camera=normal)
        Image.fromarray((rgb*255).round().astype(np.uint8)).save(d/'render.png');Image.fromarray((gt*255).round().astype(np.uint8)).save(d/'gt.png')
        json_write(d/'render_provenance.json',dict(row,radii_positive=int((pkg['radii']>0).sum())))
    with (OUT/'per_view_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    del pkg,ta,tb,raw,rgb,gt,ssim_map,normal,depth,alpha
    print(a.experiment,sid,'PSNR',row['full_psnr'],'FG',row['fg_mask_psnr'],'edge',row['inner_fg_edge5_psnr'],flush=True)
summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(int,float)) and k not in ('source_id',)}
summary.update(experiment=a.experiment,views=len(rows),representative_source_ids=representative)
json_write(OUT/'render_complete.json',dict(status='complete',summary=summary,wall_seconds=time.perf_counter()-begin,physical_gpu=1,repo=str(a.repo),dataset=str(a.dataset),checkpoint=str(checkpoint),
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,metrics='Same root-cause audit implementation: native float RGB, true binary FG, 5px edge, skimage win7/sample covariance, cached AlexLPIPS full resolution, Rec709 Sobel/8; no PNG metrics',source_ids=ids,renderer_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),evaluator_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),audit_metric_source_sha256=hashlib.sha256((AUDIT/'scripts/render_evaluate_common.py').read_bytes()).hexdigest()))
