"""Stream every frozen view from native quadratic primitives; preserve float outputs."""
import qgs_common as q
import time,zipfile
import numpy as np
import torch
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
def save_npz(path,**arrays):
 temporary=path.with_suffix('.npz.partial')
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for key,array in arrays.items():
   with z.open(key+'.npy','w',force_zip64=True) as f:np.save(f,np.asarray(array),allow_pickle=False)
 temporary.replace(path)
def main():
 done=q.ROOT/'surface_inputs/render_complete.json'
 if done.exists():print(done.read_text());return
 assert (q.ROOT/'training_complete.json').is_file()
 cfg=OmegaConf.load(q.ROOT/'actual_config.yaml')
 Scene=q.read_only_image_adapter()
 model=GaussianModel(cfg.gs_model)
 scene=Scene(cfg,model,load_iteration=30000,shuffle=False)
 validation=q.validate_cameras(scene.getTrainCameras())
 q.write(q.ROOT/'surface_inputs/camera_runtime.json',validation)
 rows=q.json.loads((q.HIST/'surface_inputs/e10_gggs_surface/render_index.json').read_text())
 refs={int(r['source_id']):r for r in rows}
 out=q.ROOT/'surface_inputs';(out/'views').mkdir(parents=True,exist_ok=True)
 bg=torch.zeros(3,device='cuda');manifest=[];finite_reports=[];start=time.time()
 torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  for c in sorted(scene.getTrainCameras(),key=lambda c:c.image_name):
   sid=int(c.image_name.split('_')[-1]);ref=refs[sid];dest=out/'views'/f'{sid:04d}.npz'
   with np.load(ref['path']) as historical:fg=historical['foreground_mask'].copy()
   rec=q.camera_record(c);rec.update(source_id=sid)
   if not dest.exists():
    pkg=render(c,model,cfg.pipeline,cfg.optimizer,bg,kernel_size=cfg.dataset.kernel_size,return_depth=True,return_normal=True)
    raw=pkg['render'].cpu().numpy().transpose(1,2,0).copy()
    alpha=pkg['render_alpha'].cpu().numpy().squeeze()
    dep=pkg['surf_depth'].cpu().numpy().squeeze()
    # Match the exact native renderer's sanitization and denominator.
    expected=pkg['render_depth'].cpu().numpy().squeeze().copy()
    median_raw=pkg['render_all'][9:10].cpu().numpy().squeeze().copy()
    median=np.nan_to_num(median_raw,nan=0,posinf=0,neginf=0)
    normal=pkg['render_normal'].cpu().numpy().transpose(1,2,0)
    normal_length=np.linalg.norm(normal,axis=-1,keepdims=True)
    normal_unit=np.divide(normal,normal_length,out=np.zeros_like(normal),where=normal_length>1e-12)
    depth_normal=pkg['surf_normal'].cpu().numpy().transpose(1,2,0)
    gt=c.original_image.cpu().numpy().transpose(1,2,0).copy()
    active=model.active_sh_degree;model.active_sh_degree=0
    mesh_pkg=render(c,model,cfg.pipeline,cfg.optimizer,bg,kernel_size=-999.0,stop_z_gradient=True,return_depth=True,return_normal=True)
    diffuse=mesh_pkg['render'].cpu().numpy().transpose(1,2,0).copy()
    mesh_depth=mesh_pkg['surf_depth'].cpu().numpy().squeeze().copy()
    mesh_alpha=mesh_pkg['render_alpha'].cpu().numpy().squeeze().copy()
    mesh_normal=mesh_pkg['render_normal'].cpu().numpy().transpose(1,2,0).copy()
    model.active_sh_degree=active
    save_npz(dest,render_raw=raw,render_diffuse=diffuse,gt_rgb=gt,foreground_mask=fg,alpha=alpha,depth_surface=dep,
       depth_expected=expected,depth_median=median,depth_median_native_raw=median_raw,native_normal=normal,depth_normal_camera=depth_normal,
       depth_native_mesh=mesh_depth,alpha_native_mesh=mesh_alpha,normal_native_mesh_camera=mesh_normal,
       normal_world=normal@np.asarray(c.world_view_transform.cpu())[:3,:3].T,
       normal_unit_camera=normal_unit,normal_unit_world=normal_unit@np.asarray(c.world_view_transform.cpu())[:3,:3].T,
       depth_normal_world=depth_normal@np.asarray(c.world_view_transform.cpu())[:3,:3].T,
       render_dist=pkg['render_dist'].cpu().numpy().squeeze(),curvature=pkg['render_curvature'].cpu().numpy().squeeze())
    checked=dict(render_raw=raw,render_diffuse=diffuse,gt_rgb=gt,alpha=alpha,depth_expected=expected,depth_surface=dep,
      native_normal=normal,depth_normal_camera=depth_normal,depth_median_native_raw=median_raw,
      depth_native_mesh=mesh_depth,alpha_native_mesh=mesh_alpha,normal_native_mesh_camera=mesh_normal)
    counts={k:int((~np.isfinite(a)).sum()) for k,a in checked.items()}
    # Retain native invalid values as quality diagnostics. RGB/alpha/depth used
    # for metrics and fusion must actually be finite; no implicit PNG cast is
    # accepted as numerical validation.
    finite_reports.append(dict(source_id=sid,nonfinite_values=counts,
      raw_normal_and_median_retained=True,native_depth_sanitization='Original QGS renderer torch.nan_to_num; not added by evaluation'))
   else:
    with np.load(dest) as stored:
     counts={k:int((~np.isfinite(stored[k])).sum()) for k in ['render_raw','render_diffuse','gt_rgb','alpha','depth_expected','depth_surface',
       'native_normal','depth_normal_camera','depth_median_native_raw','depth_native_mesh','alpha_native_mesh','normal_native_mesh_camera']}
    finite_reports.append(dict(source_id=sid,nonfinite_values=counts,existing_actual_output_revalidated=True))
   q.write(out/'numerical_validation.json',dict(status='running',records=finite_reports))
   for field in ['render_raw','render_diffuse','gt_rgb','alpha','depth_expected','depth_surface','depth_native_mesh','alpha_native_mesh']:
    assert counts[field]==0,(sid,field,'Actual native output contains nonfinite metric/fusion values; original NPZ retained')
   manifest.append(dict(source_id=sid,path=str(dest.resolve()),camera=rec,gt_sha256=ref['gt_sha256'],mask_sha256=ref['mask_sha256']))
   q.write(out/'render_index.json',manifest)
   print('E13 native render',sid,flush=True)
 assert len(manifest)==208
 q.write(out/'numerical_validation.json',dict(status='passed',views=208,records=finite_reports,
  scope='Actual finite RGB/GT/alpha/native sanitized depth; weighted native normal and raw median defects are reported unchanged'))
 result=dict(status='complete',experiment='E13',views=208,model_count=len(model.get_xyz),checkpoint=str((q.ROOT/'checkpoints/chkpnt30000.pth').resolve()),
    native_quadratic_ply=str((q.ROOT/'checkpoints/point_cloud/iteration_30000/point_cloud.ply').resolve()),
    normal_coordinates='native_normal camera alpha-weighted native channel; normal_unit_camera normalized; explicit normal_world and normal_unit_world transforms',depth_ratio=float(cfg.pipeline.depth_ratio),
    wall_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,physical_gpu=1,
    RGB='native full SH raw; separate official SH0/-999/stop_z_gradient=True mesh RGB/depth/alpha/normal retained',foreground='original E10 frozen RAP; not inserted into training objective')
 q.write(done,result);print(q.json.dumps(result),flush=True)
if __name__=='__main__':main()

