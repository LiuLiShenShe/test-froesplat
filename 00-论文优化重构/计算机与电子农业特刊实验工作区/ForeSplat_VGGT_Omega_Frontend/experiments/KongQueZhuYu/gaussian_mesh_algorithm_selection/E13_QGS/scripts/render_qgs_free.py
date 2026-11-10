"""Native QGS free views plus unchanged fixed-pixel legacy leaf/edge windows."""
import qgs_common as q
import argparse,time
import numpy as np,torch
from PIL import Image
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from scene.dataset_readers import readColmapCameras
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary
from gaussian_renderer import render
from render_qgs import save_npz
def png(path,a):
 assert np.isfinite(a).all(),(str(path),'Nonfinite display values; no silent uint8 conversion')
 Image.fromarray((np.clip(a,0,1)*255).round().astype(np.uint8)).save(path)
def fixed_windows():
 index=q.json.loads((q.ROOT/'surface_inputs/render_index.json').read_text());byid={r['source_id']:r for r in index}
 texture=q.json.loads((q.DUAL/'render_comparison/visualizations/GT_texture_crop_protocol.json').read_text())
 edge=q.json.loads((q.HIST/'comparisons/visualizations/leaf_boundary_roi_protocol.json').read_text())
 centres=q.json.loads((q.HIST/'comparisons/visualizations/protocol.json').read_text())['depth_display_centres']
 scale=q.json.loads((q.HIST/'configs/standard_spatial_scale.json').read_text())
 out=q.ROOT/'visualizations/fixed_windows';out.mkdir(parents=True,exist_ok=True);rows=[]
 for label,protocol in [('leaf_texture',texture),('leaf_edge',edge)]:
  for r in protocol['records']:
   sid=r['source_id'];x0,y0,x1,y1=r['bounds_native']
   if label=='leaf_texture':x0+=64;y0+=64;x1-=64;y1-=64
   assert x1-x0==192 and y1-y0==192
   with np.load(byid[sid]['path']) as d:
    rgb=d['render_raw'][y0:y1,x0:x1];gt=d['gt_rgb'][y0:y1,x0:x1]
    alpha=d['alpha'][y0:y1,x0:x1];depth=d['depth_surface'][y0:y1,x0:x1]
    normal=d['normal_unit_camera'][y0:y1,x0:x1]
   prefix=f'{label}_source_{sid:04d}'
   png(out/f'{prefix}_rgb.png',rgb);png(out/f'{prefix}_gt.png',gt);png(out/f'{prefix}_alpha.png',alpha)
   png(out/f'{prefix}_normal_camera.png',normal*.5+.5)
   depth_display=(depth-centres[str(sid)]['standard'])/scale['diagonal']+.5
   depth_display[alpha<.01]=0;png(out/f'{prefix}_depth_display.png',depth_display)
   np.save(out/f'{prefix}_depth.npy',depth)
   rows.append(dict(source_id=sid,window=label,bounds_native=[x0,y0,x1,y1],native_pixels_preserved=True,
     RGB=str((out/f'{prefix}_rgb.png').resolve()),GT=str((out/f'{prefix}_gt.png').resolve()),depth=str((out/f'{prefix}_depth.npy').resolve())))
 q.write(out/'INDEX.json',dict(status='complete',views=8,windows=rows,texture_protocol=str(q.DUAL/'render_comparison/visualizations/GT_texture_crop_protocol.json'),
    boundary_protocol=str(q.HIST/'comparisons/visualizations/leaf_boundary_roi_protocol.json'),normal_coordinates='camera',
    depth_display='same historical standard foreground depth centres and diagonal; no per-output recentering; original float crop separately preserved'))
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--protocol',type=q.Path,default=q.ROOT.parent/'comparison/shared_free_views.json')
 parser.add_argument('--out',type=q.Path,default=q.ROOT/'free_views')
 parser.add_argument('--skip-fixed-windows',action='store_true')
 args=parser.parse_args()
 out=args.out.resolve();done=out/'complete.json'
 if done.exists():return
 cfg=OmegaConf.load(q.ROOT/'actual_config.yaml')
 model=GaussianModel(cfg.gs_model);model.load_ply(str(q.ROOT/'checkpoints/point_cloud/iteration_30000/point_cloud.ply'))
 model.active_sh_degree=model.max_sh_degree
 ds=q.Path(cfg.root_dir);ex=read_extrinsics_binary(str(ds/'sparse/0/images.bin'));intr=read_intrinsics_binary(str(ds/'sparse/0/cameras.bin'))
 infos={int(c.image_name.split('_')[-1]):c for c in readColmapCameras(ex,intr,str(ds/'images'),cfg.dataset)}
 protocol_path=args.protocol.resolve()
 protocol=q.json.loads(protocol_path.read_text())
 out.mkdir(parents=True,exist_ok=True);manifest=[];bg=torch.full((3,),float(protocol['background']),device='cuda');start=time.time()
 with torch.no_grad():
  for r in protocol['views']:
   sid=r['source_id'];c=Camera(infos[sid],(r['width'],r['height']),uid=sid)
   c.world_view_transform=torch.tensor(r['world_view_transform'],dtype=torch.float32,device='cuda')
   c.projection_matrix=torch.tensor(r['projection_matrix'],dtype=torch.float32,device='cuda')
   c.full_proj_transform=torch.tensor(r['full_proj_transform'],dtype=torch.float32,device='cuda')
   for field in ['Fx','Fy','Cx','Cy']:assert abs(getattr(c,field)-r[field])<1e-8
   c.camera_center=torch.tensor(r['camera_center'],dtype=torch.float32,device='cuda')
   w2c=np.array(r['world_view_transform']).T;c.R=w2c[:3,:3].T;c.T=w2c[:3,3]
   pkg=render(c,model,cfg.pipeline,cfg.optimizer,bg,kernel_size=cfg.dataset.kernel_size,return_depth=True,return_normal=True)
   rgb=pkg['render'].cpu().numpy().transpose(1,2,0);alpha=pkg['render_alpha'].cpu().numpy().squeeze()
   depth=pkg['surf_depth'].cpu().numpy().squeeze();normal=pkg['render_normal'].cpu().numpy().transpose(1,2,0)
   nl=np.linalg.norm(normal,axis=-1,keepdims=True);normal_unit=np.divide(normal,nl,out=np.zeros_like(normal),where=nl>1e-12)
   name=r['name'].replace('+','p').replace('-','m');dest=out/f'{name}.npz'
   save_npz(dest,render_raw=rgb,alpha=alpha,depth_surface=depth,native_normal=normal,normal_unit_camera=normal_unit)
   counts={k:int((~np.isfinite(a)).sum()) for k,a in dict(render_raw=rgb,alpha=alpha,depth_surface=depth,native_normal=normal).items()}
   assert all(counts[k]==0 for k in ['render_raw','alpha','depth_surface']),(r['name'],'Nonfinite native RGB/alpha/depth; raw NPZ retained')
   png(out/f'{name}_rgb_gray_background.png',rgb);png(out/f'{name}_alpha.png',alpha);png(out/f'{name}_normal_camera.png',normal_unit*.5+.5)
   manifest.append(dict(name=r['name'],path=str(dest.resolve()),camera=r,source_id=sid,free_view_no_GT=True,
    native_nonfinite_value_counts=counts,raw_native_normal_retained=True))
   print('Native QGS deterministic free view',r['name'],flush=True)
 make_fixed=(not args.skip_fixed_windows) and protocol_path.name=='shared_free_views.json'
 if make_fixed:fixed_windows()
 if protocol_path.name=='shared_free_views.json':assert len(manifest)==8
 q.write(out/'render_index.json',manifest)
 q.write(done,dict(status='complete',native_model=str((q.ROOT/'checkpoints/point_cloud/iteration_30000/point_cloud.ply').resolve()),views=len(manifest),wall_seconds=time.time()-start,
  protocol=str(protocol_path.resolve()),protocol_sha256=q.hashlib.sha256(protocol_path.read_bytes()).hexdigest(),RGB_background=[float(protocol['background'])]*3,normal_coordinates='camera',fixed_legacy_leaf_edge_windows_preserved=make_fixed))
if __name__=='__main__':main()

