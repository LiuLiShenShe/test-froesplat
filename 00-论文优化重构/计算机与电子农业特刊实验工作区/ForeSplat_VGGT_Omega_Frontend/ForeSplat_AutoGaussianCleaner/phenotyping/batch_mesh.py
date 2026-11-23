"""Existing GGGS native median-depth renderer and unchanged bounded TSDF."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
import fcntl,argparse,resource,subprocess
from types import SimpleNamespace
from unittest.mock import patch

def render_frames(ctx,metric,selected_views=None):
 import torch
 from runtime import install_io
 install_io(ctx.meter);install_scalar_ply_io(ctx.meter);torch.set_num_threads(8);sys.path.insert(0,str(GGGS))
 factor=read(ctx.out/'calibration/scale_calibration.json')['scale_cm_per_unit'] if metric else 1.
 if metric:
  from metric_renderer_units import activate
  write(ctx.out/'mesh/renderer_units.json',activate(factor))
 from scene import Scene,GaussianModel
 from gaussian_renderer import render
 from arguments import ModelParams,PipelineParams,OptimizationParams
 from utils.graphics_utils import getProjectionMatrix
 parser=argparse.ArgumentParser();lp,pp=ModelParams(parser),PipelineParams(parser);OptimizationParams(parser)
 native=ctx.out/'gaussian/original'
 if ctx.plant=='KongQueZhuYu':native=PRE/'gaussian/GGGS'
 config=read(native/'training_configuration.json');args=parser.parse_args(config['cli']);args.source_path=str(ctx.common);args.model_path=str(ctx.out/'mesh/renderer_metadata');Path(args.model_path).mkdir(parents=True,exist_ok=True)
 # Scene wants a load_iteration path; a source symlink preserves original PLY.
 source=ctx.out/'gaussian/metric_gaussians.ply' if metric else ctx.out/'gaussian/original_gaussians.ply';target=Path(args.model_path)/'point_cloud/iteration_30000/point_cloud.ply'
 if not target.exists():link(target,source)
 model=GaussianModel(3,0);scene=Scene(lp.extract(args),model,load_iteration=30000,shuffle=False);pipe=pp.extract(args);background=torch.zeros(3,device='cuda');data,cameras=ctx.cameras(metric);frames=ctx.out/'mesh/surface/frames';frames.mkdir(parents=True,exist_ok=True)
 for camera in sorted(scene.getTrainCameras()+scene.getTestCameras(),key=lambda c:c.image_name):
  name=camera.image_name;path=frames/f'{name}.npz'
  if selected_views is not None and name not in selected_views:continue
  if path.exists():continue
  c=cameras[name];c2w=np.eye(4);c2w[:3,:3]=np.array(c['R']).T;c2w[:3,3]=c['center'];w2c=np.linalg.inv(c2w);camera.T=np.array(c['tvec']);camera.world_view_transform=torch.tensor(w2c,dtype=torch.float32,device='cuda').T;camera.camera_center=torch.tensor(c['center'],device='cuda',dtype=torch.float32);camera.znear=.01*factor;camera.zfar=100*factor;camera.projection_matrix=getProjectionMatrix(znear=camera.znear,zfar=camera.zfar,fovX=camera.FoVx,fovY=camera.FoVy).T.cuda();camera.full_proj_transform=camera.world_view_transform@camera.projection_matrix
  with torch.inference_mode():result=render(camera,model,pipe,background,args.kernel_size)
  z=result['median_depth'].squeeze().cpu().numpy();alpha=result['mask'].squeeze().cpu().numpy();rgb=(result['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8');mask=ctx.mask(name);valid=np.isfinite(z)&(z>0)&(alpha>=.5)&mask;npz(path,ctx.meter,depth=np.where(valid,z,0).astype('float32'),K=np.array(data['K']),c2w=c2w,rgb=rgb)
  if name in sorted(cameras)[::max(1,len(cameras)//4)]:png(ctx.out/'gaussian'/f'{name}_render.png',rgb,ctx.meter)
  print('METRIC_RENDER',ctx.plant,name,'valid',int(valid.sum()),flush=True)

def main(ctx,fuse_only=False,*,fuse_worker=None):
 marker=ctx.out/'mesh/complete.json'
 if marker.exists():return
 ctx.status('tsdf','running');cal=read(ctx.out/'calibration/scale_calibration.json');metric=has_metric_scale(cal);unit='cm' if metric else 'model_units';wall=time.time();cpu=time.process_time();folder=ctx.out/'mesh/surface';folder.mkdir(parents=True,exist_ok=True);frames=folder/'frames';frames.mkdir(exist_ok=True)
 if not fuse_only:
  lock=None
  if ctx.plant!='KongQueZhuYu':lock=(PRE/'summary/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
  if ctx.plant=='KongQueZhuYu':
   s=cal['scale_cm_per_unit'] if metric else 1.
   for frame in sorted((PRE/'mesh/GGGS_surface/frames').glob('*.npz')):
    target=frames/frame.name
    if target.exists():continue
    with np.load(io.BytesIO(ctx.meter.read(frame))) as d:z=d['depth']*s;K=d['K'];c2w=d['c2w'].copy();rgb=d['rgb']
    c2w[:3,3]*=s;npz(target,ctx.meter,depth=z.astype('float32'),K=K,c2w=c2w,rgb=rgb)
  else:
   render_frames(ctx,metric)
   import gc,torch
   gc.collect();torch.cuda.empty_cache()
  if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  write(ctx.out/'mesh/render_complete.json',dict(metric=metric,frames=len(list(frames.glob('*.npz'))),wall_s=time.time()-wall,io=ctx.meter.stats()))
  # Reuse the existing TSDF environment, which owns its optional media deps.
  # The GPU renderer process has released GPU1 before CPU-only integration.
  worker=fuse_worker if fuse_worker is not None else [str(CODE/'batch_mesh.py'),ctx.plant,'--fuse-only']
  subprocess.run([str(UV),'run','--no-project','--python',str(ROOT/'2d-gaussian-splatting-main/venv/bin/python'),*worker],check=True,cwd=ROOT);return
 data,cameras=ctx.cameras(metric);K=np.array(data['K']);train={Path(n).stem for n in (ctx.common/'train_views.txt').read_text().splitlines()};samples=[]
 sys.path.insert(0,str(FRONT/'experiments/KongQueZhuYu/geometry_mesh_final_selection/scripts'));from surface_protocol import normals_from_grid
 for frame in sorted(frames.glob('*.npz')):
  if frame.stem not in train:continue
  with np.load(io.BytesIO(ctx.meter.read(frame))) as d:z=d['depth'];c2w=d['c2w']
  yy,xx=np.mgrid[:z.shape[0],:z.shape[1]];xyz=np.stack([(xx+.5-K[0,2])*z/K[0,0],(yy+.5-K[1,2])*z/K[1,1],z],-1)@c2w[:3,:3].T+c2w[:3,3];normals,normal_valid=normals_from_grid(xyz,z,z>0,c2w[:3,3]);yy,xx=np.where((z>0)&normal_valid)
  if len(xx):
   ids=np.linspace(0,len(xx)-1,min(4096,len(xx)),dtype=int);y,x=yy[ids],xx[ids];samples.append(xyz[y,x].astype('float32'))
 if not samples:raise RuntimeError('No valid nativeGGGS target depth')
 low,high=np.quantile(np.concatenate(samples),[.005,.995],axis=0);diagonal=float(np.linalg.norm(high-low));voxel=diagonal/512
 scale=dict(diagonal=diagonal,origin=((low+high)/2).tolist(),sampling_voxel=diagonal/1024,nksr_voxel=voxel,robust_bbox_min=low.tolist(),robust_bbox_max=high.tolist(),unit=unit)
 if ctx.plant=='KongQueZhuYu':
  historical=read(PRE/'mesh/GGGS_surface/spatial_scale.json');s=cal['scale_cm_per_unit'] if metric else 1.;scale={key:(np.array(value)*s).tolist() if isinstance(value,list) else value*s for key,value in historical.items()};scale['unit']=unit;scale['reused_resolution_source']=str(PRE/'mesh/GGGS_surface/spatial_scale.json');voxel=scale['nksr_voxel']
 write(folder/'spatial_scale.json',scale)
 # Same official boundedTSDF integration method and camera half-pixel mapping.
 import open3d as o3d
 import torch
 # Native GGGS and official2DGS both name their package `utils`.
 # Rendering has finished; resolve the frozen TSDF package explicitly.
 for module in list(sys.modules):
  if module=='utils' or module.startswith('utils.'):del sys.modules[module]
 sys.path.insert(0,str(PRE/'sources/2dgs_official'));from utils import mesh_utils as bounded
 selected=[p for p in sorted(frames.glob('*.npz')) if p.stem in train];parameters=[];maximum=0.
 for path in selected:
  with np.load(io.BytesIO(ctx.meter.read(path))) as d:z=d['depth'];c2w=d['c2w'];K=d['K']
  h,w=z.shape;maximum=max(maximum,float(z.max()));c=o3d.camera.PinholeCameraParameters();c.intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2]-.5,K[1,2]-.5);c.extrinsic=np.linalg.inv(c2w);parameters.append(c)
 class Cache:
  index=None;value=None
  def get(self,index):
   if self.index!=index:
    with np.load(io.BytesIO(ctx.meter.read(selected[index]))) as d:self.value={k:d[k] for k in ['depth','rgb']}
    self.index=index
   return self.value
 cache=Cache()
 class Maps:
  def __init__(self,key):self.key=key
  def __getitem__(self,index):
   a=cache.get(index)[self.key];return torch.from_numpy(a).permute(2,0,1).float()/255 if self.key=='rgb' else torch.from_numpy(a[None])
 adapter=SimpleNamespace(viewpoint_stack=[SimpleNamespace(gt_alpha_mask=None) for _ in selected],rgbmaps=Maps('rgb'),depthmaps=Maps('depth'))
 with patch.object(bounded,'to_cam_open3d',lambda unused:parameters):mesh=bounded.GaussianExtractor.extract_mesh_bounded(adapter,voxel_size=voxel,sdf_trunc=5*voxel,depth_trunc=max(maximum*1.01,1e-3),mask_backgrond=False)
 source_gaussian=ctx.out/'gaussian'/('metric_gaussians.ply' if metric else 'original_gaussians.ply');write(ctx.out/'mesh/input_index.json',dict(gaussian=signature(source_gaussian),cameras=signature(ctx.out/'gaussian/metric_cameras.json' if metric else ctx.common/'cameras.json'),training_views=sorted(train),heldout_views=[Path(name).stem for name in (ctx.common/'test_views.txt').read_text().splitlines()],integration_frames=[signature(p) for p in selected],all_rendered_frames=[signature(p) for p in sorted(frames.glob('*.npz'))],length_unit=unit,voxel_size=voxel,sdf_trunc=5*voxel,depth_trunc=max(maximum*1.01,1e-3),depth_scale=1.,mask_backgrond=False,source_file=str(PRE/'sources/2dgs_official/utils/mesh_utils.py'),method='GaussianExtractor.extract_mesh_bounded',frame_rule='One nativeGGGS median cameraZ depth per registered view; alpha>=.5 and archived or replacement SAM3 pottedplant envelope; training views only integrated',extent_rule='Existing normals_from_grid and 4096 evenly spaced valid samples/training view;0.5/99.5XYZ quantiles; scene diagonal/512. KongQue reuses exact selected scene extent scaled tocm.'))
 mesh.compute_vertex_normals();ctx.save_mesh(ctx.out/'mesh/raw_mesh.ply',mesh);components,counts,areas=mesh.cluster_connected_triangles();remove=np.array(counts)[np.array(components)]<50;retained=np.flatnonzero(~remove);npz(ctx.out/'mesh/source_indices.npz',ctx.meter,post_original_raw_face_ids=retained);mesh.remove_triangles_by_mask(remove);mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();target=ctx.out/'mesh'/('mesh_cm.ply' if metric else 'mesh_units.ply');ctx.save_mesh(target,mesh)
 result=dict(status='complete',unit=unit,metric=metric,mesh=str(target),vertices=len(mesh.vertices),faces=len(mesh.triangles),frames=len(selected),cached_views=len(cameras),voxel_length=voxel,sdf_trunc=5*voxel,depth_scale=1.,implementation='Unmodified official2DGS GaussianExtractor.extract_mesh_bounded reused from selected baseline adapter',relative_scene_resolution='robust0.5/99.5pct scene diagonal/512;5voxel SDF truncation',post='Only remove components<50 triangles, no smoothing/closing/holefill',gaussian_source=str(ctx.out/'gaussian'/('metric_gaussians.ply' if metric else 'original_gaussians.ply')),metric_source='Scaled existing native depth/camera is exact uniform equivariance' if ctx.plant=='KongQueZhuYu' else 'Native median cameraZ rendered from metricGaussian and metriccameras' if metric else 'Native model-unit Gaussian depth; centimetres unavailable',applied_scale_count=1 if metric else 0,subsequent_trait_scale=1,wall_s=time.time()-wall,cpu_s=time.process_time()-cpu,io=ctx.meter.stats());write(marker,result);ctx.status('tsdf','complete',metric=metric,vertices=len(mesh.vertices),faces=len(mesh.triangles));print('BOUNDED_TSDF_COMPLETE',ctx.plant,len(mesh.triangles),unit,flush=True)

if __name__=='__main__':
 main(Context(sys.argv[1]),'--fuse-only' in sys.argv);sys.stdout.flush();sys.stderr.flush();os._exit(0)
