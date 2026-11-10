"""Run official QGS bounded TSDF on its native depth; preserve every mesh stage."""
import qgs_common as q
import time
import numpy as np
import torch,open3d as o3d
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.mesh_utils import GaussianExtractor,post_process_mesh
class LazyMaps:
 def __init__(self,index,key,mask=False,clip_rgb=False):self.index=index;self.key=key;self.mask=mask;self.clip_rgb=clip_rgb
 def __getitem__(self,i):
  with np.load(self.index[i]['path']) as d:
   arr=d[self.key].copy()
   if self.mask:arr[~d['foreground_mask']]=0
   # The official RGB8 conversion casts float*255 directly. Clamp its color
   # input to prevent >1 SH colors wrapping to black; keep original float maps.
   if self.clip_rgb:arr=arr.clip(0,1)
  if arr.ndim==3:arr=arr.transpose(2,0,1)
  return torch.from_numpy(np.ascontiguousarray(arr))
 def __len__(self):return len(self.index)
def main():
 out=q.ROOT/'meshes/native_TSDF';done=out/'complete.json'
 if done.exists():print(done.read_text());return
 assert (q.ROOT/'surface_inputs/render_complete.json').is_file()
 cfg=OmegaConf.load(q.ROOT/'actual_config.yaml');Scene=q.read_only_image_adapter()
 model=GaussianModel(cfg.gs_model);scene=Scene(cfg,model,load_iteration=30000,shuffle=False)
 cameras={int(c.image_name.split('_')[-1]):c for c in scene.getTrainCameras()}
 index=q.json.loads((q.ROOT/'surface_inputs/render_index.json').read_text())
 scale=q.json.loads((q.HIST/'configs/standard_spatial_scale.json').read_text())
 started=time.time()
 extractor=GaussianExtractor(model,render,cfg.pipeline,cfg.optimizer,bg_color=[0,0,0])
 # The official fusion loop receives lazy saved native depth/color maps. It
 # performs the same Open3D TSDF integration without keeping 208 maps in memory.
 extractor.viewpoint_stack=[cameras[int(r['source_id'])] for r in index]
 for c in extractor.viewpoint_stack:c.gt_alpha_mask=None
 extractor.rgbmaps=LazyMaps(index,'render_diffuse',clip_rgb=True)
 extractor.depthmaps=LazyMaps(index,'depth_native_mesh',mask=True)
 mesh=extractor.extract_mesh_bounded(voxel_size=scale['tsdf_voxel'],sdf_trunc=scale['tsdf_sdf_trunc'],depth_trunc=scale['tsdf_depth_trunc'],mask_backgrond=True)
 assert len(mesh.vertices)>0 and len(mesh.triangles)>0
 out.mkdir(parents=True,exist_ok=True)
 mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(out/'raw.ply'),mesh,write_ascii=False)
 # Keep the upstream top-10 post result separately; use the historical <50
 # triangle-only post rule for a directly comparable primary result.
 upstream=post_process_mesh(mesh,cluster_to_keep=10)
 o3d.io.write_triangle_mesh(str(out/'official_top10_post.ply'),upstream,write_ascii=False)
 q.sys.path.insert(0,str(q.HIST/'scripts'))
 from mesh_artifacts import save_meshes
 stats=save_meshes(mesh,out,scale)
 q.write(done,dict(status='complete',experiment='E13',source='native quadratic checkpoint',mesh_algorithm='official QGS GaussianExtractor.extract_mesh_bounded Open3D TSDF',
   depth_ratio=float(cfg.pipeline.depth_ratio),depth_source='native QGS expected/median mixture from official SH0/kernel_size=-999/stop_z_gradient=True mesh rendering',mask='Original shared RAP applied during fusion only',
   color_encoding='Original native SH0 float RGB retained in NPZ; clip[0,1] before official RGB8 conversion avoids uint8 wrap; depth/fusion unchanged',
   scale=scale,elapsed_seconds=time.time()-started,physical_gpu=1,official_post_rule='top10 clusters, preserved as official_top10_post.ply',
   comparable_post_rule='all components >=50 faces; no smoothing or hole fill',**stats))
 print(done.read_text(),flush=True)
if __name__=='__main__':main()

