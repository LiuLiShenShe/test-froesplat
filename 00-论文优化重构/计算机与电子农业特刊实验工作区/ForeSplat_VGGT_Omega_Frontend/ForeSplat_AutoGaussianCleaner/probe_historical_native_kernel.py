import os;os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8'
from pathlib import Path
import json,sys,numpy as np,torch
from PIL import Image
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply
from ForeSplat_AutoGaussianCleaner.evaluation.free_view_evaluation import camera
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
root=BASE/'ForeSplat_AutoGaussianCleaner/experiments/Historical_TwoStage_Cleaner/v4/KongQueZhuYu'
r=json.loads((root/'execution.json').read_text())['reference']; config=r['config'];scene=Scene(r['scene'],config);torch.set_num_threads(8)
model,_=scene.load_model(r['scene']['baseline_models']['R5'])
protocol=json.loads((root/'stage2_residual/visualizations/automatic_free_views.json').read_text())
out=root/'comparison/native_kernel_probe';out.mkdir(parents=True,exist_ok=True)
for v in protocol['definitions']:
 if v['name'] not in ('top','orbit_03'):continue
 c=camera(np.asarray(v['K']),np.asarray(v['E']),v['width'],v['height'],v['name'],0)
 for k in [0.,.1,.3,.5,1.0]:
  with torch.no_grad(): a=arrays(scene.renderer.render(c,model,scene.pipe,scene.background,k,require_depth=True))
  rgb=a['rgb']+(1-a['alpha'][...,None]);Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(out/f'{v["name"]}_k{k:g}.png')
print(out)
from scipy.spatial.transform import Rotation
from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
xyz=model.get_xyz.detach();q=model.get_rotation.detach().cpu().numpy();rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();scales,opacity=model.get_scaling_n_opacity_with_3D_filter
s=scales.detach().cpu().numpy();cov=(rot*s[:,None,:])@np.swapaxes(rot*s[:,None,:],1,2)
for v in protocol['definitions']:
 if v['name'] not in ('top','orbit_03'):continue
 E=np.asarray(v['E']);K=np.asarray(v['K']);c=camera(K,E,v['width'],v['height'],v['name'],0)
 p=xyz.cpu().numpy()@E[:3,:3].T+E[:3,3];z=np.maximum(p[:,2],1e-6);u=np.clip(p[:,0]/z,-1.3*np.tan(c.FoVx/2),1.3*np.tan(c.FoVx/2));vv=np.clip(p[:,1]/z,-1.3*np.tan(c.FoVy/2),1.3*np.tan(c.FoVy/2))
 J=np.zeros((len(p),2,3));J[:,0,0]=K[0,0]/z;J[:,1,1]=K[1,1]/z;J[:,0,2]=-K[0,0]*u/z;J[:,1,2]=-K[1,1]*vv/z
 T=J@E[:3,:3];cc=T@cov@np.swapaxes(T,1,2);det=cc[:,0,0]*cc[:,1,1]-cc[:,0,1]**2
 for k in [.3]:
  coef=np.sqrt(np.maximum(det,1e-6)/np.maximum((cc[:,0,0]+k)*(cc[:,1,1]+k)-cc[:,0,1]**2,1e-6));op=opacity/torch.as_tensor(coef[:,None],dtype=torch.float32,device='cuda')
  settings=GaussianRasterizationSettings(image_height=c.image_height,image_width=c.image_width,tanfovx=np.tan(c.FoVx/2),tanfovy=np.tan(c.FoVy/2),kernel_size=k,bg=scene.background,scale_modifier=1.,viewmatrix=c.world_view_transform,projmatrix=c.full_proj_transform,sh_degree=model.active_sh_degree,sg_degree=model.active_sg_degree,campos=c.camera_center,prefiltered=False,require_depth=True,debug=False)
  with torch.no_grad():
   a=GaussianRasterizer(raster_settings=settings)(means3D=xyz,means2D=torch.zeros_like(xyz),shs=model.get_features,sg_axis=model.get_sg_axis,sg_sharpness=model.get_sg_sharpness,sg_color=model.get_sg_color,colors_precomp=None,opacities=op.float(),scales=scales,rotations=model.get_rotation,cov3Ds_precomp=None)
  rgb=a[0].cpu().numpy().transpose(1,2,0)+(1-a[3].cpu().numpy().squeeze()[...,None]);Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(out/f'{v["name"]}_uncompensated.png')
print('Uncompensated native diagnostic complete')
