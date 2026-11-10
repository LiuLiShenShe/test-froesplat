"""Actual QGS forward/backward and depth-ray checks with a noncentral camera."""
import qgs_common as q
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.depth_utils import depths_to_points
from utils.mesh_utils import to_cam_open3d
def main():
 out=q.ROOT/'camera_validation';out.mkdir(exist_ok=True)
 image=out/'synthetic.png';Image.fromarray(np.zeros((96,128,3),dtype=np.uint8)).save(image)
 ci=SimpleNamespace(R=np.eye(3),T=np.zeros(3),FovX=2*np.arctan(128/360),FovY=2*np.arctan(96/420),image_name='synthetic',
   image_path=str(image),cam_intr=np.array([180.,210.,70.3,40.1]),downsample=1.,use_alpha=False,image_type='iterable',width=128,height=96)
 c=Camera(ci,(128,96),0)
 cfg=q.config();model=GaussianModel(cfg.gs_model)
 model._xyz=torch.nn.Parameter(torch.tensor([[.08,-.04,2.]],device='cuda'))
 model._features_dc=torch.nn.Parameter(torch.zeros((1,1,3),device='cuda'))
 model._features_rest=torch.nn.Parameter(torch.zeros((1,15,3),device='cuda'))
 model._scaling=torch.nn.Parameter(model.scaling_inverse_activation(torch.tensor([[.03,.03,.0001]],device='cuda')))
 model._rotation=torch.nn.Parameter(torch.tensor([[1.,0,0,0]],device='cuda'))
 model._opacity=torch.nn.Parameter(torch.logit(torch.tensor([[.8]],device='cuda')))
 model.active_sh_degree=3
 result=render(c,model,cfg.pipeline,cfg.optimizer,torch.zeros(3,device='cuda'),kernel_size=-1,return_depth=True,return_normal=True)
 alpha=result['render_alpha'].squeeze();yy,xx=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(128,device='cuda'),indexing='ij')
 assert torch.isfinite(alpha).all() and float(alpha.sum())>0
 centroid=torch.stack([(alpha*xx).sum(),(alpha*yy).sum()])/alpha.sum()
 ideal=torch.tensor([77.5,35.9],device='cuda');err=float(torch.linalg.norm(centroid-ideal))
 assert err<.15,(centroid.cpu().tolist(),err)
 points,rays=depths_to_points(c,torch.full((1,96,128),2.,device='cuda'))
 points=points.reshape(96,128,3)
 ray_error=float(abs(points[35,77,0]-(77-c.Cx)*2/c.Fx))
 assert ray_error<1e-6
 (result['render'][0]*xx).sum().backward()
 assert torch.isfinite(model._xyz.grad).all() and model._xyz.grad.abs().sum()>0
 mesh_cam=to_cam_open3d([c])[0]
 K=np.array([[c.Fx,0,c.Cx],[0,c.Fy,c.Cy],[0,0,1]])
 mesh_error=float(abs(mesh_cam.intrinsic.intrinsic_matrix-K).max());assert mesh_error<1e-4
 q.write(q.ROOT/'kernel_validation.json',dict(status='passed',physical_gpu=1,native_quadratic_forward_backward=True,
  noncentral_alpha_centroid_error_px=err,alpha_centroid=centroid.detach().cpu().tolist(),xyz_gradient=model._xyz.grad.cpu().tolist(),
  depth_ray_error=ray_error,Open3D_K_error=mesh_error,signed_quadratic_scaling=model.get_scaling.detach().cpu().tolist()))
 print('QGS native noncentral forward/backward+ray+mesh camera validation passed',err,flush=True)
if __name__=='__main__':main()

