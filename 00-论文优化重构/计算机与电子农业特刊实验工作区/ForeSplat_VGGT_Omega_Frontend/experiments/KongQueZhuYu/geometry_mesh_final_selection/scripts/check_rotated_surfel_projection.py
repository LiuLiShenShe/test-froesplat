"""Actual nonidentity-pose surfel projection, rays and gradient validation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[1];DUAL=ROOT.parent/'geometry_rgb_dual_frontend'
sys.path.insert(0,str(ROOT/'camera_interface/2dgs_corrected'))
from scene.cameras import Camera
from gaussian_renderer import render
from utils.point_utils import depths_to_points

records=json.loads((DUAL/'render_comparison/E10/camera_runtime.json').read_text())['records'];refs={int(r['source_id']):r for r in records};rows=[]
for sid in [0,53,105]:
    r=refs[sid];E=np.asarray(r['world_view_transform']).T;C=np.linalg.inv(E);w,h=128,96;sx=w/r['width'];sy=h/r['height']
    fx=r['Fx']*sx;fy=r['Fy']*sy;cx=(r['Cx']+.5)*sx-.5;cy=(r['Cy']+.5)*sy-.5
    cam=Camera(0,E[:3,:3].T,E[:3,3],2*np.arctan(w/2/fx),2*np.arctan(h/2/fy),torch.zeros(3,h,w),None,'test',0,fx=fx,fy=fy,cx=cx,cy=cy)
    pc=np.array([.08,-.04,2.]);world=C[:3,:3]@pc+C[:3,3];q=Rotation.from_matrix(C[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
    xyz=torch.tensor(world[None],device='cuda',dtype=torch.float32,requires_grad=True)
    gs=Namespace(get_xyz=xyz,get_opacity=torch.tensor([[.8]],device='cuda'),get_scaling=torch.tensor([[.04,.04]],device='cuda'),get_rotation=torch.tensor(q[None],device='cuda',dtype=torch.float32),get_features=torch.zeros(1,16,3,device='cuda'),active_sh_degree=3,max_sh_degree=3)
    out=render(cam,gs,Namespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False,depth_ratio=0.),torch.zeros(3,device='cuda'))
    a=out['rend_alpha'][0];yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
    centre=torch.stack([(a*xx).sum(),(a*yy).sum()])/a.sum();ideal=torch.tensor([fx*pc[0]/pc[2]+cx,fy*pc[1]/pc[2]+cy],device='cuda');error=float(torch.linalg.norm(centre-ideal));assert error<.1,(sid,error)
    support=a>.05
    expected_error=float((out['depth_expected'][0][support]-2.).abs().max())
    median_error=float((out['depth_median'][0][support]-2.).abs().max())
    assert expected_error<1e-4 and median_error<1e-4,(sid,expected_error,median_error)
    (out['render'][0]*xx).sum().backward();assert torch.isfinite(xyz.grad).all() and xyz.grad.abs().sum()>0
    point=depths_to_points(cam,torch.full((1,h,w),2.,device='cuda')).reshape(h,w,3)[45,71].cpu().numpy();recovered=E[:3,:3]@point+E[:3,3];expected=np.array([(71-cx)*2/fx,(45-cy)*2/fy,2.]);rayerr=float(abs(recovered-expected).max());assert rayerr<1e-5
    rows.append(dict(source_id=sid,actual_cuda_alpha_centroid_error_px=error,depth_ray_camera_error=rayerr,actual_CUDA_expected_depth_error=expected_error,actual_CUDA_median_depth_error=median_error,finite_gradient=True))
(ROOT/'camera_interface/rotated_surfel_validation.json').write_text(json.dumps(dict(status='passed',records=rows,physical_gpu=1),indent=2)+'\n');print(rows)
