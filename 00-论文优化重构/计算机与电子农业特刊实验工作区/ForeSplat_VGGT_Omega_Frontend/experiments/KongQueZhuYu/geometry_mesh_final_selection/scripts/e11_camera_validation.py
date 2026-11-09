import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[1];DUAL=ROOT.parent/'geometry_rgb_dual_frontend'
sys.path.insert(0,str(ROOT/'camera_interface/2dgs_corrected'))
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import loadCam
from scene.cameras import Camera
from utils.point_utils import depths_to_points
from utils.mesh_utils import to_cam_open3d
from gaussian_renderer import render

def record(c):
    d=dict(name=c.image_name,width=c.image_width,height=c.image_height,Fx=c.Fx,Fy=c.Fy,Cx=c.Cx,Cy=c.Cy)
    for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:d[k]=getattr(c,k).detach().cpu().tolist()
    return d

def check_cameras(cameras):
    expected=json.loads((DUAL/'render_comparison/E10/camera_runtime.json').read_text())['records']
    refs={r['name']:r for r in expected};records=[];maxerr=0.;rng=np.random.default_rng(0)
    for c in sorted(cameras,key=lambda c:c.image_name):
        r=refs[c.image_name];assert (c.image_width,c.image_height)==(1080,1920)
        for k in ['Fx','Fy','Cx','Cy']:assert abs(getattr(c,k)-r[k])<1e-8
        for k in ['world_view_transform','projection_matrix','full_proj_transform']:
            assert np.max(np.abs(np.asarray(getattr(c,k).cpu())-r[k]))<1e-7
        pc=np.column_stack([rng.uniform(-.4,.4,64),rng.uniform(-.4,.4,64),rng.uniform(.5,2,64)])
        P=np.asarray(c.projection_matrix.cpu()).T;clip=np.column_stack([pc,np.ones(64)])@P.T
        pix=((clip[:,:2]/clip[:,3,None]+1)*[c.image_width,c.image_height]-1)/2
        ideal=pc[:,:2]/pc[:,2,None]*[c.Fx,c.Fy]+[c.Cx,c.Cy]
        maxerr=max(maxerr,float(np.abs(pix-ideal).max()));records.append(record(c))
    assert len(records)==208 and maxerr<.001
    return {'records':records,'projection_error_max_px':maxerr,'runtime_K_W2C_P_identical_to_E10':True}

def main():
    ds=DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected'
    info=readColmapSceneInfo(str(ds),'images',False)
    args=Namespace(resolution=2,data_device='cuda',mask_dir='',mask_mode='none')
    cameras=[loadCam(args,i,c,1.) for i,c in enumerate(info.train_cameras)]
    result=check_cameras(cameras)
    # Exact training GT parity with E10's recorded floating tensor hashes.
    table=__import__('csv').DictReader((DUAL/'render_comparison/E10/per_view_metrics.csv').open())
    expected={int(r['source_id']):r['gt_sha256'] for r in table}
    for c in cameras:
        rgb=c.original_image.cpu().numpy().transpose(1,2,0)
        sid=int(c.image_name.split('_')[-1]);assert hashlib.sha256(rgb.tobytes()).hexdigest()==expected[sid]
    result['all208_float_GT_hashes_identical_to_E10']=True
    o3=to_cam_open3d(cameras)
    for c,o in zip(cameras,o3):
        assert np.max(np.abs(o.intrinsic.intrinsic_matrix-[[c.Fx,0,c.Cx],[0,c.Fy,c.Cy],[0,0,1]]))<.0003
    # Real surfel rasterizer: noncentral synthetic camera and differentiable plane.
    c=Camera(0,np.eye(3),np.zeros(3),2*np.arctan(128/360),2*np.arctan(96/420),torch.zeros(3,96,128),None,'synthetic',0,fx=180,fy=210,cx=70.3,cy=40.1)
    xyz=torch.tensor([[.08,-.04,2.]],device='cuda',requires_grad=True)
    pc=Namespace(get_xyz=xyz,get_opacity=torch.tensor([[.8]],device='cuda'),get_scaling=torch.tensor([[.03,.03]],device='cuda'),get_rotation=torch.tensor([[1.,0,0,0]],device='cuda'),get_features=torch.zeros(1,16,3,device='cuda'),active_sh_degree=3,max_sh_degree=3)
    pipe=Namespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False,depth_ratio=0.)
    out=render(c,pc,pipe,torch.zeros(3,device='cuda'))
    alpha=out['rend_alpha'][0];yy,xx=torch.meshgrid(torch.arange(96,device='cuda'),torch.arange(128,device='cuda'),indexing='ij')
    center=torch.stack([(alpha*xx).sum(),(alpha*yy).sum()])/alpha.sum()
    ideal=torch.tensor([77.5,35.9],device='cuda');err=float(torch.linalg.norm(center-ideal));assert err<.1
    points=depths_to_points(c,torch.full((1,96,128),2.,device='cuda')).reshape(96,128,3)
    pix=points[35,77];assert abs(float(pix[0])-(77-c.Cx)*2/c.Fx)<1e-6
    (out['render'][0]*xx).sum().backward();assert torch.isfinite(xyz.grad).all() and xyz.grad.abs().sum()>0
    result.update(synthetic_alpha_centroid_error_px=err,synthetic_gradient=xyz.grad.cpu().tolist(),real_surfel_forward_backward_passed=True,depth_unprojection_and_Open3D_K_passed=True,physical_gpu=1,status='passed')
    (ROOT/'camera_interface/E11_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('E11 all208 K/W2C/P/GT match E10; actual surfel forward/backward, normal-ray and mesh camera checks passed')

if __name__=='__main__':main()
