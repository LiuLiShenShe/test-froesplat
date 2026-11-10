"""Render the actual exported refined Gaussian model with native 3DGS renderer and explicit moments."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,time,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];OFF=ROOT/'official'
sys.path.insert(0,str(OFF));sys.path.append(str(OFF/'gaussian_splatting'))
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from utils.general_utils import build_rotation
from sugar_scene.cameras import load_gs_cameras
sys.path.append(str(ROOT.parent.parent/'geometry_mesh_final_selection/scripts'))
from surface_protocol import save_npz_fast
from PIL import Image
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    folder=ROOT/'stage_renders/refined'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'render_complete.json').exists():return
    done=json.loads((ROOT/'refinement_complete.json').read_text())
    model=GaussianModel(3);model.load_ply(done['gaussian_ply'])
    cams=load_gs_cameras(str(ROOT/'input_colmap'),str(ROOT/'training'),load_gt_images=True)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');started=time.time();records=[]
    rotations=build_rotation(model.get_rotation)
    axis=model.get_scaling.argmin(1)
    normals=rotations[torch.arange(len(rotations),device='cuda'),:,axis]
    assert len(cams)==208
    for c in cams:
        with torch.no_grad():
            rgb=render(c,model,pipe,bg)['render'].cpu().numpy().transpose(1,2,0)
            pc=model.get_xyz@torch.tensor(c.R,dtype=torch.float32,device='cuda')+torch.tensor(c.T,dtype=torch.float32,device='cuda')
            z=pc[:,2]
            moments=render(c,model,pipe,bg,override_color=torch.stack([torch.ones_like(z),z,z*z],dim=1))['render']
            alpha=moments[0];depth=moments[1]/alpha.clamp_min(1e-8)
            oriented=normals.clone()
            flip=((c.camera_center[None]-model.get_xyz)*oriented).sum(1)<0
            oriented[flip]*=-1
            avg=render(c,model,pipe,bg,override_color=(oriented+1)/2)['render']/alpha.clamp_min(1e-8)[None]*2-1
            avg=torch.nn.functional.normalize(avg,dim=0)
            avg[:,alpha<1e-8]=0
        gt=c.original_image.cpu().numpy().transpose(1,2,0)
        mask=np.asarray(Image.open(ROOT/'input_colmap/masks'/f'{c.image_name}.png').convert('L').resize((c.image_width,c.image_height),Image.Resampling.NEAREST))>127
        rawalpha=alpha.cpu().numpy();rawdepth=depth.cpu().numpy();rawnormal=avg.cpu().numpy().transpose(1,2,0)
        dest=folder/f'{c.image_name}.npz'
        save_npz_fast(dest,render_raw=rgb,gt_rgb=gt,foreground_mask=mask,alpha=rawalpha,depth_surface=np.zeros_like(rawdepth),native_normal=np.zeros_like(rawnormal),depth_expected_center_aux=rawdepth,normal_smallest_axis_aux=rawnormal)
        camera=dict(width=c.image_width,height=c.image_height,Fx=c.Fx,Fy=c.Fy,Cx=c.Cx,Cy=c.Cy)
        for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:camera[k]=getattr(c,k).cpu().tolist()
        records.append(dict(source_id=int(c.image_name.split('_')[-1]),path=str(dest),camera=camera,gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        write(folder/'render_index.json',records)
        print('E14 refined',c.image_name,flush=True)
    write(folder/'render_complete.json',dict(status='complete',views=208,source_model=done['gaussian_ply'],model_count=len(model.get_xyz),representation='Native SuGaR exported3D Gaussian model',depth='Unavailable in original3DGS native rasterizer; primary depth_surface zero sentinel is not a measurement',normal='Unavailable in original3DGS native rasterizer; primary native_normal zero sentinel is not a measurement',auxiliary_depth='depth_expected_center_aux: alpha-normalized native override-colour moment rendering of center-Z; not native2DGSmedian surface depth',auxiliary_normal='normal_smallest_axis_aux: alpha-normalized native override-colour rendering of quaternion smallest-scale axes; not native surface normal',elapsed_seconds=time.time()-started))
if __name__=='__main__':main()

