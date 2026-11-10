"""Native fixed-eight and deterministic shared orbit renderings, with truthful channel availability."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,json,time,argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from PIL import Image
from matplotlib import colormaps
ROOT=Path(__file__).resolve().parents[1];OFF=ROOT/'official'
p=argparse.ArgumentParser();p.add_argument('--method',choices=['surfel','refined'],required=True);p.add_argument('--protocol',type=Path,default=None);p.add_argument('--out',type=Path,default=None);args=p.parse_args()
sys.path.insert(0,str(OFF))
if args.method=='surfel':
    sys.path.insert(0,str(OFF/'gaussian_splatting_2d'))
    from scene import GaussianModel
    from gaussian_renderer import render
else:
    sys.path.append(str(OFF/'gaussian_splatting'))
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from utils.general_utils import build_rotation
sys.path.append(str(ROOT.parent.parent/'geometry_mesh_final_selection/scripts'))
from surface_protocol import save_npz_fast
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def png(path,a):
    Image.fromarray((np.clip(a,0,1)*255).round().astype(np.uint8)).save(path)
def camera(record):
    # Exact shared float32 matrices/centre are consumed without recomputing orbit poses.
    h,w=record['height'],record['width']
    W=torch.tensor(record['world_view_transform'],dtype=torch.float32,device='cuda')
    projection=torch.tensor(record['projection_matrix'],dtype=torch.float32,device='cuda')
    full=torch.tensor(record['full_proj_transform'],dtype=torch.float32,device='cuda')
    center=torch.tensor(record['camera_center'],dtype=torch.float32,device='cuda')
    E=W.T.cpu().numpy()
    return SimpleNamespace(image_width=w,image_height=h,Fx=record['Fx'],Fy=record['Fy'],Cx=record['Cx'],Cy=record['Cy'],znear=.01,zfar=100.,FoVx=2*np.arctan(w/(2*record['Fx'])),FoVy=2*np.arctan(h/(2*record['Fy'])),R=E[:3,:3].T,T=E[:3,3],world_view_transform=W,projection_matrix=projection,full_proj_transform=full,camera_center=center)
def main():
    folder=args.out or ROOT/'visual_views'/(args.method if args.protocol is None else args.method+'_'+args.protocol.stem);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    dual=ROOT.parent.parent/'geometry_rgb_dual_frontend'
    refs={int(x['name'].split('_')[-1]):x for x in json.loads((dual/'render_comparison/E10/camera_runtime.json').read_text())['records']}
    scale=json.loads((ROOT.parent.parent/'geometry_mesh_final_selection/configs/standard_spatial_scale.json').read_text())
    shared_path=args.protocol or ROOT.parent/'comparison/shared_free_views.json'
    shared=json.loads(shared_path.read_text())
    assert len(shared['views'])>0 and float(shared['background'])==.25
    protocol=dict(fixed_views=[0,26,53,79,105,131,157,183] if args.protocol is None else [],free_views=[v['name'] for v in shared['views']],shared_free_views_path=str(shared_path.resolve()),shared_free_views_sha256=__import__('hashlib').sha256(shared_path.read_bytes()).hexdigest(),world_axis=shared.get('axis','declared exact inspection pose'),pivot=shared.get('pivot',scale['origin']),viewport=[1080,1920],rgb_background=[.25,.25,.25],pose_formula='Consume exact shared WVT/PVM/center float32 matrices; no independent orbit recomputation',mesh_depth_colour_range=[0,scale['tsdf_depth_trunc']])
    write(ROOT/'visual_view_protocol.json' if args.protocol is None else folder/'protocol.json',protocol)
    if args.method=='surfel':
        stages=[('final_30000',ROOT/'training/point_cloud/iteration_30000/point_cloud.ply'),('cluster_after',ROOT/'cluster_pruning/after.ply')]
    else:
        stages=[('refined',Path(json.loads((ROOT/'refinement_complete.json').read_text())['gaussian_ply']))]
    model=GaussianModel(3);pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,depth_ratio=1.0)
    gray=torch.full((3,),.25,device='cuda');black=torch.zeros(3,device='cuda');started=time.time();records=[]
    for stage,ply in stages:
        model.load_ply(str(ply))
        if args.method=='refined':
            rot=build_rotation(model.get_rotation);axis=model.get_scaling.argmin(1)
            axisnormal=rot[torch.arange(len(rot),device='cuda'),:,axis]
        views=[(f'fixed_{sid:04d}',sid,refs[sid]) for sid in protocol['fixed_views']]
        views.extend((r['name'],r['source_id'],r) for r in shared['views'])
        for name,sid,record in views:
            c=camera(record);E=np.asarray(record['world_view_transform'],dtype=np.float32).T
            out=folder/stage/name;out.mkdir(parents=True,exist_ok=True)
            with torch.no_grad():
                package=render(c,model,pipe,gray)
                rgb=package['render'].cpu().numpy().transpose(1,2,0)
                if args.method=='surfel':
                    alpha=package['rend_alpha'].cpu().numpy()[0]
                    depth=package['surf_depth'].cpu().numpy()[0]
                    normal=package['rend_normal'].cpu().numpy().transpose(1,2,0)
                    arrays=dict(render_raw=rgb,alpha=alpha,depth_surface=depth,native_normal=normal)
                    available=dict(rgb='native2DGS graybg',alpha='native2DGS',depth='native median depth_ratio1',normal='native2DGS world normal')
                else:
                    z=(model.get_xyz@torch.tensor(c.R,dtype=torch.float32,device='cuda')+torch.tensor(c.T,dtype=torch.float32,device='cuda'))[:,2]
                    moments=render(c,model,pipe,black,override_color=torch.stack([torch.ones_like(z),z,z*z],dim=1))['render']
                    alpha_t=moments[0];depth_t=moments[1]/alpha_t.clamp_min(1e-8)
                    n=axisnormal.clone();n[((c.camera_center[None]-model.get_xyz)*n).sum(1)<0]*=-1
                    normal_t=render(c,model,pipe,black,override_color=(n+1)/2)['render']/alpha_t.clamp_min(1e-8)[None]*2-1
                    normal_t=torch.nn.functional.normalize(normal_t,dim=0);normal_t[:,alpha_t<1e-8]=0
                    alpha=alpha_t.cpu().numpy();depth=depth_t.cpu().numpy();normal=normal_t.cpu().numpy().transpose(1,2,0)
                    arrays=dict(render_raw=rgb,alpha=alpha,depth_expected_center_aux=depth,normal_smallest_axis_aux=normal)
                    available=dict(rgb='native3DGS graybg on actual SuGaR export',alpha='exact native colour1 channel on blackbg',native_depth='unavailable original3DGS API',native_normal='unavailable original3DGS API',aux_depth='expected center-Z via native override-colour /alpha; not surface median',aux_normal='smallest-scale quaternion axis via native override-colour /alpha; not surface normal')
            png(out/'rgb_gray.png',rgb);png(out/'alpha.png',alpha)
            depvis=colormaps['turbo'](np.clip(depth/scale['tsdf_depth_trunc'],0,1))[...,:3]
            depvis[alpha<.01]=0;normalvis=(normal+1)/2;normalvis[alpha<.01]=0
            suffix='' if args.method=='surfel' else '_aux'
            png(out/f'depth{suffix}.png',depvis);png(out/f'normal{suffix}.png',normalvis)
            save_npz_fast(out/'channels.npz',**arrays)
            write(out/'availability_and_pose.json',dict(availability=available,source_model=str(ply),base_source_id=sid,W2C=E.tolist(),world_view_transform=record['world_view_transform'],projection_matrix=record['projection_matrix'],full_proj_transform=record['full_proj_transform'],camera_center=record['camera_center'],intrinsics=[c.Fx,c.Fy,c.Cx,c.Cy],normal_display='world coordinate RGB (n+1)/2',depth_display_range=[0,scale['tsdf_depth_trunc']]))
            records.append(dict(stage=stage,view=name,directory=str(out),source_model=str(ply)))
            print('E14 native visual',stage,name,flush=True)
    write(folder/'complete.json',dict(status='complete',views=records,elapsed_seconds=time.time()-started,protocol=protocol))
if __name__=='__main__':main()

