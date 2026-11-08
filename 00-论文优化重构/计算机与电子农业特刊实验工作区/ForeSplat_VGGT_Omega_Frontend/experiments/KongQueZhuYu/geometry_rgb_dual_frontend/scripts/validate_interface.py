"""Exercise actual GGGS Camera, CUDA RGB/depth/sample/backward paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(8)
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OMEGA=ROOT.parents[2]
p=argparse.ArgumentParser();p.add_argument('--interface',choices=['original','corrected'],required=True);p.add_argument('--synthetic-only',action='store_true');args=p.parse_args()
REPO=OMEGA/'gggs_community' if args.interface=='original' else ROOT/'camera_interface/gggs_corrected'
sys.path.insert(0,str(REPO))
if args.interface=='corrected':sys.path.insert(0,str(REPO/'corrected_rasterizer'))
from utils.cuda_preload import preload_cuda_ext_deps
preload_cuda_ext_deps()
from scene.cameras import Camera
from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
def project(cam,xyz):
    hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],dim=1)@cam.full_proj_transform
    ndc=hom[:,:2]/hom[:,3:4]
    return (ndc+1)*torch.tensor([cam.image_width,cam.image_height],device='cuda')/2-.5
def camera(w,h,fx,fy,cx,cy,e):
    kw=dict(colmap_id=0,R=e[:3,:3].T,T=e[:3,3],FoVx=2*math.atan(w/(2*fx)),FoVy=2*math.atan(h/(2*fy)),image=torch.zeros(3,h,w),gt_alpha_mask=None,image_name='test',uid=0,data_device='cpu')
    if args.interface=='corrected':kw['intrinsics']=(fx,fy,cx,cy)
    return Camera(**kw)
def synthetic():
    torch.manual_seed(0);rng=np.random.default_rng(0);records=[]
    for angle in [0,19,-35]:
        w,h=160,192;fx,fy=180.,210.;cx,cy=87.3,90.1
        e=np.eye(4);e[:3,:3]=Rotation.from_euler('y',angle,degrees=True).as_matrix();e[:3,3]=[.31,-.13,.25]
        cam=camera(w,h,fx,fy,cx,cy,e)
        local=np.c_[rng.uniform(-.3,.3,512),rng.uniform(-.3,.3,512),rng.uniform(1.5,3,512)]
        world=(local-e[:3,3])@e[:3,:3];xyz=torch.tensor(world,dtype=torch.float32,device='cuda')
        expected=local[:,:2]/local[:,2,None]*[fx,fy]+[cx,cy]
        error=np.linalg.norm(project(cam,xyz).cpu().numpy()-expected,axis=1)
        center_local=np.array([[.08,-.04,2.]])
        mean=torch.tensor((center_local-e[:3,3])@e[:3,:3],dtype=torch.float32,device='cuda',requires_grad=True)
        q=Rotation.from_matrix(e[:3,:3].T).as_quat();rot=torch.tensor([[q[3],*q[:3]]],dtype=torch.float32,device='cuda');scale=torch.tensor([[.08,.08,.008]],device='cuda');opacity=torch.tensor([[.99]],device='cuda')
        setting=GaussianRasterizationSettings(image_height=h,image_width=w,tanfovx=w/(2*fx),tanfovy=h/(2*fy),kernel_size=0.,bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,sh_degree=0,sg_degree=0,campos=cam.camera_center,prefiltered=False,require_depth=True,debug=True)
        raster=GaussianRasterizer(setting)
        color,radius,depth,alpha,normal=raster(means3D=mean,means2D=torch.zeros_like(mean,requires_grad=True),opacities=opacity,colors_precomp=torch.ones(1,3,device='cuda'),scales=scale,rotations=rot,sg_axis=torch.empty(1,0,3,device='cuda'),sg_sharpness=torch.empty(1,0,1,device='cuda'),sg_color=torch.empty(1,0,3,device='cuda'))
        sampled,inside=raster.sample_depth(points3D=mean,means3D=mean,opacities=opacity,scales=scale,rotations=rot)
        expected_center=center_local[:,:2]/center_local[:,2,None]*[fx,fy]+[cx,cy]
        actual_sample=sampled.detach().cpu().numpy();sample_uv=actual_sample[:,:2]/actual_sample[:,2,None]*[fx,fy]+[cx,cy]
        yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij');mass=alpha.sum();centroid=torch.stack([(alpha*xx).sum()/mass,(alpha*yy).sum()/mass]).detach().cpu().numpy()
        # Backward executes depth, normal and sampled-depth derivatives together.
        loss=(color*xx).mean()+depth.mean()*.01+sampled.sum()*.01+normal.mean()*.01
        loss.backward()
        assert torch.isfinite(mean.grad).all() and float(mean.grad.abs().sum())>0
        record=dict(rotation_deg=angle,K=[fx,fy,cx,cy],matrix_projection_error_p50_px=float(np.median(error)),matrix_projection_error_max_px=float(error.max()),
            alpha_centroid_error_px=float(np.linalg.norm(centroid-expected_center[0])),sample_ray_reprojection_error_px=float(np.linalg.norm(sample_uv-expected_center)),sample_inside=inside.detach().cpu().tolist(),sample_camera_xyz=actual_sample.tolist(),backward_gradient=mean.grad.cpu().tolist())
        records.append(record)
        if args.interface=='corrected':
            assert record['matrix_projection_error_max_px']<2e-4,record
            assert record['alpha_centroid_error_px']<.15,record
            assert record['sample_ray_reprojection_error_px']<2e-4 and bool(inside.all()),record
    import diff_gaussian_rasterization as extension
    result=dict(interface=args.interface,tests=records,physical_gpu=1,extension_file=str(extension._C.__file__),extension_sha256=hashlib.sha256(Path(extension._C.__file__).read_bytes()).hexdigest(),status='passed' if args.interface=='corrected' else 'baseline_measured')
    (ROOT/'camera_interface'/f'synthetic_{args.interface}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
def actual():
    import csv,pycolmap
    from scene import Scene,GaussianModel
    records=[]
    for model,dirname in [('Omega','VGGT_Omega_Original_RGB'),('Standard','Standard_VGGT_Original_RGB')]:
        out=ROOT/dirname;dataset=out/('COLMAP_Corrected' if args.interface=='corrected' else 'COLMAP_Original');run=ROOT/'camera_interface'/f'runtime_{model}_{args.interface}';run.mkdir(exist_ok=True)
        ns=argparse.Namespace(sh_degree=3,sg_degree=0,source_path=str(dataset),model_path=str(run),images='images',dataset='',resolution=2,white_background=False,data_device='cpu',eval=False,use_decoupled_appearance=0,disable_filter3D=False,kernel_size=0.,multi_view_num=8,multi_view_max_angle=30,multi_view_min_dis=.01,multi_view_max_dis=1.5)
        # Camera-only loader, no initialized Gaussians or optimizer.
        from scene.dataset_readers import readColmapSceneInfo
        from utils.camera_utils import cameraList_from_camInfos
        info=readColmapSceneInfo(str(dataset),'images',False)
        cams=cameraList_from_camInfos(info.train_cameras,1.,ns)
        reference=pycolmap.Reconstruction(str(out/'COLMAP_Corrected/sparse/0'))
        cameras=json.loads((out/'cameras.json').read_text())['frames'];byname={r['image_name']:r for r in cameras}
        samples=[]
        for cam in cams:
            name=cam.image_name+'.png';i=next(im for im in reference.images.values() if im.name==name);kk=reference.cameras[i.camera_id].params.copy();kk[:2]/=2;kk[2:]=(kk[2:]+.5)/2-.5
            rng=np.random.default_rng(int(cam.image_name.split('_')[-1]));xyz=np.c_[rng.uniform(-.4,.4,128),rng.uniform(-.4,.4,128),rng.uniform(1.,3.,128)]
            e=np.asarray(cam.world_view_transform.cpu()).T;world=(xyz-e[:3,3])@e[:3,:3]
            actual_uv=project(cam,torch.tensor(world,dtype=torch.float32,device='cuda')).cpu().numpy();desired=xyz[:,:2]/xyz[:,2,None]*kk[:2]+kk[2:];error=np.linalg.norm(actual_uv-desired,axis=1)
            row=dict(model=model,interface=args.interface,source_id=int(cam.image_name.split('_')[-1]),width=cam.image_width,height=cam.image_height,fx=cam.Fx,fy=cam.Fy,cx=cam.Cx,cy=cam.Cy,desired_fx=kk[0],desired_fy=kk[1],desired_cx=kk[2],desired_cy=kk[3],projection_mean_px=float(error.mean()),projection_max_px=float(error.max()))
            if row['source_id'] in [0,26,53,79,105,131,157,183]:
                from scipy.ndimage import binary_erosion
                mask=np.load(out/'per_frame'/f'model_mask_{cam.image_name}.npy').astype(bool)
                valid=np.load(out/'per_frame'/f'valid_foreground_{cam.image_name}.npy').astype(bool)
                points=np.load(out/'per_frame'/f'world_points_{cam.image_name}.npy')
                alignment=np.array(json.loads((out/'fusion_metadata.json').read_text())['scene_alignment_matrix']) if model=='Omega' else np.eye(4)
                edge=mask&~binary_erosion(mask)
                for label,selection in [('foreground',valid),('edge',valid&edge)]:
                    xx=points[selection]
                    xx=xx[np.linspace(0,len(xx)-1,min(512,len(xx))).astype(int)] if len(xx) else xx
                    if not len(xx):continue
                    xx=xx@alignment[:3,:3].T+alignment[:3,3]
                    loc=xx@e[:3,:3].T+e[:3,3];target=loc[:,:2]/loc[:,2,None]*kk[:2]+kk[2:]
                    actual=project(cam,torch.tensor(xx,dtype=torch.float32,device='cuda')).cpu().numpy();res=np.linalg.norm(actual-target,axis=1)
                    row[label+'_reprojection_p50_px']=float(np.median(res));row[label+'_reprojection_max_px']=float(np.max(res));row[label+'_points']=len(res)
            records.append(row);samples.append(dict(row,w2c=e.tolist(),c2w=np.linalg.inv(e).tolist(),projection_matrix=cam.projection_matrix.cpu().tolist(),full_proj_transform=cam.full_proj_transform.cpu().tolist(),frontend_record=byname[name]))
            if args.interface=='corrected':assert row['projection_max_px']<.002,row
        (ROOT/'camera_interface'/f'{model}_{args.interface}_runtime_parameters.json').write_text(json.dumps(samples,indent=2))
        del cams
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (ROOT/'camera_interface'/f'projection_{args.interface}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    print('actual camera interface checked',args.interface,len(records),flush=True)

def pair_invariants():
    def image_headers(path):
        result={}
        for line in path.read_text().splitlines():
            if not line or line.startswith('#'):continue
            tokens=line.split(maxsplit=10)
            if len(tokens)>=10 and tokens[9].endswith('.png'):
                result[tokens[9]]=np.array(list(map(float,tokens[:9])))
        return result
    def camera_headers(path):
        return {int(t[0]):dict(size=list(map(int,t[2:4])),K=np.array(list(map(float,t[4:8])))) for line in path.read_text().splitlines() if line and not line.startswith('#') for t in [line.split()]}
    source=json.loads((ROOT/'configs/prepared_sources.json').read_text());checks=[]
    for model,dirname in [('Omega','VGGT_Omega_Original_RGB'),('Standard','Standard_VGGT_Original_RGB')]:
        out=ROOT/dirname;old=out/'COLMAP_Original';new=out/'COLMAP_Corrected'
        assert os.path.samefile(old/'sparse/0/points3D.ply',new/'sparse/0/points3D.ply')
        old_images=image_headers(old/'sparse/0/images.txt');new_images=image_headers(new/'sparse/0/images.txt')
        assert set(old_images)==set(new_images)=={r['image_name'] for r in source}
        max_pose_error=0.
        for name in old_images:
            a,b=old_images[name].copy(),new_images[name].copy()
            if np.dot(a[1:5],b[1:5])<0:b[1:5]*=-1
            max_pose_error=max(max_pose_error,float(np.max(np.abs(a-b))))
        assert max_pose_error<1e-8,max_pose_error
        oc=camera_headers(old/'sparse/0/cameras.txt');nc=camera_headers(new/'sparse/0/cameras.txt')
        assert set(oc)==set(nc) and all(oc[k]['size']==nc[k]['size'] for k in oc)
        focal_error=max(float(np.max(np.abs(oc[k]['K'][:2]-nc[k]['K'][:2]))) for k in oc);assert focal_error<1e-8,focal_error
        for row in source:
            name=row['image_name']
            for directory,field in [('images','training_rgb_path'),('masks','mask_path')]:
                assert os.path.samefile(old/directory/name,new/directory/name)
                assert os.path.samefile(old/directory/name,row[field])
        checks.append(dict(model=model,views=208,identical_initialization_file=True,identical_training_RGB_and_masks=True,
            pose_parameter_max_absolute_difference=max_pose_error,focal_max_absolute_difference=focal_error,image_sizes_unchanged=True,
            principal_point_max_absolute_change_original_px=max(float(np.max(np.abs(oc[k]['K'][2:]-nc[k]['K'][2:]))) for k in oc)))
    optimizer_unchanged=hashlib.sha256((OMEGA/'gggs_community/scene/gaussian_model.py').read_bytes()).hexdigest()==hashlib.sha256((ROOT/'camera_interface/gggs_corrected/scene/gaussian_model.py').read_bytes()).hexdigest()
    assert optimizer_unchanged
    (ROOT/'configs/pair_invariants.json').write_text(json.dumps(dict(status='passed',checks=checks,core_optimizer_unchanged=optimizer_unchanged),indent=2))
    print('Both frontend interface pairs preserve point cloud, poses, focal lengths, image sizes, training RGB and masks',flush=True)
if __name__=='__main__':
    synthetic()
    if not args.synthetic_only:actual()
    if not args.synthetic_only and args.interface=='corrected':
        pair_invariants()
        from compare_interface_matches import main as compare_matches
        compare_matches()
