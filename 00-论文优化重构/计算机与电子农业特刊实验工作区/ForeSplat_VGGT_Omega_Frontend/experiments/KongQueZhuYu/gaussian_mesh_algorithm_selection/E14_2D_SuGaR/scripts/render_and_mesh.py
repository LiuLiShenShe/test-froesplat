"""Preserve intervention-stage renders; fuse identical median-depth TSDF before/after native DBSCAN."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,json,time,hashlib,traceback
from pathlib import Path
from argparse import ArgumentParser
import numpy as np,torch,open3d as o3d
ROOT=Path(__file__).resolve().parents[1];OFF=ROOT/'official/gaussian_splatting_2d';OUT=ROOT/'training'
sys.path.insert(0,str(OFF))
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render
sys.path.append(str(ROOT.parent.parent/'geometry_mesh_final_selection/scripts'))
from surface_protocol import save_npz_fast
from mesh_artifacts import save_meshes
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def camrecord(c):
    r=dict(width=c.image_width,height=c.image_height,Fx=c.Fx,Fy=c.Fy,Cx=c.Cx,Cy=c.Cy)
    for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:r[k]=getattr(c,k).detach().cpu().tolist()
    return r
def fuse(vol,c,depth,mask,rgb,scale):
    d=depth.copy();d[~mask|~np.isfinite(d)|(d<=0)]=0
    color=o3d.geometry.Image(np.ascontiguousarray((rgb.clip(0,1)*255).astype(np.uint8)))
    dep=o3d.geometry.Image(np.ascontiguousarray(d,dtype=np.float32))
    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(color,dep,depth_scale=1.,depth_trunc=scale['tsdf_depth_trunc'],convert_rgb_to_intensity=False)
    K=o3d.camera.PinholeCameraIntrinsic(c.image_width,c.image_height,c.Fx,c.Fy,c.Cx,c.Cy)
    E=c.world_view_transform.T.cpu().numpy().astype(np.float64)
    vol.integrate(rgbd,K,E)
def finish_mesh(vol,folder,stage,scale,start):
    folder.mkdir(parents=True,exist_ok=True)
    result=dict(stage=stage,algorithm='Official bounded Open3D ScalableTSDFVolume; native median depth_ratio1; RAP mask',scale=scale,views=208,post_rule='Same M1–M5 components<50 triangles removed; no smoothing',elapsed_seconds=time.time()-start)
    try:
        mesh=vol.extract_triangle_mesh()
        if not len(mesh.vertices) or not len(mesh.triangles):
            raise ValueError('TSDF extraction returned no triangle surface')
        result.update(save_meshes(mesh,folder,scale))
        result.update(status='complete',raw_mesh=str((folder/'raw.ply').resolve()),post_mesh=str((folder/'post.ply').resolve()) if (folder/'post.ply').is_file() else None)
        if not result['post_faces']:
            result['post_mesh']=None
            result['post_status']='empty_after_native_component_rule'
        else:result['post_status']='complete'
        if stage=='final_30000':
            native=o3d.geometry.TriangleMesh(mesh)
            labels,counts,_=native.cluster_connected_triangles()
            native.remove_triangles_by_mask(np.asarray(labels)!=int(np.argmax(np.asarray(counts))))
            native.remove_unreferenced_vertices()
            path=folder/'native_largest_cluster.ply'
            if not o3d.io.write_triangle_mesh(str(path),native):raise IOError('Native largest-cluster export failed')
            result['native_largest_cluster_mesh']=str(path.resolve())
    except Exception as err:
        result.update(status='failed',reason=str(err),traceback=traceback.format_exc())
        for key,name in [('raw_mesh','raw.ply'),('post_mesh','post.ply')]:
            path=folder/name
            if path.is_file():result[key]=str(path.resolve())
    write(folder/'complete.json',result)
    return result
def main():
    marker=ROOT/'native_render_mesh_complete.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('status')=='complete' and all(r.get('status')=='complete' and r.get('post_status')=='complete' for r in old.get('mesh_results',[])):
            return
    assert (ROOT/'training_complete.json').exists()
    parser=ArgumentParser();mp=ModelParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args(['-s',str(ROOT/'input_colmap'),'-m',str(OUT),'-r','2','--depth_ratio','1.0','--initialization_prior',''])
    model=GaussianModel(3)
    cache_ready=all((ROOT/'stage_renders'/s/'render_complete.json').is_file() for s in ['initial','cluster_before','cluster_after','final_30000'])
    if cache_ready:
        # Camera-only reconstruction from actual, audited runtime matrices avoids
        # loading unused 208 prior arrays and RGB images during cached TSDF recovery.
        sys.path.append(str(ROOT.parent/'scripts'))
        from gggs_runtime import cameras
        cams,_=cameras(ROOT/'runtime_camera_validation.json')
        cams.sort(key=lambda c:int(c.image_name.split('_')[-1]))
        reference=json.loads((ROOT/'stage_renders/final_30000/render_index.json').read_text())
        for c,r in zip(cams,reference):
            assert int(c.image_name.split('_')[-1])==r['source_id']
            actual=camrecord(c)
            for key,value in r['camera'].items():
                assert np.array_equal(np.asarray(actual[key]),np.asarray(value)),key
    else:
        scene=Scene(mp.extract(args),model,load_iteration=30000,shuffle=False)
        cams=scene.getTrainCameras()
    assert len(cams)==208
    pipe=pp.extract(args);bg=torch.zeros(3,device='cuda')
    stages=[('initial',OUT/'point_cloud/iteration_1/point_cloud.ply'),('cluster_before',OUT/'point_cloud/iteration_7000/point_cloud.ply'),('cluster_after',ROOT/'cluster_pruning/after.ply'),('final_30000',OUT/'point_cloud/iteration_30000/point_cloud.ply')]
    scale=json.loads((ROOT.parent.parent/'geometry_mesh_final_selection/configs/standard_spatial_scale.json').read_text())
    started=time.time();mesh_results=[]
    for stage,path in stages:
        folder=ROOT/'stage_renders'/stage;folder.mkdir(parents=True,exist_ok=True)
        model.load_ply(str(path));model.active_sh_degree=0 if stage=='initial' else 3;records=[];stage_started=time.time()
        cached=(folder/'render_complete.json').is_file() and all((folder/f'{c.image_name}.npz').is_file() for c in cams)
        if cached:
            records=json.loads((folder/'render_index.json').read_text())
        vol=None
        if stage!='initial':
            vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=scale['tsdf_voxel'],sdf_trunc=scale['tsdf_sdf_trunc'],color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for c in cams:
            dest=folder/f'{c.image_name}.npz'
            if cached:
                if vol is None:continue
                with np.load(dest) as arrays:
                    depth=arrays['depth_surface'];mask=arrays['foreground_mask']
            else:
                with torch.no_grad():pkg=render(c,model,pipe,bg)
                rgb=pkg['render'].detach().cpu().numpy().transpose(1,2,0)
                alpha=pkg['rend_alpha'].detach().cpu().numpy()[0]
                depth=pkg['surf_depth'].detach().cpu().numpy()[0]
                normal=pkg['rend_normal'].detach().cpu().numpy().transpose(1,2,0)
                gt=c.original_image.detach().cpu().numpy().transpose(1,2,0)
                mask=c.gt_alpha_mask.detach().cpu().numpy()[0]>=.5
                extra={k:pkg[k].detach().cpu().numpy().squeeze() for k in ['depth_expected','depth_median']}
                save_npz_fast(dest,render_raw=rgb,gt_rgb=gt,foreground_mask=mask,alpha=alpha,depth_surface=depth,native_normal=normal,**extra)
                records.append(dict(source_id=int(c.image_name.split('_')[-1]),path=str(dest),camera=camrecord(c),gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
            if vol is not None:
                model.active_sh_degree=0
                with torch.no_grad():mesh_rgb=render(c,model,pipe,bg)['render'].cpu().numpy().transpose(1,2,0)
                model.active_sh_degree=3
                fuse(vol,c,depth,mask,mesh_rgb,scale)
            if not cached:write(folder/'render_index.json',records)
            print('E14 render',stage,c.image_name,flush=True)
        if not cached:write(folder/'render_complete.json',dict(status='complete',views=208,model_count=len(model.get_xyz),source_model=str(path.resolve()),active_sh_degree=model.active_sh_degree,native_representation='2D surfels with two scales',depth='official median depth_ratio1',mask='same nearest-resized RAP mask',elapsed_seconds=time.time()-stage_started))
        if vol is not None:
            meshfolder=ROOT/'initial_mesh' if stage=='final_30000' else ROOT/'cluster_pruning'/(stage+'_mesh')
            result=finish_mesh(vol,meshfolder,stage,scale,stage_started);mesh_results.append(result)
            if stage=='final_30000':write(ROOT/'initial_mesh_complete.json',result)
            print('E14 TSDF',stage,result['status'],flush=True)
    write(ROOT/'native_render_mesh_complete.json',dict(status='complete',native_render_stages=4,views_per_stage=208,mesh_results=mesh_results,elapsed_seconds=time.time()-started,open3d=o3d.__version__,reused_cached_depth=True,attribution='7K pre/post use identical checkpoint iteration, camera/mask and TSDF scale; later30K adds further optimization/densification; failed Open3D0.20 records retained in recovery_open3d019'))
if __name__=='__main__':main()

