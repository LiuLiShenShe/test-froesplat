"""Execute the untouched local GaussianExtractor, redirecting all new files."""
from common import *
import argparse, time, traceback, resource
from argparse import Namespace

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample',choices=SCENES,required=True)
    p.add_argument('--method',choices=['M0','M1','M2'],required=True)
    args=p.parse_args();gpu_idle();verify_frozen();import_paths()
    import torch,numpy as np,open3d as o3d
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    from utils.mesh_utils import GaussianExtractor,post_process_mesh
    from mesh_metrics import structure
    output=DATA/'meshes'/args.sample/args.method/'main'
    assert not output.exists(),'Never replace mesh success or failure evidence'
    output.mkdir(parents=True)
    c=config(args.sample);c['model']['model_path']=str(model_path(args.sample,args.method))
    mc=read_json(STAGE/f'configs/{args.sample}_MESH.json')
    label=f'{args.sample}_{args.method}'
    run={'sample':args.sample,'method':args.method,'start_time':now(),'GPU_UUID':GPU_UUID,
         'command':sys.argv,'mesh_protocol_sha256':sha(STAGE/f'configs/{args.sample}_MESH.json'),
         'model_path':c['model']['model_path'],'output_path':str(output),'status':'RUNNING',
         'source_sha256':sha(GS/'utils/mesh_utils.py'),'depth_units':'arbitrary inherited COLMAP units'}
    started=time.perf_counter();timings={};renders=[];srows=[]
    try:
        g=GaussianModel(3)
        scene=Scene(Namespace(**c['model']),g,load_iteration=30000,shuffle=False)
        checkpoint=Path(c['model']['model_path'])/'chkpnt30000.pth'
        captured,iteration=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert iteration==30000
        for i,f in enumerate(['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'],1):
            assert torch.equal(getattr(g,f).detach().cpu(),captured[i]),f
        del captured
        run.update(checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                   PLY_sha256=sha(Path(c['model']['model_path'])/'point_cloud/iteration_30000/point_cloud.ply'),Gaussian_count=len(g.get_xyz))
        train=scene.getTrainCameras();assert sorted(x.image_name for x in train)==mc['integration_cameras']
        assert all(x.gt_alpha_mask is not None for x in train)
        run['integration_camera_count']=len(train)
        g.active_sh_degree=0
        selected={train[i].image_name for i in [0,len(train)//2,len(train)-1]}
        def observed_render(cam,model,**kw):
            package=render(cam,model,**kw)
            path=output/'fusion_float'/f'{cam.image_name}.pt';path.parent.mkdir(exist_ok=True)
            data={key:package[key].detach().cpu() for key in ['render','rend_alpha','surf_depth']}
            if cam.image_name in selected:
                data.update({key:package[key].detach().cpu() for key in ['rend_normal','surf_normal']})
            assert all(torch.isfinite(v).all() for v in data.values())
            torch.save(data,path)
            renders.append({'sample':args.sample,'method':args.method,'camera':cam.image_name,'path':str(path),'sha256':sha(path),
                            'width':cam.image_width,'height':cam.image_height,'gt_alpha_sha256':tensor_sha(cam.gt_alpha_mask),
                            'world_to_camera':cam.world_view_transform.T.cpu().tolist(),'full_projection':cam.full_proj_transform.cpu().tolist(),
                            'FoVx':cam.FoVx,'FoVy':cam.FoVy,'normal_saved':cam.image_name in selected})
            return package
        extractor=GaussianExtractor(g,observed_render,Namespace(**c['pipeline']),bg_color=[0,0,0])
        mark=time.perf_counter();extractor.reconstruction(train);torch.cuda.synchronize()
        timings['depth_alpha_RGB_render_seconds']=time.perf_counter()-mark
        assert abs(extractor.radius-mc['camera_radius'])<mc['camera_radius']*1e-4
        mark=time.perf_counter()
        mesh=extractor.extract_mesh_bounded(voxel_size=mc['voxel_size'],sdf_trunc=mc['sdf_trunc'],depth_trunc=mc['depth_trunc'],mask_backgrond=True)
        timings['TSDF_fusion_seconds']=time.perf_counter()-mark
        mark=time.perf_counter()
        raw_stats,boundaries=structure(mesh)
        mesh.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(output/'raw.ply'),mesh)
        post=post_process_mesh(mesh,cluster_to_keep=mc['num_cluster'])
        post.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(output/'post.ply'),post)
        timings['postprocessing_validation_write_seconds']=time.perf_counter()-mark
        for name,m in [('raw',mesh),('post',post)]:
            stats,boundary=structure(m)
            reloaded=o3d.io.read_triangle_mesh(str(output/f'{name}.ply'))
            assert structure(reloaded)[0]==stats,'Reload changes mesh statistics'
            assert np.isfinite(np.asarray(reloaded.vertex_normals)).all()
            np.save(output/f'{name}_boundary_edges.npy',boundary)
            srows.append({'sample':args.sample,'method':args.method,'variant':'main','mesh_stage':name,**stats,
                          'mesh_path':str(output/f'{name}.ply'),'mesh_sha256':sha(output/f'{name}.ply'),'reload_verified':True})
        run.update(status='PASS',timings=timings,wall_seconds=time.perf_counter()-started,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_csv(STAGE/f'metrics/{label}_MESH_STRUCTURE.csv',srows)
        write_json(output/'FUSION_FLOAT_MANIFEST.json',renders)
    except BaseException as error:
        run.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)
        write_json(output/'RUN.json',run);write_json(STAGE/f'reports/{label}_MESH_RUN.json',run)
        raise
    write_json(output/'RUN.json',run);write_json(STAGE/f'reports/{label}_MESH_RUN.json',run)
    print('MESH_COMPLETED',label,run['wall_seconds'],[(x['mesh_stage'],x['vertices'],x['triangles']) for x in srows],flush=True)

if __name__=='__main__':main()
