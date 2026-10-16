"""One predeclared pruned model through the frozen, untouched E3 TSDF protocol."""
from common import *
from native_context import cameras
from argparse import Namespace
import time
import traceback


def structure(mesh):
    import numpy as np
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
    assert len(v)>0 and len(f)>0
    assert np.isfinite(v).all() and f.min()>=0 and f.max()<len(v)
    return {'vertices':len(v),'triangles':len(f),'bounds_min':v.min(0).tolist(),'bounds_max':v.max(0).tolist(),
            'nonfinite_vertex_values':int((~np.isfinite(v)).sum()),'degenerate_faces':int(((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,0]==f[:,2])).sum())}


def main():
    assert_gpu();p=plan();import_paths()
    import numpy as np
    import torch
    import open3d as o3d
    from gaussian_renderer import render
    from plugins.gaussian_compression.evaluation.native import load_native
    from utils.mesh_utils import GaussianExtractor,post_process_mesh
    torch.set_num_threads(4)
    choice=p['mesh'];protocol=read_json(choice['protocol_path'])
    assert sha(choice['protocol_path'])==choice['protocol_sha256']
    run=next(r for r in rows(STAGE/'COMPRESSION_RUN_MANIFEST.csv') if r['run_id']==choice['run_id'])
    output=DATA/'mesh'/choice['run_id'];output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','started_at':now(),'run_id':run['run_id'],'model_sha256':run['output_model_sha256'],
            'protocol':protocol,'protocol_sha256':choice['protocol_sha256'],'active_SH_degree':choice['SH_degree'],
            'GPU_uuid':GPU_UUID,'background':p['background'],'geometric_accuracy_GT':'NONE'}
    started=time.perf_counter();renders=[];timings={}
    try:
        scene,_=cameras(p)
        g=load_native(run['pruned_model_path'],GS,choice['SH_degree'])
        train=scene.getTrainCameras();assert sorted(c.image_name for c in train)==protocol['integration_cameras']
        assert all(c.gt_alpha_mask is not None for c in train)
        result['integration_camera_count']=len(train)
        def observed_render(cam,model,**kw):
            package=render(cam,model,**kw)
            selected={k:package[k].detach().cpu() for k in ['render','rend_alpha','surf_depth']}
            assert all(torch.isfinite(v).all() for v in selected.values())
            alpha=selected['rend_alpha'];depth=selected['surf_depth']
            assert alpha.min()>=0 and alpha.max()<=1.00001
            assert (depth>0).any()
            path=output/'fusion_float'/f'{cam.image_name}.pt';path.parent.mkdir(exist_ok=True)
            torch.save(selected,path)
            renders.append({'camera':cam.image_name,'width':cam.image_width,'height':cam.image_height,
                'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
                'alpha_min':alpha.min().item(),'alpha_max':alpha.max().item(),
                'positive_depth_pixels':int((depth>0).sum()),'depth_max':depth.max().item(),
                'frozen_gt_alpha_sha256':tensor_sha(cam.gt_alpha_mask),
                'world_view_transform_sha256':tensor_sha(cam.world_view_transform),'full_proj_transform_sha256':tensor_sha(cam.full_proj_transform)})
            return package
        extractor=GaussianExtractor(g,observed_render,Namespace(**p['pipeline']),bg_color=p['background'])
        tick=time.perf_counter();extractor.reconstruction(train);torch.cuda.synchronize()
        timings['depth_alpha_RGB_render_and_evidence_write_seconds']=time.perf_counter()-tick
        assert abs(extractor.radius-protocol['camera_radius'])<protocol['camera_radius']*1e-4
        assert np.allclose(extractor.center.cpu().numpy(),protocol['camera_focus'],atol=1e-5)
        tick=time.perf_counter()
        mesh=extractor.extract_mesh_bounded(voxel_size=protocol['voxel_size'],sdf_trunc=protocol['sdf_trunc'],depth_trunc=protocol['depth_trunc'],mask_backgrond=True)
        timings['TSDF_fusion_seconds']=time.perf_counter()-tick
        stats=structure(mesh);mesh.compute_vertex_normals()
        assert o3d.io.write_triangle_mesh(str(output/'raw.ply'),mesh)
        post=post_process_mesh(mesh,cluster_to_keep=protocol['num_cluster']);post.compute_vertex_normals()
        assert o3d.io.write_triangle_mesh(str(output/'post.ply'),post)
        meshes=[]
        for name,generated in [('raw',mesh),('post',post)]:
            path=output/f'{name}.ply';again=o3d.io.read_triangle_mesh(str(path))
            original=structure(generated);reloaded=structure(again)
            assert original==reloaded
            assert np.isfinite(np.asarray(again.vertex_normals)).all()
            meshes.append({'variant':name,**original,'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'reload_success':True})
        write_csv(STAGE/'configs/MESH_FUSION_FLOAT_INDEX.csv',renders)
        write_json(output/'FUSION_FLOAT_MANIFEST.json',renders)
        result.update(status='PASS',completed_at=now(),meshes=meshes,timings=timings,
            all_depth_alpha_finite=True,all_integration_views_nonempty_depth=True,
            wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),mesh_dir=str(output),
            masking='frozen training gt_alpha_mask < 0.5 depth invalidation; no predicted alpha threshold',
            interpretation='engineering TSDF compatibility only, no geometric accuracy improvement claim')
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        write_json(output/'RUN.json',result);write_json(STAGE/'reports/MESH_COMPATIBILITY.json',result)
    print('MESH_COMPATIBILITY_PASS',[(m['variant'],m['vertices'],m['triangles']) for m in meshes],flush=True)

if __name__=='__main__':main()
