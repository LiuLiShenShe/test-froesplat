"""Supplementary predeclared TSDF sweep; reuses real main-protocol float caches."""
from common import *
from argparse import ArgumentParser,Namespace
import time,traceback,resource

def main():
    p=ArgumentParser();p.add_argument('--sample',choices=SCENES,required=True);a=p.parse_args()
    gpu_idle();verify_frozen();import_paths()
    import torch,numpy as np,open3d as o3d
    from PIL import Image
    from utils.mesh_utils import GaussianExtractor,post_process_mesh
    from utils.graphics_utils import getProjectionMatrix
    from utils.general_utils import PILtoTorch
    from mesh_metrics import structure
    mc=read_json(STAGE/f'configs/{a.sample}_MESH.json');main=DATA/'meshes'/a.sample/'M0/main'
    assert read_json(main/'RUN.json')['status']=='PASS'
    manifest=read_json(main/'FUSION_FLOAT_MANIFEST.json')
    rows_by_name={Path(r['registered_name']).stem:r for r in rows(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv') if r['sample']==a.sample and r['group']=='B_R0'}
    cameras=[]
    for r in manifest:
        w,h=r['width'],r['height']
        with Image.open(rows_by_name[r['camera']]['loader_image_path']) as image:alpha=PILtoTorch(image.split()[3],(w,h))
        assert tensor_sha(alpha)==r['gt_alpha_sha256']
        cameras.append(Namespace(image_width=w,image_height=h,gt_alpha_mask=alpha,
                       projection_matrix=getProjectionMatrix(znear=.01,zfar=100.,fovX=r['FoVx'],fovY=r['FoVy']).T.cuda(),
                       world_view_transform=torch.tensor(r['world_to_camera']).T.cuda()))
    class Cache:
        def __init__(self,key):self.key=key
        def __getitem__(self,index):return torch.load(manifest[index]['path'],map_location='cpu',weights_only=True)[self.key]
    extractor=object.__new__(GaussianExtractor);extractor.viewpoint_stack=cameras;extractor.rgbmaps=Cache('render');extractor.depthmaps=Cache('surf_depth')
    results=[];runs=[]
    for factor in mc['sensitivity_voxel_factors']:
        out=DATA/'meshes'/a.sample/'M0'/f'sensitivity_voxel_{factor:g}';assert not out.exists();out.mkdir(parents=True)
        start=time.perf_counter();run={'sample':a.sample,'method':'M0','variant':f'sensitivity_voxel_{factor:g}','voxel_factor':factor,
               'primary_result':False,'depth_trunc':mc['depth_trunc'],'voxel_size':mc['voxel_size']*factor,'sdf_trunc':mc['sdf_trunc']*factor,
               'source_manifest_sha256':sha(main/'FUSION_FLOAT_MANIFEST.json'),'status':'RUNNING','command':sys.argv,'GPU_UUID':GPU_UUID}
        try:
            raw=extractor.extract_mesh_bounded(voxel_size=run['voxel_size'],sdf_trunc=run['sdf_trunc'],depth_trunc=run['depth_trunc'],mask_backgrond=True)
            raw.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(out/'raw.ply'),raw)
            post=post_process_mesh(raw,cluster_to_keep=mc['num_cluster']);post.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(out/'post.ply'),post)
            for stage,m in [('raw',raw),('post',post)]:
                stats,_=structure(m);path=out/f'{stage}.ply';assert structure(o3d.io.read_triangle_mesh(str(path)))[0]==stats
                results.append({'sample':a.sample,'method':'M0','variant':run['variant'],'mesh_stage':stage,'voxel_factor':factor,**stats,'mesh_path':str(path),'mesh_sha256':sha(path),'reload_verified':True})
            run['status']='PASS'
        except BaseException as e:
            run.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
        run.update(wall_seconds=time.perf_counter()-start,peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(out/'RUN.json',run);runs.append(run)
    write_csv(STAGE/f'metrics/{a.sample}_M0_MESH_SENSITIVITY.csv',results)
    write_json(STAGE/f'reports/{a.sample}_M0_MESH_SENSITIVITY.json',runs)

if __name__=='__main__':main()
