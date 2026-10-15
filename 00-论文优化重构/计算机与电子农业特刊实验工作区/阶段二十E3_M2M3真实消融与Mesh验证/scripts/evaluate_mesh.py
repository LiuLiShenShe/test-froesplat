"""CPU raycasting in frozen registered cameras; consistency proxies, not GT."""
from common import *
from argparse import ArgumentParser,Namespace
import math,time

def camera_from_frozen(sample,name,w,h):
    import numpy as np,torch
    import_paths()
    from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary,qvec2rotmat
    from utils.graphics_utils import getWorld2View2
    sparse=Path(config(sample)['model']['source_path'])/'sparse/0'
    extrinsics=read_extrinsics_binary(sparse/'images.bin');intrinsics=read_intrinsics_binary(sparse/'cameras.bin')
    image=next(v for v in extrinsics.values() if Path(v.name).stem==name);intrinsic=intrinsics[image.camera_id]
    assert intrinsic.model=='PINHOLE'
    R=qvec2rotmat(image.qvec).T
    matrix=torch.tensor(getWorld2View2(R,image.tvec)).T
    return Namespace(image_width=w,image_height=h,FoVx=2*math.atan(intrinsic.width/(2*intrinsic.params[0])),
                     FoVy=2*math.atan(intrinsic.height/(2*intrinsic.params[1])),world_view_transform=matrix)

def main():
    p=ArgumentParser();p.add_argument('--sample',choices=SCENES,required=True);p.add_argument('--method',choices=['M0','M1','M2'],required=True);a=p.parse_args()
    import torch,numpy as np,open3d as o3d
    from mesh_metrics import raycast,reprojection
    core=metric_module();label=f'{a.sample}_{a.method}'
    inputs=rows(STAGE/f'metrics/{label}_FULL.csv');selected=set(sorted(r['loader_image_name'] for r in inputs)[i] for i in [0,len(inputs)//2,len(inputs)-1])
    result=[];start=time.perf_counter();target=DATA/'mesh_reprojection'/a.sample/a.method;assert not target.exists();target.mkdir(parents=True)
    for stage in ['raw','post']:
        mesh=o3d.io.read_triangle_mesh(str(DATA/'meshes'/a.sample/a.method/'main'/f'{stage}.ply'))
        for r in inputs:
            name=r['loader_image_name'];w,h=int(r['render_width']),int(r['render_height'])
            cam=camera_from_frozen(a.sample,name,w,h)
            d=torch.load(r['float_render_path'],map_location='cpu',weights_only=True)
            roi=core.load_roi(r['ROI_path'],w,h).numpy()
            depth,normal=raycast(mesh,cam)
            row={'sample':a.sample,'method':a.method,'mesh_stage':stage,'camera':name,**reprojection(depth,d['depth'].numpy(),d['geometry_replay_alpha'].numpy(),roi)}
            if name in selected:
                path=target/f'{stage}_{name}.npz';np.savez_compressed(path,depth=depth,normal=normal)
                row.update(raycast_float_path=str(path),raycast_float_sha256=sha(path))
            result.append(row);print('RAYCAST',label,stage,name,row['mesh_ROI_coverage_proxy'],flush=True)
    write_csv(STAGE/f'metrics/{label}_MESH_REPROJECTION.csv',result)
    write_json(STAGE/f'reports/{label}_MESH_REPROJECTION.json',{'status':'PASS','sample':a.sample,'method':a.method,'views':len(inputs),
               'meshes':2,'wall_seconds':time.perf_counter()-start,'coordinate_source':'Frozen COLMAP PINHOLE qvec/tvec and native getWorld2View2',
               'K_convention':'Native to_cam_open3d; focal from frozen FOV, principal point (W-1)/2,(H-1)/2',
               'depth_interpretation':'Ray direction camera z=1; t_hit is camera-axis depth, no metric scale',
               'independent_geometry_GT':'NONE','qualitative_view_selection':'Sorted held-out first, middle, last; fixed by rule before examining results'})

if __name__=='__main__':main()
