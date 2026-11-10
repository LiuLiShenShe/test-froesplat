"""GGGS native median-depth VoxelBlockGrid CPU fusion on a provided208 index."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import sys,argparse,json,time
from pathlib import Path
import numpy as np
import open3d as o3d
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection'
sys.path.insert(0,str(OLD/'scripts'))
from mesh_artifacts import save_meshes
from surface_protocol import write
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--alpha-filter',action='store_true');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
if (a.out/'complete.json').exists():raise SystemExit(0)
scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text());index=json.loads((a.input/'render_index.json').read_text());assert len(index)==208;start=time.time();counts=[]
volume=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),attr_dtypes=(o3d.core.float32,)*3,attr_channels=(1,1,3),voxel_size=scale['tsdf_voxel'],block_resolution=16,block_count=50000,device=o3d.core.Device('CPU:0'))
for row in index:
    c=row['camera']
    with np.load(row['path']) as d:depth=d['depth_surface'].copy();mask=d['foreground_mask'];alpha=d['alpha'];rgb=np.clip(d['render_raw'],0,1)
    valid=mask&np.isfinite(depth)&(depth>0)
    if a.alpha_filter:valid &= alpha>=.5
    counts.append(dict(source_id=row['source_id'],integrated_pixels=int((valid&(depth<=scale['tsdf_depth_trunc'])).sum())))
    depth[~valid]=0;dep=o3d.t.geometry.Image(np.ascontiguousarray(depth,np.float32));col=o3d.t.geometry.Image(np.ascontiguousarray(rgb,np.float32))
    K=o3d.core.Tensor(np.array([[c['Fx'],0,c['Cx']],[0,c['Fy'],c['Cy']],[0,0,1]],np.float64));E=o3d.core.Tensor(np.asarray(c['world_view_transform'],np.float64).T)
    blocks=volume.compute_unique_block_coordinates(dep,K,E,1.,scale['tsdf_depth_trunc'],5.)
    volume.integrate(blocks,dep,col,K,E,1.,scale['tsdf_depth_trunc'],5.)
    print('TSDF',a.out.name,row['source_id'],flush=True)
mesh=volume.extract_triangle_mesh().to_legacy();stats=save_meshes(mesh,a.out,scale)
write(a.out/'complete.json',dict(status='native_fusion_complete_pending_evaluation',scale=scale,input=str(a.input),depth='GGGS native median',mask='RAP',alpha_filter=a.alpha_filter,elapsed_seconds=time.time()-start,open3d=o3d.__version__,frames=counts,**stats))
sys.stdout.flush();os._exit(0)

