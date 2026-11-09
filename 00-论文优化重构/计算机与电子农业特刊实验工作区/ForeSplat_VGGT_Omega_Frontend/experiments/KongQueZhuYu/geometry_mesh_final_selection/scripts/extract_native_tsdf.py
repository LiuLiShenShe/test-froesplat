"""Native GGGS VoxelBlockGrid / official 2DGS bounded ScalableTSDFVolume."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,json,time,hashlib
import numpy as np
import open3d as o3d
from surface_protocol import ROOT,PROTOCOL,scale_for,write
from mesh_artifacts import save_meshes

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF'],required=True);p.add_argument('--alpha-filter',action='store_true');a=p.parse_args()
    exp='E10' if a.experiment.startswith('M4') else 'E11';inp=ROOT/'surface_inputs'/f'{exp.lower()}_{"gggs" if exp=="E10" else "2dgs"}_surface'
    out=ROOT/'meshes'/a.experiment
    if a.alpha_filter:out=out/'supplementary_alpha_filtered'
    if (out/'complete.json').exists():return
    scale=scale_for('standard');index=json.loads((inp/'render_index.json').read_text());start=time.time();counts=[]
    if exp=='E10':
        volume=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),attr_dtypes=(o3d.core.float32,)*3,attr_channels=(1,1,3),
            voxel_size=scale['tsdf_voxel'],block_resolution=16,block_count=50000,device=o3d.core.Device('CPU:0'))
    else:
        volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=scale['tsdf_voxel'],sdf_trunc=scale['tsdf_sdf_trunc'],color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for row in index:
        c=row['camera']
        with np.load(row['path']) as d:
            depth=d['depth_surface'].copy();mask=d['foreground_mask'];alpha=d['alpha'];rgb=np.clip(d['render_raw'],0,1)
        valid=mask&np.isfinite(depth)&(depth>0)
        if a.alpha_filter:valid &= alpha>=PROTOCOL['surface_alpha_min']
        counts.append(dict(source_id=row['source_id'],integrated_pixels=int((valid&(depth<=scale['tsdf_depth_trunc'])).sum()),depth_truncated_pixels=int((valid&(depth>scale['tsdf_depth_trunc'])).sum())))
        depth[~valid]=0;depth=np.ascontiguousarray(depth,dtype=np.float32)
        K=np.array([[c['Fx'],0,c['Cx']],[0,c['Fy'],c['Cy']],[0,0,1]],dtype=np.float64);E=np.asarray(c['world_view_transform'],dtype=np.float64).T
        if exp=='E10':
            dep=o3d.t.geometry.Image(depth);col=o3d.t.geometry.Image(np.ascontiguousarray(rgb,dtype=np.float32));k=o3d.core.Tensor(K);e=o3d.core.Tensor(E)
            blocks=volume.compute_unique_block_coordinates(dep,k,e,1.,scale['tsdf_depth_trunc'],float(PROTOCOL['tsdf_sdf_voxels']))
            volume.integrate(blocks,dep,col,k,e,1.,scale['tsdf_depth_trunc'],float(PROTOCOL['tsdf_sdf_voxels']))
        else:
            col=o3d.geometry.Image(np.ascontiguousarray((rgb*255).astype(np.uint8)));dep=o3d.geometry.Image(depth)
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(col,dep,depth_scale=1.,depth_trunc=scale['tsdf_depth_trunc'],convert_rgb_to_intensity=False)
            k=o3d.camera.PinholeCameraIntrinsic(c['width'],c['height'],c['Fx'],c['Fy'],c['Cx'],c['Cy']);volume.integrate(rgbd,k,E)
        print(a.experiment,row['source_id'],'TSDF integrated',flush=True)
    mesh=volume.extract_triangle_mesh();mesh=mesh.to_legacy() if exp=='E10' else mesh;stats=save_meshes(mesh,out,scale)
    write(out/'complete.json',dict(status='complete',experiment=a.experiment,input=str(inp),scale=scale,protocol=PROTOCOL,elapsed_seconds=time.time()-start,open3d_version=o3d.__version__,
        fusion='GGGS native Open3D VoxelBlockGrid CPU' if exp=='E10' else '2DGS native bounded Open3D ScalableTSDFVolume',
        depth='GGGS median' if exp=='E10' else '2DGS expected, unchanged depth_ratio=0',mask='Original RAP mask',alpha_filter=a.alpha_filter,
        comparison_note='Main TSDF preserves native mask/depth integration; alpha-filtered supplement matches NKSR surface validity to isolate this adapter difference',frames=counts,**stats))
    print(a.experiment,stats,flush=True)

if __name__=='__main__':
    # Independent CPU scheduling may reach the same cached experiment later.
    # Serialize writers and re-check completion inside main after acquiring it.
    import fcntl,sys
    method=sys.argv[sys.argv.index('--experiment')+1]
    target=ROOT/'meshes'/method
    if '--alpha-filter' in sys.argv:target=target/'supplementary_alpha_filtered'
    target.mkdir(parents=True,exist_ok=True)
    with (target/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
