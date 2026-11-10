"""P4 native bounded2DGS TSDF; no training/representation modification."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import sys,time,json,copy
from pathlib import Path
import numpy as np
import open3d as o3d
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection'
sys.path.insert(0,str(OLD/'scripts'))
from mesh_artifacts import save_meshes
from surface_protocol import write
out=OUT/'P4_2DGS_Depth_Mesh';out.mkdir(parents=True,exist_ok=True)
scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text())
index=json.loads((OLD/'surface_inputs/e11_2dgs_surface/render_index.json').read_text())
variants=[('expected_mask', 'depth_expected',False,1.),('median_mask','depth_median',False,1.),('expected_mask_alpha','depth_expected',True,1.),('median_mask_alpha','depth_median',True,1.),('median_mask_alpha_voxel075','depth_median',True,.75),('median_mask_alpha_voxel125','depth_median',True,1.25)]
volumes={};records={};starts={}
for name,key,alpha,mult in variants:
    d=out/name
    if (d/'complete.json').exists():continue
    starts[name]=time.time()
    volumes[name]=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=scale['tsdf_voxel']*mult,sdf_trunc=scale['tsdf_sdf_trunc']*mult,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    records[name]=[]
write(out/'parameters.json',dict(variants=variants,scale=scale,input_index=str(OLD/'surface_inputs/e11_2dgs_surface/render_index.json'),checkpoint=str(OLD/'E11_2DGS_Corrected/checkpoints/point_cloud/iteration_30000/point_cloud.ply'),native_algorithm='2DGS bounded Open3D ScalableTSDFVolume',mask='original RAP mask',alpha_threshold=.5,post_rule='all components >=50 triangles',views=208,open3d=o3d.__version__))
for row in index:
    with np.load(row['path']) as a:
        arrays={k:a[k] for k in ['depth_expected','depth_median','alpha','foreground_mask','render_raw']}
    c=row['camera'];E=np.asarray(c['world_view_transform'],np.float64).T
    K=o3d.camera.PinholeCameraIntrinsic(c['width'],c['height'],c['Fx'],c['Fy'],c['Cx'],c['Cy'])
    rgb=np.ascontiguousarray((np.clip(arrays['render_raw'],0,1)*255).astype(np.uint8))
    for name,key,alpha,mult in variants:
        if name not in volumes:continue
        depth=arrays[key].copy();valid=arrays['foreground_mask']&np.isfinite(depth)&(depth>0)
        if alpha:valid &= arrays['alpha']>=.5
        records[name].append(dict(source_id=row['source_id'],foreground_pixels=int(arrays['foreground_mask'].sum()),valid_depth_pixels=int(valid.sum()),depth_zero_fg=int((arrays['foreground_mask']&(~np.isfinite(depth)|(depth<=0))).sum())))
        depth[~valid]=0
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(np.ascontiguousarray(depth,np.float32)),depth_scale=1.,depth_trunc=scale['tsdf_depth_trunc'],convert_rgb_to_intensity=False)
        volumes[name].integrate(rgbd,K,E)
    print('P4 fused',row['source_id'],'all variants',flush=True)
for name,key,alpha,mult in variants:
    if name not in volumes:continue
    mesh=volumes[name].extract_triangle_mesh();d=out/name;stats=save_meshes(mesh,d,scale)
    write(d/'complete.json',dict(status='native_fusion_complete_pending_evaluation',elapsed_seconds=time.time()-starts[name],depth_source=key,mask='RAP',alpha_min=.5 if alpha else None,voxel_size=scale['tsdf_voxel']*mult,truncation=scale['tsdf_sdf_trunc']*mult,frames=records[name],**stats))
    del volumes[name]
    print('P4',name,stats,flush=True)
write(out/'complete.json',dict(status='6_native_fusions_complete_pending_evaluation',variants=[x[0] for x in variants],views=208))
sys.stdout.flush();os._exit(0)

