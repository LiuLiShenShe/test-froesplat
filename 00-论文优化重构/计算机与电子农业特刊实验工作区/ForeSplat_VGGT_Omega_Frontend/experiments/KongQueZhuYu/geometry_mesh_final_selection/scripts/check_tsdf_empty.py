import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8'
from pathlib import Path
import json,numpy as np,open3d as o3d,time
p=Path(__file__).resolve().parents[1];inp=p/'surface_inputs/e11_2dgs_surface';row=json.loads((inp/'render_index.json').read_text())[0];c=row['camera'];scale=json.loads((p/'configs/standard_spatial_scale.json').read_text())
with np.load(row['path']) as d:z=d['depth_surface'].copy();mask=d['foreground_mask'];rgb=(np.clip(d['render_raw'],0,1)*255).astype('uint8')
E=np.ascontiguousarray(np.asarray(c['world_view_transform']).T);K=o3d.camera.PinholeCameraIntrinsic(c['width'],c['height'],c['Fx'],c['Fy'],c['Cx'],c['Cy']);records=[]
for kind in ['real_first_view','constant_plane']:
 depth=z.copy() if kind.startswith('real') else np.full_like(z,.8)
 if kind.startswith('real'):depth[~mask]=0
 v=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=scale['tsdf_voxel'],sdf_trunc=scale['tsdf_sdf_trunc'],color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
 rd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(rgb)),o3d.geometry.Image(np.ascontiguousarray(depth,dtype=np.float32)),depth_scale=1.,depth_trunc=scale['tsdf_depth_trunc'],convert_rgb_to_intensity=False)
 actual=np.asarray(rd.depth);print('RGBD',kind,actual.shape,actual.dtype,float(actual.max()),int((actual>0).sum()),v.voxel_length,v.sdf_trunc,flush=True)
 for i in range(3):v.integrate(rd,K,E)
 m=v.extract_triangle_mesh();pc=v.extract_voxel_point_cloud();r=dict(kind=kind,positive_depth_pixels=int((actual>0).sum()),max_depth=float(actual.max()),vertices=len(m.vertices),faces=len(m.triangles),voxels=len(pc.points),version=o3d.__version__,numpy=np.__version__);records.append(r);print(r,flush=True)
out=p/'camera_interface'/f'tsdf_empty_validation_o3d_{o3d.__version__}.json';out.write_text(json.dumps(records,indent=2)+'\n')
