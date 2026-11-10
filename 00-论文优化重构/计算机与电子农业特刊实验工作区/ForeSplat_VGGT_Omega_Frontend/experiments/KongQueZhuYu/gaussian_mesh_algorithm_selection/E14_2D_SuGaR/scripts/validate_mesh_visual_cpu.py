"""CPU control: a flat two-triangle square must show its actual interior diagonal, with zero normal/image edge."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
import numpy as np,open3d as o3d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_mesh_geometry_visuals import actual_raycast,images_from_channels,png
mesh=o3d.geometry.TriangleMesh()
v=np.array([[-.8,-.8,2],[.8,-.8,2],[.8,.8,2],[-.8,.8,2]],float);f=np.array([[0,1,2],[0,2,3]],np.int32)
mesh.vertices=o3d.utility.Vector3dVector(v);mesh.triangles=o3d.utility.Vector3iVector(f)
mesh.triangle_uvs=o3d.utility.Vector2dVector(np.array([[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]],float))
mesh.triangle_material_ids=o3d.utility.IntVector([0,0])
mesh.textures=[o3d.geometry.Image(np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],np.uint8))]
camera=dict(width=100,height=100,Fx=100.,Fy=100.,Cx=50.,Cy=50.,world_view_transform=np.eye(4).tolist())
data=actual_raycast(mesh,camera)
inner=data['wireframe'][24:77,24:77];assert inner.sum()>=50
n=data['normal_world'][data['silhouette']];assert np.ptp(n,axis=0).max()==0
assert data['has_native_uv_texture'] and not data['has_vertex_color']
assert np.max(np.abs(data['texture'][50,50]-.5))<1e-6
assert data['wireframe'][50,50] and data['silhouette'][50,50]
folder=ROOT/'cpu_visual_validation';folder.mkdir(exist_ok=True)
for name,img in images_from_channels(data,camera).items():png(folder/(name+'.png'),img)
obj=folder/'synthetic_uv_fixture.obj'
assert o3d.io.write_triangle_mesh(str(obj),mesh)
loaded=o3d.io.read_triangle_mesh(str(obj));actual=actual_raycast(loaded,camera)
assert actual['has_native_uv_texture'] and np.max(np.abs(actual['texture'][50,50]-.5))<1e-6
assert np.max(np.abs(actual['texture'][30,30]-[.375,.25,.75]))<1e-6
scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(loaded))
cached=actual_raycast(loaded,camera,scene=scene)
for key in ['depth','normal_world','silhouette','wireframe','texture']:assert np.array_equal(actual[key],cached[key]),key
del scene
record=dict(shared_BVH_first_hit_equivalence=True,actual_OBJ_UV_MTL_PNG_load_passed=True,UV_axis_orientation_check=True,texture_offcenter_actual=actual['texture'][30,30].tolist(),texture_offcenter_expected=[.375,.25,.75],fixture_vertices=len(loaded.vertices),fixture_triangles=len(loaded.triangles),status='passed',GPU=False,control='Synthetic flat square split into two actual triangles. Interior geometric diagonal is visible while face normal and gray surface are constant.',inner_diagonal_wireframe_pixels=int(inner.sum()),normal_variation=float(np.ptp(n,axis=0).max()),texture_center_actual=data['texture'][50,50].tolist(),texture_center_expected=[.5,.5,.5],vertex_color_available=False,UV_texture_available=True,experimental_model=False)
(folder/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

