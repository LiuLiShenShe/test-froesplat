"""Export center positions as a clearly labelled Point Cloud PLY, never Gaussian."""
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData,PlyElement
ROOT=Path(__file__).resolve().parents[1]
def main():
 native=ROOT/'checkpoints/point_cloud/iteration_30000/point_cloud.ply'
 assert native.is_file()
 ply=PlyData.read(native);v=ply['vertex'].data
 assert all(f'scale_{i}' in v.dtype.names for i in range(6))
 xyz=np.column_stack([v[k] for k in ['x','y','z']])
 rgb=np.clip(np.column_stack([v[f'f_dc_{i}'] for i in range(3)])*0.28209479177387814+0.5,0,1)
 finite=np.isfinite(xyz).all(1)&np.isfinite(rgb).all(1)
 ids=np.flatnonzero(finite);xyz=xyz[finite];rgb=rgb[finite]
 data=np.empty(len(ids),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
 for i,k in enumerate(['x','y','z']):data[k]=xyz[:,i]
 for i,k in enumerate(['red','green','blue']):data[k]=(rgb[:,i]*255).round().astype(np.uint8)
 out=ROOT/'viewer_exports';out.mkdir(exist_ok=True)
 path=out/'quadratic_centers_rgb_point_cloud.ply'
 PlyData([PlyElement.describe(data,'vertex')],text=False).write(path)
 mapping=out/'center_point_native_indices.npz';np.savez_compressed(mapping,native_primitive_indices=ids)
 (out/'INDEX.json').write_text(json.dumps(dict(artifact_type='Point Cloud PLY',format='Finite XYZRGB center sample only',path=str(path.resolve()),source_native_model=str(native.resolve()),points=len(ids),native_primitives=len(v),nonfinite_centers_excluded=int((~finite).sum()),native_index_mapping=str(mapping.resolve()),original_native_model_unchanged=True,representation_warning='No quadratic shape, normals, opacity or SH angular appearance preserved; not a Gaussian PLY or a QGS surface model',viewer='MeshLab/Open3D point-cloud view'),indent=2)+'\n')
 print(path.resolve())
if __name__=='__main__':main()

