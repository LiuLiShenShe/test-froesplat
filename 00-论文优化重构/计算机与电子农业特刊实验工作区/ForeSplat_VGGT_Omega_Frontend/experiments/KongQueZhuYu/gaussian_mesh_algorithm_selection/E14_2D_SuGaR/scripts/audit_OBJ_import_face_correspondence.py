"""Locate actual native OBJ faces absent from the geometry-only import; no geometry changes."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
info=json.loads((ROOT/'refinement_complete.json').read_text())
obj=Path(info['textured_obj']);ply=Path(info['refined_raw_ply'])
audit=json.loads((ROOT/'native_OBJ_import_face_audit.json').read_text())
v=np.empty((audit['obj_vertices'],3),np.float64);f=np.empty((audit['obj_faces'],3),np.int32)
i=j=0;start=time.time()
with obj.open() as stream:
 for line in stream:
  if line.startswith('v '):v[i]=[float(x) for x in line.split()[1:4]];i+=1
  elif line.startswith('f '):f[j]=[int(x.split('/')[0])-1 for x in line.split()[1:]];j+=1
assert (i,j)==(len(v),len(f))
data=PlyData.read(ply);vv=np.stack([data['vertex'][axis] for axis in 'xyz'],1)
ff=np.stack(data['face']['vertex_indices']).astype(np.int32)
def records(a):return np.ascontiguousarray(a).view(np.dtype(('V',a.dtype.itemsize*a.shape[1]))).ravel()
unique_import_vertices,inverse_import=np.unique(vv,axis=0,return_inverse=True)
distance,nearest=cKDTree(v).query(unique_import_vertices,k=2,workers=4)
assert len(unique_import_vertices)==len(v) and len(np.unique(nearest[:,0]))==len(v),'Nearest coordinate mapping is not one-to-one'
assert np.all(distance[:,0]<distance[:,1]),'Ambiguous nearest-coordinate correspondence'
assert distance[:,0].max()<2e-7,'Imported coordinates depart from native precision mapping'
mapped=nearest[:,0][inverse_import][ff].astype(np.int32)
native_faces=records(np.sort(f,axis=1));import_faces=records(np.sort(mapped,axis=1))
unique_native,counts_native=np.unique(native_faces,return_counts=True)
unique_import,counts_import=np.unique(import_faces,return_counts=True)
missing_keys=np.setdiff1d(unique_native,unique_import);extra_keys=np.setdiff1d(unique_import,unique_native)
missing=np.flatnonzero(np.isin(native_faces,missing_keys))
v32=v.astype(np.float32)
zero32=np.linalg.norm(np.cross(v32[f[:,1]]-v32[f[:,0]],v32[f[:,2]]-v32[f[:,0]]),axis=1)==0
out=dict(status='complete',native_OBJ=str(obj.resolve()),geometry_import=str(ply.resolve()),native_faces=len(f),import_faces=len(ff),import_vertex_native_nearest_map_one_to_one=True,maximum_import_native_coordinate_distance=float(distance[:,0].max()),minimum_nearest_vs_second_distance_margin=float((distance[:,1]-distance[:,0]).min()),missing_native_face_ids_zero_based=missing.tolist(),unexpected_import_face_keys=int(len(extra_keys)),duplicate_native_face_keys=int((counts_native>1).sum()),duplicate_import_face_keys=int((counts_import>1).sum()),native_float32_zero_area_faces=int(zero32.sum()),missing_faces=[dict(face_id=int(k),vertex_ids=f[k].tolist(),vertices=v[f[k]].tolist(),area_float32=float(np.linalg.norm(np.cross(v32[f[k,1]]-v32[f[k,0]],v32[f[k,2]]-v32[f[k,0]]))*.5),area_float64=float(np.linalg.norm(np.cross(v[f[k,1]]-v[f[k,0]],v[f[k,2]]-v[f[k,0]]))*.5)) for k in missing],original_OBJ_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),elapsed_seconds=time.time()-start,scope='Actual native triangle membership audit; coordinate rounding explicitly measured; unchanged models, no claim about internal anatomical holes or specific importer implementation reason')
(ROOT/'native_OBJ_import_face_correspondence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
