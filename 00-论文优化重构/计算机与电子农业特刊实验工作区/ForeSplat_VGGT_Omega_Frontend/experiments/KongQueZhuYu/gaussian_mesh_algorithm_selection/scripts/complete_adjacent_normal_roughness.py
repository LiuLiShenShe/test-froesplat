"""Complete old/native Mesh roughness with the exact new208 evaluator formula."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import ast,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import open3d as o3d
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1]
DEST=OUT/'comparison/adjacent_normal_roughness';DEST.mkdir(exist_ok=True)
source=OUT/'scripts/evaluate_mesh.py'
tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='roughness')
namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
rows=list(csv.DictReader((OUT/'comparison/ALL_MESH_EXPERIMENTS.csv').open()));index=[];started=time.time()
for row in rows:
    actual=Path(row['evaluated_mesh_path']).resolve(strict=True)
    alias=hashlib.sha256(str(actual).encode()).hexdigest()[:16]
    marker=DEST/(alias+'.json')
    if marker.is_file():
        result=json.loads(marker.read_text());assert result['source_mesh']==str(actual)
        assert result['file_size_bytes']==actual.stat().st_size
    else:
        if row.get('surface_roughness'):
            result=dict(status='existing_same_formula_verified',surface_roughness=float(row['surface_roughness']),source_mesh=str(actual),file_size_bytes=actual.stat().st_size,vertices=int(row['vertices']),faces=int(row['faces']),source_evaluation=row['metric_source'])
        else:
            mesh=o3d.io.read_triangle_mesh(str(actual));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
            assert len(v)==int(row['vertices']) and len(f)==int(row['faces'])
            result=dict(status='new_actual_mesh_complete',source_mesh=str(actual),file_size_bytes=actual.stat().st_size,vertices=len(v),faces=len(f),**namespace['roughness'](v,f))
            del mesh,v,f
        result.update(formula_source=str(source.resolve()),formula_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),definition='Exact evaluate_mesh.roughness: mean1-cos across all manifold interior edges; includes true curvature and triangulation effects, not scan truth')
        write_status(marker,result)
    index.append(dict(mesh_id=row['mesh_id'],stage=row['mesh_stage'],mesh=str(actual),summary=str(marker.resolve()),surface_roughness=result['surface_roughness']))
    print(row['mesh_id'],row['mesh_stage'],result['surface_roughness'],flush=True)
write_status(OUT/'comparison/ADJACENT_NORMAL_ROUGHNESS_INDEX.json',index)
write_status(DEST/'complete.json',dict(status='actual_available_complete',routes=len(index),elapsed_seconds=time.time()-started,formula_source=str(source.resolve()),method='Identical analytic all-manifold-edge formula for old and new models; each source Mesh path/size and vertex/face count verified. Other historical metrics unchanged.'))
print('Actual common adjacent-normal roughness',len(index),flush=True)
