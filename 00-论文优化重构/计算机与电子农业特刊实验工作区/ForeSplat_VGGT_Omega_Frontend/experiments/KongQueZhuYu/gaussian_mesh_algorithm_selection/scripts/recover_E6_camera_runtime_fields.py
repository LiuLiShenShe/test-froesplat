"""Recover missing E6 scalar camera fields from the actual retained historical FoV/P."""
import json,hashlib,math
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'root_cause_audit/render_error_analysis/checkpoint_rerender_v1/E6/camera_runtime.json'
DEST=OUT/'comparison/history_uniform/E6';DEST.mkdir(parents=True,exist_ok=True)
original=json.loads(OLD.read_text());records=[];checks=[]
for row in original['records']:
 r=dict(row);P=np.asarray(r['projection_matrix'],float).T
 fx=float(r['fov_derived_fx']);fy=float(r['fov_derived_fy'])
 assert fx>0 and fy>0 and np.isfinite(P).all()
 assert np.isclose(math.tan(r['FoVx']/2),r['width']/(2*fx),atol=1e-14)
 assert np.isclose(math.tan(r['FoVy']/2),r['height']/(2*fy),atol=1e-14)
 assert abs(P[3,2]-1)<1e-12 and abs(P[3,3])<1e-12
 r.update(Fx=fx,Fy=fy,Cx=float(((P[0,2]+1)*r['width']-1)/2),Cy=float(((P[1,2]+1)*r['height']-1)/2),scalar_camera_fields_source='Actual retained FoV focal fields and original centered transposed P; integer-centre ndc2pix; historical native normal unprojection convention retained')
 for key in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
  assert r[key]==row[key]
 probes=np.asarray([[0,0,1,1],[.1,.05,1,1],[-.1,-.05,1,1],[.04,-.02,.5,1]],float)
 projected=probes@P.T;uv=(projected[:,:2]/projected[:,3:4]+1)*[r['width']/2,r['height']/2]-.5
 scalar=probes[:,:2]/probes[:,2:3]*[fx,fy]+[r['Cx'],r['Cy']]
 error=float(np.abs(uv-scalar).max());assert error<.001
 checks.append(dict(source_id=r['source_id'],FoVx_recovered_error=abs(2*math.atan(r['width']/(2*fx))-r['FoVx']),FoVy_recovered_error=abs(2*math.atan(r['height']/(2*fy))-r['FoVy']),scalar_vs_retained_float32_P_probe_max_pixel_error=error,Fx=fx,Fy=fy,Cx=r['Cx'],Cy=r['Cy']))
 records.append(r)
assert len(records)==208 and len({r['source_id'] for r in records})==208
new=DEST/'recovered_camera_runtime.json';new.write_text(json.dumps(dict(original,records=records),ensure_ascii=False,indent=2)+'\n')
(DEST/'CAMERA_FIELD_RECOVERY_AUDIT.json').write_text(json.dumps(dict(status='complete',source_camera=str(OLD.resolve()),source_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),recovered_camera=str(new.resolve()),all208_original_W2C_P_fullP_center_unchanged=True,source_file_unchanged=True,maximum_probe_error_px=max(r['scalar_vs_retained_float32_P_probe_max_pixel_error'] for r in checks),rows=checks,scope='Historical E6 centered native P is retained; this is metadata recovery, not the corrected E10 interface or new frontend inference. Native depth-normal helper keeps its historical pixel convention.'),ensure_ascii=False,indent=2)+'\n')
print('E6 recovered actual208 scalar fields; all matrices unchanged; max pixel error',max(r['scalar_vs_retained_float32_P_probe_max_pixel_error'] for r in checks))
