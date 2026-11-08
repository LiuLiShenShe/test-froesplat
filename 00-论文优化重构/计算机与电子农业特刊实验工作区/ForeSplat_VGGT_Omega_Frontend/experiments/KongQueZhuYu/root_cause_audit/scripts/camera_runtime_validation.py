"""Check actual checkpoint renderer Camera matrices, not only cfg or FoV."""
import json
import numpy as np
from camera_numeric_audit import OUT,OMEGA

def main():
    root=OMEGA/'experiments/KongQueZhuYu/root_cause_audit'
    chain=json.loads((OUT/'camera_matrix_chain.json').read_text());results={}
    for experiment,method in [('E1','VGGT_Omega'),('E5','Standard_VGGT'),('E6','Standard_VGGT')]:
        path=root/'render_error_analysis/checkpoint_rerender_v1'/experiment/'camera_runtime.json'
        if not path.exists():continue
        data=json.loads(path.read_text());records=data['records'];errors=[]
        for c in records:
            original=chain[method+'/'+c['name']+'.png'];K=np.array(original['K_gggs']);E=np.array(original['W2C_colmap'])
            V=np.array(c['world_view_transform']);P=np.array(c['projection_matrix']);full=np.array(c['full_proj_transform'])
            R=V.T[:3,:3];t=V.T[:3,3]
            rays=np.array([[0.,0.],[c['width']/2,c['height']/2],[c['width']-1.,c['height']-1.],[c['width']*.3,c['height']*.7]])
            xc=np.c_[(rays[:,0]-K[0,2])/K[0,0],(rays[:,1]-K[1,2])/K[1,1],np.ones(len(rays))]
            X=(xc-t)@np.linalg.inv(R).T
            clip=np.c_[X,np.ones(len(X))]@full;uv=((clip[:,:2]/clip[:,3:4]+1)*np.array([c['width'],c['height']])-1)/2
            errors.append(dict(name=c['name'],runtime_WH=[c['width'],c['height']],K_to_runtime_focal_max=max(abs(c['fov_derived_fx']-K[0,0]),abs(c['fov_derived_fy']-K[1,1])),
                runtime_W2C_to_colmap_max=float(np.max(np.abs(V.T-E))),runtime_full_matrix_composition_max=float(np.max(np.abs(full-V@P))),
                ray_to_actual_raster_projection_max_px=float(np.max(np.linalg.norm(rays-uv,axis=1))),
                runtime_principal=[c.get('Cx',(c['width']-1)/2),c.get('Cy',(c['height']-1)/2)]))
        results[experiment]=dict(camera_count=len(records),all_source_names_unique=len({c['name'] for c in records})==len(records),
                W_H_unique=sorted({tuple(c['runtime_WH']) for c in errors}),
                focal_max_error=max(c['K_to_runtime_focal_max'] for c in errors),
                W2C_max_error=max(c['runtime_W2C_to_colmap_max'] for c in errors),
                full_matrix_composition_max=max(c['runtime_full_matrix_composition_max'] for c in errors),
                ray_roundtrip_actual_raster_max_px=max(c['ray_to_actual_raster_projection_max_px'] for c in errors),per_frame=errors)
    (OUT/'actual_runtime_camera_verification.json').write_text(json.dumps(results,indent=2));print(json.dumps({e:{k:v for k,v in r.items() if k!='per_frame'} for e,r in results.items()},indent=2))

if __name__=='__main__':main()
