"""Audit saved COLMAP cameras against actual preprocessing and GGGS projection.

No model inference, training or history writes. Pixel coordinates are integer
centres. Matrix projection below follows GGGS CUDA ndc2Pix exactly.
"""
from pathlib import Path
import csv
import importlib.util
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AREA = Path('/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区')
OMEGA = AREA / 'ForeSplat_VGGT_Omega_Frontend'
STANDARD = AREA / 'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison'
OUT = OMEGA / 'experiments/KongQueZhuYu/root_cause_audit/camera_projection_comparison'

spec = importlib.util.spec_from_file_location('audit_colmap_loader', OMEGA/'gggs_community/scene/colmap_loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)

def colmap(dataset):
    cameras = loader.read_intrinsics_binary(str(dataset/'sparse/0/cameras.bin'))
    images = loader.read_extrinsics_binary(str(dataset/'sparse/0/images.bin'))
    result = {}
    for image in images.values():
        c = cameras[image.camera_id]
        if c.model != 'PINHOLE': raise ValueError(c.model)
        fx, fy, cx, cy = c.params
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
        E = np.eye(4); E[:3,:3] = loader.qvec2rotmat(image.qvec); E[:3,3] = image.tvec
        result[image.name] = {'K': K, 'E': E, 'W': c.width, 'H': c.height}
    return result

def pil_resize_affine(sx, sy):
    return np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]])

def pinhole_project(K, E, xyz):
    xc = xyz@E[:3,:3].T+E[:3,3]
    uvw = xc@K.T
    return uvw[:,:2]/uvw[:,2:3]

def gg_projection(K, W, H):
    P=np.zeros((4,4)); P[0,0]=2*K[0,0]/W;P[1,1]=2*K[1,1]/H
    P[3,2]=1;P[2,2]=100/(100-.01);P[2,3]=-100*.01/(100-.01)
    return P

def cuda_project(P,E,xyz,W,H):
    homogeneous=np.c_[xyz,np.ones(len(xyz))]@(P@E).T
    ndc=homogeneous[:,:2]/homogeneous[:,3:4]
    return ((ndc+1)*np.array([W,H])-1)/2

def main():
    OUT.mkdir(exist_ok=True)
    omega_ds=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
    saved=json.loads((omega_ds/'per_frame/cameras.json').read_text())
    omega_frames={f['image_name']:f for f in saved['frames']}
    std_frames={f['image_name']:f for f in json.loads((STANDARD/'Standard_VGGT_Frontend/metadata/input_manifest.json').read_text())['frames']}
    sets={'VGGT_Omega':colmap(omega_ds),'Standard_VGGT':colmap(STANDARD/'Standard_VGGT_Dataset')}
    rows=[]; matrices={}; summary={}
    rng=np.random.default_rng(0)
    for method,cams in sets.items():
        frames=omega_frames if method=='VGGT_Omega' else std_frames
        model_wh=(384,688) if method=='VGGT_Omega' else (518,518)
        checkpoint=OMEGA/'experiments/KongQueZhuYu/E1_GGGS_Baseline/checkpoints/cameras.json' if method=='VGGT_Omega' else STANDARD/'E5_Standard_VGGT_GGGS_Baseline/checkpoints/cameras.json'
        stored={f['img_name']+'.png':f for f in json.loads(checkpoint.read_text())}
        method_rows=[]; names=sorted(cams)
        for name in names:
            c=cams[name];K=c['K'];E=c['E'];W,H=c['W'],c['H'];wt,ht=round(W/2),round(H/2)
            f=frames[name]; A=np.array(f['pixel_transform'] if method=='VGGT_Omega' else f['affine518'])
            Km=A@K
            Aactual=A.copy()
            if method=='Standard_VGGT': Aactual[:2,2]+=(A[0,0]-1)/2
            Ktrue=np.linalg.inv(Aactual)@Km
            B=pil_resize_affine(wt/W,ht/H);Kt_actual=B@K;Kt_true=B@Ktrue
            Kgg=np.array([[K[0,0]*wt/W,0,(wt-1)/2],[0,K[1,1]*ht/H,(ht-1)/2],[0,0,1]])
            xyz_cam=np.c_[rng.uniform(-.25,.25,(256,2)),rng.uniform(1,3,256)]
            xyz_world=(xyz_cam-E[:3,3])@E[:3,:3]
            p_gg=cuda_project(gg_projection(Kgg,wt,ht),E,xyz_world,wt,ht)
            p_expected=pinhole_project(Kgg,E,xyz_world)
            p_saved=pinhole_project(Kt_actual,E,xyz_world)
            p_true=pinhole_project(Kt_true,E,xyz_world)
            z=stored[name];C=np.linalg.inv(E)
            align_error=None
            if method=='VGGT_Omega':
                Eraw=np.eye(4);Eraw[:3,:4]=f['extrinsic_model_w2c']
                alignment=np.array(saved['scene_alignment_matrix'])
                align_error=float(np.max(np.abs(E-Eraw@np.linalg.inv(alignment))))
            row=dict(method=method,name=name,W_original=W,H_original=H,W_model=model_wh[0],H_model=model_wh[1],W_train=wt,H_train=ht,
                fx_original=K[0,0],fy_original=K[1,1],cx_original=K[0,2],cy_original=K[1,2],
                fx_over_W=K[0,0]/W,fy_over_H=K[1,1]/H,cx_over_W=K[0,2]/W,cy_over_H=K[1,2]/H,
                cx_minus_W_half=K[0,2]-W/2,cy_minus_H_half=K[1,2]-H/2,
                fx_model=Km[0,0],fy_model=Km[1,1],cx_model=Km[0,2],cy_model=Km[1,2],
                cx_train_saved=Kt_actual[0,2],cy_train_saved=Kt_actual[1,2],cx_train_true_pixel_transform=Kt_true[0,2],cy_train_true_pixel_transform=Kt_true[1,2],
                cx_gggs=Kgg[0,2],cy_gggs=Kgg[1,2],
                principal_loss_train_px=float(np.linalg.norm(Kt_actual[:2,2]-Kgg[:2,2])),
                actual_pixel_transform_to_gggs_offset_px=float(np.linalg.norm(Kt_true[:2,2]-Kgg[:2,2])),
                cuda_to_gggs_K_max_error_px=float(np.max(np.linalg.norm(p_gg-p_expected,axis=1))),
                saved_K_to_raster_median_error_px=float(np.median(np.linalg.norm(p_saved-p_gg,axis=1))),
                true_pixel_K_to_raster_median_error_px=float(np.median(np.linalg.norm(p_true-p_gg,axis=1))),
                rotation_orthogonality_max=float(np.max(np.abs(E[:3,:3]@E[:3,:3].T-np.eye(3)))),rotation_det=float(np.linalg.det(E[:3,:3])),
                colmap_to_checkpoint_camera_center_error=float(np.linalg.norm(C[:3,3]-z['position'])),
                colmap_to_checkpoint_rotation_max=float(np.max(np.abs(C[:3,:3]-np.array(z['rotation'])))),
                colmap_to_checkpoint_fx_error=abs(K[0,0]-z['fx']),colmap_to_checkpoint_fy_error=abs(K[1,1]-z['fy']),
                raw_to_aligned_w2c_max=align_error)
            rows.append(row);method_rows.append(row)
            matrices[method+'/'+name]=dict(K_colmap=K.tolist(),K_model=Km.tolist(),K_train_saved=Kt_actual.tolist(),K_train_true_pixel_transform=Kt_true.tolist(),K_gggs=Kgg.tolist(),W2C_colmap=E.tolist(),A_recorded=A.tolist(),A_pixel_centres=Aactual.tolist(),B_training_resize=B.tolist())
        summary[method]={key:{'min':float(np.min([r[key] for r in method_rows if r[key] is not None])), 'median':float(np.median([r[key] for r in method_rows if r[key] is not None])), 'max':float(np.max([r[key] for r in method_rows if r[key] is not None]))} for key in ['fx_over_W','fy_over_H','cx_minus_W_half','cy_minus_H_half','principal_loss_train_px','actual_pixel_transform_to_gggs_offset_px','cuda_to_gggs_K_max_error_px','colmap_to_checkpoint_camera_center_error','rotation_orthogonality_max']}
        if method=='VGGT_Omega':summary[method]['raw_to_aligned_w2c_max']=max(r['raw_to_aligned_w2c_max'] for r in method_rows)
    with (OUT/'intrinsic_chain_per_frame.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'camera_matrix_chain.json').write_text(json.dumps(matrices,indent=2))
    (OUT/'camera_numeric_summary.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(2,3,figsize=(12,6))
    for method in sets:
        rr=[r for r in rows if r['method']==method]
        for a,k in zip(ax.flat,['fx_over_W','fy_over_H','cx_over_W','cy_over_H','cx_minus_W_half','cy_minus_H_half']):
            a.plot([r[k] for r in rr],label=method);a.set_title(k);a.set_xlabel('sequence index')
    ax[0,0].legend();fig.tight_layout();fig.savefig(OUT/'intrinsics_distributions.png',dpi=180);fig.savefig(OUT/'intrinsics_distributions.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
