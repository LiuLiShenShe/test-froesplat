"""Read-only CPU utilities for the saved frontend root-cause audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,hashlib
import numpy as np

AUDIT=Path(__file__).resolve().parents[1]
OMEGA=Path(__file__).resolve().parents[4]
WORKSPACE=OMEGA.parent
PREVIOUS=WORKSPACE/'ForeSplat_Fast3R_AMB3R_Frontend'/'experiments/KongQueZhuYu'
STANDARD=WORKSPACE/'ForeSplat_Standard_VGGT_Frontend'/'experiments/KongQueZhuYu/standard_vggt_comparison'
OMEGA_DATA=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def q(values,percentile):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return float(np.quantile(a,percentile)) if a.size else None

def source_id(name):return int(Path(name).stem.split('_')[-1])

def quaternion_rotation(qw,qx,qy,qz):
    vector=np.array([qw,qx,qy,qz],float);vector/=np.linalg.norm(vector)
    w,x,y,z=vector
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def read_colmap_cameras(directory):
    directory=Path(directory);intrinsic={};poses={};ids={};dimensions={}
    with (directory/'cameras.txt').open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            s=line.split();camera_id=int(s[0]);assert s[1]=='PINHOLE'
            fx,fy,cx,cy=map(float,s[4:8]);intrinsic[camera_id]=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            dimensions[camera_id]=[int(s[2]),int(s[3])]
    with (directory/'images.txt').open() as f:
        while True:
            line=f.readline()
            if not line:break
            if not line.strip() or line.startswith('#'):continue
            s=line.split();assert len(s)>=10
            name=s[9];image_id=int(s[0]);camera_id=int(s[8]);e=np.eye(4)
            e[:3,:3]=quaternion_rotation(*map(float,s[1:5]));e[:3,3]=list(map(float,s[5:8]))
            if name in poses:raise ValueError('Duplicate COLMAP source filename')
            poses[name]=e;ids[name]=[image_id,camera_id];f.readline() # skip real point2D observations without parsing them
    return intrinsic,poses,ids,dimensions

def camera_sources():
    sources={}
    raw=json.loads((OMEGA_DATA/'per_frame/cameras.json').read_text())['frames']
    transforms=json.loads((OMEGA_DATA/'metadata/preprocess_transforms.json').read_text())['frames']
    by_name={r['image_name']:r for r in transforms};names=[r['image_name'] for r in raw]
    e=[];k=[];a=[];size=[]
    for r in raw:
        pose=np.eye(4);pose[:3]=r['extrinsic_model_w2c'];e.append(pose);k.append(r['intrinsic_model']);t=by_name[r['image_name']]
        a.append(t['pixel_transform']);size.append([t['model_width'],t['model_height']])
    sources['Omega_depth']=dict(names=names,w2c=np.asarray(e),c2w=np.linalg.inv(e),k_model=np.asarray(k),a_actual=np.asarray(a),a_recorded=np.asarray(a),size_wh=size,path=str(OMEGA_DATA))
    meta=json.loads((STANDARD/'Standard_VGGT_Frontend/metadata.json').read_text())
    manifest=json.loads((STANDARD/'Standard_VGGT_Frontend/metadata/input_manifest.json').read_text())['frames']
    records={r['image_name']:r for r in manifest}
    km,ep,ids,dimensions=read_colmap_cameras(STANDARD/'Standard_VGGT_Dataset/sparse/0')
    names=meta['image_names'];assert set(names)==set(ep)==set(records)
    e=np.array([ep[n] for n in names]);recorded=np.array([records[n]['affine518'] for n in names]);actual=recorded.copy()
    actual[:,0,2]+=(actual[:,0,0]-1)/2;actual[:,1,2]+=(actual[:,1,1]-1)/2
    k=np.array([recorded[i]@km[ids[n][1]] for i,n in enumerate(names)])
    sources['Standard_depth']=dict(names=names,w2c=e,c2w=np.linalg.inv(e),k_model=k,a_actual=actual,a_recorded=recorded,
        size_wh=[[518,518]]*len(names),path=str(STANDARD/'Standard_VGGT_Frontend'),colmap_ids=ids,
        camera_storage='raw K/extrinsic arrays absent; K_model recovered from saved A_recorded K_original, W2C recovered from exported COLMAP quaternion/translation; no BA in producer')
    for method in ['Fast3R','Fast3R_FP32','AMB3R_Base_OFF','AMB3R_Base_ON','AMB3R_SfM','Fast3R_AMB3R_SfM']:
        p=PREVIOUS/method;b=json.loads((p/'bundle.json').read_text());t=json.loads((p/'pixel_transforms.json').read_text())
        with np.load(p/'cameras/cameras.npz') as f:c=dict(f)
        assert list(c['names'])==b['input_names']
        sources[method]=dict(names=b['input_names'],w2c=c['w2c'],c2w=c['c2w'],k_model=c['intrinsics_model'],
            a_actual=np.array([r['pixel_transform'] for r in t]),a_recorded=np.array([r['pixel_transform'] for r in t]),
            size_wh=[r['input_size_wh'] for r in t],path=str(p),bundle=b,transforms=t)
    return sources

def camera_angles(relative):
    return np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))

def fit_similarity(source,target):
    s0=source-source.mean(0);t0=target-target.mean(0);cov=t0.T@s0/len(source)
    u,sv,vt=np.linalg.svd(cov);reflection=np.linalg.det(u@vt)<0
    d=np.diag([1.,1.,-1. if reflection else 1.]);r=u@d@vt
    scale=float(np.sum(sv*np.diag(d))/max(np.mean(np.sum(s0*s0,axis=1)),1e-15))
    return scale,r,target.mean(0)-scale*r@source.mean(0)
