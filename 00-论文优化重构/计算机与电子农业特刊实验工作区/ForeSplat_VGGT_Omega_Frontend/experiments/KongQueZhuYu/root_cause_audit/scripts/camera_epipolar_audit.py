"""Independent RGB feature evidence for saved cameras. No depth is used.

Features come from the same archived RGB/mask grid for both models. Mutual
SIFT Lowe 0.75 matches are selected by image-only F RANSAC (1.5px), then
tested against each saved model F. Errors are in the shared 384x688 grid.
This is a correspondence diagnostic, not a calibrated camera ground truth.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,json,time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from camera_numeric_audit import OMEGA, STANDARD, OUT, colmap

def skew(t):
    x,y,z=t;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def fundamental(K1,K2,E1,E2):
    relative=E2@np.linalg.inv(E1)
    return np.linalg.inv(K2).T@skew(relative[:3,3])@relative[:3,:3]@np.linalg.inv(K1)

def sampson(F,p1,p2):
    x=np.c_[p1,np.ones(len(p1))];y=np.c_[p2,np.ones(len(p2))]
    l2=x@F.T;l1=y@F
    return np.abs(np.sum(y*l2,axis=1))/np.sqrt(np.maximum(np.sum(l1[:,:2]**2+l2[:,:2]**2,axis=1),1e-30))

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    ds=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
    frames=json.loads((ds/'per_frame/cameras.json').read_text())['frames']
    names=[f['image_name'] for f in frames];A={f['image_name']:np.array(f['pixel_transform']) for f in frames}
    methods={'VGGT_Omega':colmap(ds),'Standard_VGGT':colmap(STANDARD/'Standard_VGGT_Dataset')}
    feature_dir=OUT/'sift_features';feature_dir.mkdir(exist_ok=True)
    def extract(name):
        file=feature_dir/(Path(name).stem+'.npz')
        if file.exists():
            z=np.load(file);return name,dict(points=z['points'],descriptors=z['descriptors'],edge=z['edge'])
        stem=Path(name).stem;rgb=np.load(ds/'per_frame'/('model_rgb_'+stem+'.npy'))
        if rgb.shape[0]==3:rgb=rgb.transpose(1,2,0)
        rgb=np.clip(np.rint(rgb*255),0,255).astype(np.uint8) if rgb.dtype!=np.uint8 else rgb
        mask=np.load(ds/'per_frame'/('model_mask_'+stem+'.npy')).astype(np.uint8)*255
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        sift=cv2.SIFT_create(nfeatures=1200);key,des=sift.detectAndCompute(gray,mask)
        points=np.array([k.pt for k in key],dtype=np.float32).reshape(-1,2)
        if des is None:des=np.empty((0,128),np.float32)
        dist=cv2.distanceTransform(mask,cv2.DIST_L2,3)
        ij=np.rint(points).astype(int);edge=dist[np.clip(ij[:,1],0,mask.shape[0]-1),np.clip(ij[:,0],0,mask.shape[1]-1)]<=4
        np.savez(file,points=points,descriptors=des,edge=edge)
        return name,dict(points=points,descriptors=des,edge=edge)
    t0=time.time();features=dict(ThreadPoolExecutor(4).map(extract,names));print('features_complete',time.time()-t0,flush=True)
    bf=cv2.BFMatcher(cv2.NORM_L2);rows=[];match_archive={};offsets=[1,5,15,30,52,104]
    for gap in offsets:
        pairs=list(range(0,len(names)-gap,3))
        for i in pairs:
            j=i+gap;n1,n2=names[i],names[j];f1,f2=features[n1],features[n2]
            d1,d2=f1['descriptors'],f2['descriptors']
            if min(len(d1),len(d2))<2:continue
            a=bf.knnMatch(d1,d2,k=2);b=bf.knnMatch(d2,d1,k=2)
            reverse={m.queryIdx:m.trainIdx for mm in b if len(mm)==2 for m,n in [mm] if m.distance<.75*n.distance}
            good=[m for mm in a if len(mm)==2 for m,n in [mm] if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
            p1=np.array([f1['points'][m.queryIdx] for m in good],dtype=np.float32).reshape(-1,2)
            p2=np.array([f2['points'][m.trainIdx] for m in good],dtype=np.float32).reshape(-1,2)
            edge=np.array([f1['edge'][m.queryIdx] or f2['edge'][m.trainIdx] for m in good])
            if len(good)>=12:
                Fimg,ins=cv2.findFundamentalMat(p1,p2,cv2.FM_RANSAC,1.5,.999)
            else:Fimg,ins=None,None
            accepted=Fimg is not None and Fimg.shape==(3,3) and ins is not None and int(ins.sum())>=12
            for method,cams in methods.items():
                c1,c2=cams[n1],cams[n2];K1=A[n1]@c1['K'];K2=A[n2]@c2['K']
                F=fundamental(K1,K2,c1['E'],c2['E'])
                row=dict(method=method,source1=n1,source2=n2,offset=gap,mutual_matches=len(good),image_F_inliers=int(ins.sum()) if ins is not None else 0,pair_accepted=accepted)
                for key in ['median_sampson_px','p90_sampson_px','within_1_5px_fraction','interior_median_px','boundary_median_px','positive_triangulation_fraction','image_F_median_px']:row[key]=None
                if accepted:
                    mask=ins.ravel().astype(bool);q1,q2=p1[mask],p2[mask];err=sampson(F,q1,q2);ee=edge[mask]
                    row.update(median_sampson_px=float(np.median(err)),p90_sampson_px=float(np.quantile(err,.9)),within_1_5px_fraction=float(np.mean(err<=1.5)),interior_median_px=float(np.median(err[~ee])) if (~ee).any() else None,boundary_median_px=float(np.median(err[ee])) if ee.any() else None,image_F_median_px=float(np.median(sampson(Fimg,q1,q2))))
                    X=cv2.triangulatePoints(K1@c1['E'][:3],K2@c2['E'][:3],q1.T.astype(float),q2.T.astype(float));X=(X[:3]/X[3]).T
                    z1=(X@c1['E'][:3,:3].T+c1['E'][:3,3])[:,2];z2=(X@c2['E'][:3,:3].T+c2['E'][:3,3])[:,2]
                    row['positive_triangulation_fraction']=float(np.mean((z1>0)&(z2>0)))
                    if i%15==0:match_archive[f'{i}_{j}']=dict(source1=n1,source2=n2,p1=q1.tolist(),p2=q2.tolist())
                rows.append(row)
        print('offset_complete',gap,flush=True)
    with (OUT/'epipolar_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for method in methods:
        for gap in offsets:
            rr=[r for r in rows if r['method']==method and r['offset']==gap];ok=[r for r in rr if r['pair_accepted']]
            entry=dict(method=method,offset=gap,pairs_attempted=len(rr),pairs_accepted=len(ok))
            for key in ['median_sampson_px','p90_sampson_px','within_1_5px_fraction','positive_triangulation_fraction','image_F_median_px']:
                vals=[r[key] for r in ok if r[key] is not None];entry[key]=float(np.median(vals)) if vals else None
            summary.append(entry)
    (OUT/'epipolar_summary.json').write_text(json.dumps({'protocol':{'opencv':cv2.__version__,'features':'SIFT 1200, foreground mask, mutual Lowe .75','image_only_selection':'FM_RANSAC threshold1.5 shared-grid px; minimum12 inliers','depth_used':False,'shared_grid_WH':[384,688],'stride':3,'offsets':offsets},'summary':summary},indent=2))
    (OUT/'epipolar_match_coordinates.json').write_text(json.dumps(match_archive))
    fig,ax=plt.subplots(1,2,figsize=(9,3.5))
    for method in methods:
        ss=[s for s in summary if s['method']==method]
        for a,key in zip(ax,['median_sampson_px','within_1_5px_fraction']):
            a.plot(offsets,[s[key] for s in ss],'-o',label=method);a.set_title(key);a.set_xlabel('view offset')
    ax[0].legend();fig.tight_layout();fig.savefig(OUT/'independent_rgb_epipolar.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
