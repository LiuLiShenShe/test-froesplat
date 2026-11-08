"""Project saved source depth onto independently matched target RGB pixels."""
import json,csv
from pathlib import Path
import numpy as np
import cv2
from camera_numeric_audit import OMEGA,STANDARD,OUT,colmap

def main():
    ds=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
    frames=json.loads((ds/'per_frame/cameras.json').read_text())['frames'];Aref={f['image_name']:np.array(f['pixel_transform']) for f in frames}
    chain=json.loads((OUT/'camera_matrix_chain.json').read_text());match=json.loads((OUT/'epipolar_match_coordinates.json').read_text())
    cameras={'VGGT_Omega':colmap(ds),'Standard_VGGT':colmap(STANDARD/'Standard_VGGT_Dataset')};cache={};rows=[]
    for pair,q in match.items():
        i,j=map(int,pair.split('_'));gap=j-i
        if gap not in [1,5,15]:continue
        p1=np.array(q['p1']);p2=np.array(q['p2']);n1,n2=q['source1'],q['source2']
        original=np.c_[p1,np.ones(len(p1))]@np.linalg.inv(Aref[n1]).T
        for method,cams in cameras.items():
            c1,c2=cams[n1],cams[n2];record=chain[method+'/'+n1];A=np.array(record['A_recorded']);Km=np.array(record['K_model'])
            key=(method,n1)
            if key not in cache:
                if method=='VGGT_Omega':
                    d=np.load(ds/'per_frame'/('depth_'+Path(n1).stem+'.npy'));conf=np.load(ds/'per_frame'/('confidence_'+Path(n1).stem+'.npy'));threshold=1.252699375152588
                else:
                    d=np.load(STANDARD/'Standard_VGGT_Frontend/depth'/ (Path(n1).stem+'.npy'));conf=np.load(STANDARD/'Standard_VGGT_Frontend/confidence'/ (Path(n1).stem+'.npy'));threshold=1.0264525413513184
                cache[key]=(d.squeeze(),conf.squeeze(),threshold)
            d,conf,threshold=cache[key]
            pixels=(original@A.T)[:,:2];mapx=pixels[:,0].astype(np.float32)[None];mapy=pixels[:,1].astype(np.float32)[None]
            z=cv2.remap(d,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
            certainty=cv2.remap(conf,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
            rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(Km).T;Xcam=rays*z[:,None];pose=np.linalg.inv(c1['E']);world=Xcam@pose[:3,:3].T+pose[:3,3]
            dest=world@c2['E'][:3,:3].T+c2['E'][:3,3];Kdest=Aref[n2]@c2['K'];hom=dest@Kdest.T;uv=hom[:,:2]/hom[:,2:3]
            valid=np.isfinite(uv).all(axis=1)&np.isfinite(z)&(z>0)&(dest[:,2]>0)
            err=np.linalg.norm(uv-p2,axis=1)
            for scope,keep in [('all_matched',valid),('confidence_p20',valid&(certainty>=threshold))]:
                rows.append(dict(method=method,source1=n1,source2=n2,offset=gap,scope=scope,matches=len(p1),valid_count=int(keep.sum()),median_error_shared_grid_px=float(np.median(err[keep])) if keep.any() else None,p90_error_shared_grid_px=float(np.quantile(err[keep],.9)) if keep.any() else None,within_3px=float(np.mean(err[keep]<=3)) if keep.any() else None))
    with (OUT/'independent_match_depth_projection.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for method in cameras:
        for gap in [1,5,15]:
            for scope in ['all_matched','confidence_p20']:
                rr=[r for r in rows if r['method']==method and r['offset']==gap and r['scope']==scope and r['valid_count']>0]
                summary.append(dict(method=method,offset=gap,scope=scope,pairs=len(rr),median_error_shared_grid_px=float(np.median([r['median_error_shared_grid_px'] for r in rr])) if rr else None,p90_error_shared_grid_px=float(np.median([r['p90_error_shared_grid_px'] for r in rr])) if rr else None,within3px_pair_median=float(np.median([r['within_3px'] for r in rr])) if rr else None))
    (OUT/'independent_match_depth_projection_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
