"""Separate focal and relative-pose factors using fixed RGB correspondences.

This does not exchange world-coordinate point clouds. Relative W2C transforms
are invariant to either model's global rigid/similarity world gauge. Principal
point is fixed to the same original image centre for all four conditions.
"""
import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from camera_numeric_audit import OMEGA,STANDARD,OUT,colmap
from camera_epipolar_audit import fundamental,sampson

def main():
    ds=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
    frames=json.loads((ds/'per_frame/cameras.json').read_text())['frames'];A={f['image_name']:np.array(f['pixel_transform']) for f in frames}
    cams={'Omega':colmap(ds),'Standard':colmap(STANDARD/'Standard_VGGT_Dataset')}
    q=json.loads((OUT/'epipolar_match_coordinates.json').read_text());rows=[]
    for pair,coords in q.items():
        i,j=map(int,pair.split('_'));gap=j-i
        if gap not in [1,5]:continue
        n1,n2=coords['source1'],coords['source2'];p1,p2=np.array(coords['p1']),np.array(coords['p2'])
        for pose in cams:
            for focal in cams:
                f1,f2=cams[focal][n1],cams[focal][n2];K1=f1['K'].copy();K2=f2['K'].copy()
                K1[:2,2]=[1080.,1920.];K2[:2,2]=[1080.,1920.]
                E1,E2=cams[pose][n1]['E'],cams[pose][n2]['E']
                err=sampson(fundamental(A[n1]@K1,A[n2]@K2,E1,E2),p1,p2)
                rows.append(dict(pair=pair,offset=gap,pose=pose,focal=focal,points=len(p1),median_sampson_px=float(np.median(err)),within1_5px=float(np.mean(err<=1.5))))
    dest=Path(__file__).resolve().parent
    with (dest/'camera_factor_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for gap in [1,5]:
        for pose in cams:
            for focal in cams:
                r=[x for x in rows if x['offset']==gap and x['pose']==pose and x['focal']==focal]
                summary.append(dict(offset=gap,pose=pose,focal=focal,pairs=len(r),median_pair_sampson_px=float(np.median([x['median_sampson_px'] for x in r])),median_within1_5px=float(np.median([x['within1_5px'] for x in r]))))
    (dest/'camera_factor_results.json').write_text(json.dumps({'experiment':'fixed RGB correspondence; factorial relative pose and focal control; principal centre held fixed','new_model_inference':False,'new_training':False,'world_point_exchange':False,'results':summary,'limits':'Local epipolar diagnostic; not a globally calibrated 208-pose bundle and not a Gaussian retraining claim.'},indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
