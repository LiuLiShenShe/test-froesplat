"""Use archived Fast3R/AMB3R-SfM poses as references on the same RGB matches."""
import json,csv
from pathlib import Path
import numpy as np
from camera_numeric_audit import AREA,OUT,OMEGA,STANDARD,colmap
from camera_epipolar_audit import fundamental,sampson

def main():
    frames=json.loads((OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset/per_frame/cameras.json').read_text())['frames'];A={f['image_name']:np.array(f['pixel_transform']) for f in frames}
    cams={'VGGT_Omega':colmap(OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'),'Standard_VGGT':colmap(STANDARD/'Standard_VGGT_Dataset')}
    for m in ['Fast3R','AMB3R_SfM','Fast3R_AMB3R_SfM']:
        z=np.load(AREA/'ForeSplat_Fast3R_AMB3R_Frontend/experiments/KongQueZhuYu'/m/'cameras/cameras.npz')
        cams[m]={str(n):dict(K=z['intrinsics_original'][i],E=z['w2c'][i]) for i,n in enumerate(z['names'])}
    match=json.loads((OUT/'epipolar_match_coordinates.json').read_text());rows=[]
    for pair,q in match.items():
        i,j=map(int,pair.split('_'));gap=j-i
        if gap not in [1,5,15]:continue
        p1,p2=np.array(q['p1']),np.array(q['p2']);n1,n2=q['source1'],q['source2']
        for method,c in cams.items():
            err=sampson(fundamental(A[n1]@c[n1]['K'],A[n2]@c[n2]['K'],c[n1]['E'],c[n2]['E']),p1,p2)
            rows.append(dict(method=method,pair=pair,offset=gap,matches=len(p1),median_sampson_px=float(np.median(err)),within1_5px=float(np.mean(err<=1.5))))
    with (OUT/'reference_epipolar_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for m in cams:
        for gap in [1,5,15]:
            r=[x for x in rows if x['method']==m and x['offset']==gap]
            summary.append(dict(method=m,offset=gap,pairs=len(r),median_sampson_px=float(np.median([x['median_sampson_px'] for x in r])) if r else None,within1_5px_median=float(np.median([x['within1_5px'] for x in r])) if r else None))
    (OUT/'reference_epipolar_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
