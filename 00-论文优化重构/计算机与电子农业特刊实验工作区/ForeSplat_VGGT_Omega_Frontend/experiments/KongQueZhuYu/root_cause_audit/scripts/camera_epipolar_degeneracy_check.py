"""Reject collapsed-pose/zero-F artifacts in the decisive background control."""
from pathlib import Path
import json
import numpy as np
from camera_epipolar_audit import fundamental,sampson

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parents[2]/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
CONTROL=ROOT/'controlled_fixes/background_control_v1'

def main():
    refs={int(Path(r['image_name']).stem.split('_')[-1]):np.asarray(r['pixel_transform'])
          for r in json.loads((DATA/'metadata/preprocess_transforms.json').read_text())['frames']}
    rows=[]
    for model in ['Omega','Standard']:
        for condition in ['RAP_baseline','background_restored']:
            cams=json.loads((CONTROL/f'{model}_{condition}_24/cameras.json').read_text())['frames']
            lookup={int(r['source_index']):r for r in cams}
            for pair in sorted((CONTROL/'evaluation/fixed_foreground_matches').glob('*.npz')):
                sid,tid=map(int,pair.stem.split('_'));a=lookup[sid];b=lookup[tid]
                with np.load(pair,allow_pickle=False) as f:
                    assert bool(f['accepted'])
                    p1,p2=f['source_points'],f['target_points']
                k1=refs[sid]@np.linalg.inv(a['pixel_transform_pixel_center'])@np.asarray(a['intrinsic_model'])
                k2=refs[tid]@np.linalg.inv(b['pixel_transform_pixel_center'])@np.asarray(b['intrinsic_model'])
                e1,e2=np.asarray(a['extrinsic_model_w2c']),np.asarray(b['extrinsic_model_w2c'])
                t=(e2@np.linalg.inv(e1))[:3,3]
                f=fundamental(k1,k2,e1,e2);norm=float(np.linalg.norm(f))
                if norm<=1e-12 or np.linalg.norm(t)<=1e-10:raise AssertionError('Degenerate relative translation/F')
                x=np.c_[p1,np.ones(len(p1))];y=np.c_[p2,np.ones(len(p2))]
                l1=y@f;l2=x@f.T;den=np.sum(l1[:,:2]**2+l2[:,:2]**2,axis=1)
                delta=float(np.max(np.abs(sampson(f,p1,p2)-sampson(f/norm,p1,p2))))
                if np.any(den<=1e-30) or delta>1e-8:raise AssertionError('Sampson floor or normalization artifact')
                rows.append(dict(model=model,condition=condition,source=sid,target=tid,
                                 relative_translation_norm_native=float(np.linalg.norm(t)),F_frobenius_norm=norm,
                                 denominator_min=float(den.min()),denominator_floor_points=int((den<=1e-30).sum()),
                                 normalized_F_error_max_difference_px=delta))
    out=dict(status='passed',pairs_checked=len(rows),records=rows,
             scope='Actual48condition/pair camera+same RGB inliers; F normalization and nonzero translation checked. No inference/training, no true pose/depth claim.')
    (CONTROL/'evaluation/epipolar_degeneracy_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'passed','pairs':len(rows),'floor_points':sum(r['denominator_floor_points'] for r in rows),
                      'max_normalization_difference_px':max(r['normalized_F_error_max_difference_px'] for r in rows)}))

if __name__=='__main__':main()
