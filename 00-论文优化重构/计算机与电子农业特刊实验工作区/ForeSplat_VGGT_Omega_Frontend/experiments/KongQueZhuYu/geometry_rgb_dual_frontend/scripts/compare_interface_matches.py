"""Actual foreground RGB correspondences through old/corrected GGGS full P."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import csv,json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_erosion
ROOT=Path(__file__).resolve().parents[1]
def main():
    sources=json.loads((ROOT/'configs/prepared_sources.json').read_text());byid={r['source_index']:r for r in sources}
    pairs=[r for r in csv.DictReader((ROOT/'geometry_comparison/fixed_pair_manifest.csv').open()) if r['common_rank_offset']=='1' and r['accepted']=='True']
    records=[];summary=[]
    for model,dirname in [('Omega','VGGT_Omega_Original_RGB'),('Standard','Standard_VGGT_Original_RGB')]:
        out=ROOT/dirname
        runtime={interface:{r['source_id']:r for r in json.loads((ROOT/'camera_interface'/f'{model}_{interface}_runtime_parameters.json').read_text())} for interface in ['original','corrected']}
        alignment=np.array(json.loads((out/'fusion_metadata.json').read_text())['scene_alignment_matrix']) if model=='Omega' else np.eye(4)
        threshold=json.loads((out/'export_complete.json').read_text())['confidence_threshold']
        for pair in pairs:
            sid,tid=int(pair['source_id']),int(pair['target_id']);stem=f'crop_{sid:04d}'
            with np.load(ROOT/'geometry_comparison/fixed_foreground_matches'/f'{sid:04d}_{tid:04d}.npz') as z:p1=z['source_points'];p2=z['target_points']
            source=runtime['corrected'][sid]['frontend_record'];a=np.array(source['pixel_transform_pixel_center']);k=np.array(source['intrinsic_model']);e=np.array(source['extrinsic_model_w2c'])
            original=np.c_[p1,np.ones(len(p1))]@np.linalg.inv(np.array(byid[sid]['omega_pixel_transform'])).T
            uv=original@a.T
            depth=np.load(out/'per_frame'/f'depth_{stem}.npy');confidence=np.load(out/'per_frame'/f'confidence_{stem}.npy');mask=np.load(out/'per_frame'/f'model_mask_{stem}.npy')
            kwargs=dict(borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            depth=cv2.remap(depth,uv[:,0].astype(np.float32)[None],uv[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,**kwargs).ravel()
            confidence=cv2.remap(confidence,uv[:,0].astype(np.float32)[None],uv[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,**kwargs).ravel()
            mask=cv2.remap(mask,uv[:,0].astype(np.float32)[None],uv[:,1].astype(np.float32)[None],cv2.INTER_NEAREST,**kwargs).ravel()>0
            local=uv@np.linalg.inv(k).T*depth[:,None];world=(local-e[:3,3])@e[:3,:3];world=world@alignment[:3,:3].T+alignment[:3,3]
            valid=mask&np.isfinite(world).all(1)&(depth>0)&(confidence>=threshold)
            common_mask=np.load(ROOT/'VGGT_Omega_Original_RGB/per_frame'/f'model_mask_{stem}.npy').astype(bool)
            edge=common_mask&~binary_erosion(common_mask,iterations=3)
            leaf_edge=cv2.remap(edge.astype(np.uint8),p1[:,0][None],p1[:,1][None],cv2.INTER_NEAREST,**kwargs).ravel()>0
            original_target=np.c_[p2,np.ones(len(p2))]@np.linalg.inv(np.array(byid[tid]['omega_pixel_transform'])).T
            observed=(original_target[:,:2]+.5)/2-.5
            for interface in ['original','corrected']:
                camera=runtime[interface][tid];matrix=np.array(camera['full_proj_transform']);hom=np.c_[world,np.ones(len(world))]@matrix
                uv_render=(hom[:,:2]/hom[:,3,None]+1)*np.array([camera['width'],camera['height']])/2-.5
                error=np.linalg.norm(uv_render-observed,axis=1);eligible=valid&np.isfinite(error)&(hom[:,3]>0)
                for region,selection in [('foreground',eligible),('common_foreground_boundary',eligible&leaf_edge)]:
                    values=error[selection]
                    records.append(dict(model=model,interface=interface,source_id=sid,target_id=tid,region=region,matched_points=len(p1),eligible_points=len(values),
                        reprojection_p50_training_px=float(np.median(values)) if len(values) else None,reprojection_p90_training_px=float(np.quantile(values,.9)) if len(values) else None))
        for interface in ['original','corrected']:
            for region in ['foreground','common_foreground_boundary']:
                chosen=[r for r in records if r['model']==model and r['interface']==interface and r['region']==region and r['eligible_points']>0]
                summary.append(dict(model=model,interface=interface,region=region,pairs=len(chosen),eligible_points=sum(r['eligible_points'] for r in chosen),
                    median_pair_reprojection_p50_training_px=float(np.median([r['reprojection_p50_training_px'] for r in chosen])) if chosen else None))
    for filename,rows in [('matched_RGB_reprojection.csv',records),('matched_RGB_reprojection_summary.csv',summary)]:
        with (ROOT/'camera_interface'/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'camera_interface/matched_RGB_reprojection_protocol.json').write_text(json.dumps(dict(shared_coordinate_system='1080x1920 actual GGGS training pixels, integer centers; observed original RGB correspondences transformed with half-pixel resize',
        fixed_pairs='Same near-frame RAP image-only RANSAC correspondences as foreground geometry audit',confidence='Original frontend p20 threshold, unchanged within interface pair',
        edge='Same 3px inner boundary on shared Omega384x688 original mask; no model-selected mask tuning',
        interpretation='Real source depth projected through target runtime full P to independently observed target RGB. Combines frontend residual and interface shift; not same-view self-projection or absolute 3D GT'),indent=2))
    print('Actual old/corrected cross-view RGB reprojection complete',flush=True)
if __name__=='__main__':main()
