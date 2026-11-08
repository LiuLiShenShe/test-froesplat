#!/usr/bin/env python3
"""CPU-only diagnostics of the saved AMB3R depth-unprojection channel.

This creates tables only. Original pointmaps, poses, confidence and scale remain
unchanged; no geometry is aligned, fused, fitted, or relabeled as a local head.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from frontends.common import EXPERIMENT,FrontendBundle,json_write
from frontends.evaluate import adjacent_consistency,edge_discontinuity,projected,safe_quantile,sample_pixels,saved_npz,table_write,to_camera,valid_camera_indices


def run(bundle):
    frames=[]
    for i,name in enumerate(bundle.metadata['input_names']):
        stem=Path(name).stem
        with saved_npz(bundle.path/'pointmaps'/f'{stem}.npz') as p:
            xyz=p['pts3d_by_unprojection'];depth_metric=p['depth_metric'].reshape(xyz.shape[:2])
        with saved_npz(bundle.path/'confidence'/f'{stem}.npz') as p:
            depth_conf=p['depth_conf'];world_conf=p['world_points_conf']
        mask=np.asarray(Image.open(bundle.path/'mask'/name))>0;xyz=xyz.reshape(*mask.shape,3)
        t=bundle.pixel_transforms[i];left,top,right,bottom=t['padding_ltrb'];unpadded=np.zeros(mask.shape,bool)
        unpadded[top:mask.shape[0]-bottom,left:mask.shape[1]-right]=True
        frames.append(dict(name=name,xyz=xyz,mask=mask,c2w=bundle.cameras['c2w'][i],k=bundle.cameras['intrinsics_model'][i],
            finite=np.isfinite(xyz).all(-1)&unpadded,depth_conf=depth_conf.reshape(mask.shape),world_conf=world_conf.reshape(mask.shape),depth_metric=depth_metric))
        if (i+1)%32==0:print(f'{bundle.path.name}: loaded {i+1}/{len(bundle.metadata["input_names"])} saved depth maps',flush=True)
    anchor_indices=valid_camera_indices(bundle);anchor=bundle.cameras['c2w'][anchor_indices[0]] if len(anchor_indices) else None
    results=[]
    for branch,conf_key in [('native_depth_confidence','depth_conf'),('same_world_confidence_geometry_control','world_conf')]:
        for f in frames:
            conf=f[conf_key];f['conf']=conf;f['valid']=f['finite']&np.isfinite(conf)&(conf>1)
        clouds=[]
        for f in frames:
            ij=sample_pixels(f,1000);clouds.append(f['xyz'][ij[:,0],ij[:,1]])
        sample=np.concatenate(clouds);display=to_camera(sample,anchor) if anchor is not None else sample
        extent=max(float(np.ptp(np.quantile(display,[.025,.975],axis=0),axis=0).max()),1e-12) if len(display) else 1.
        frame_rows=[];pairs=[];prior=None
        for f in frames:
            mask=f['mask'];valid=f['valid'];edge=mask&~binary_erosion(mask);sel=mask&valid
            ij=sample_pixels(f,6000);pts=f['xyz'][ij[:,0],ij[:,1]];uv,z=projected(pts,f['c2w'],f['k'])
            error=np.linalg.norm(uv-ij[:,::-1],axis=-1)[z>0]
            frame_rows.append(dict(name=f['name'],foreground_pixels=int(mask.sum()),foreground_valid=int(sel.sum()),edge_pixels=int(edge.sum()),edge_valid=int((edge&valid).sum()),
                depth_neighbor_jump_ratio=edge_discontinuity(f),self_reprojection_p50_px=safe_quantile(error,.5),
                depth_metric_all_foreground_p10_m=safe_quantile(f['depth_metric'][mask],.1),depth_metric_all_foreground_p50_m=safe_quantile(f['depth_metric'][mask],.5),depth_metric_all_foreground_p90_m=safe_quantile(f['depth_metric'][mask],.9)))
            if prior is not None:
                for src,tgt in ((prior,f),(f,prior)):pairs.append(dict(source=src['name'],target=tgt['name'],**adjacent_consistency(src,tgt,extent)))
            prior=f
        fg=sum(r['foreground_pixels'] for r in frame_rows);edges=sum(r['edge_pixels'] for r in frame_rows)
        summary=dict(method=bundle.path.name,geometry_channel='official pts3d_by_unprojection in world coordinates; not a local prediction head',
            confidence_control=branch,confidence_field=conf_key,views=len(frames),foreground_coverage=sum(r['foreground_valid'] for r in frame_rows)/max(fg,1),
            edge_coverage=sum(r['edge_valid'] for r in frame_rows)/max(edges,1),normalization_extent=extent,
            depth_neighbor_jump_ratio=safe_quantile([r['depth_neighbor_jump_ratio'] for r in frame_rows if r['depth_neighbor_jump_ratio'] is not None],.5),
            self_reprojection_p50_px=safe_quantile([r['self_reprojection_p50_px'] for r in frame_rows if r['self_reprojection_p50_px'] is not None],.5),
            adjacent_depth_agreement_fraction=safe_quantile([p['depth_agreement_fraction'] for p in pairs if p.get('depth_agreement_fraction') is not None],.5),
            adjacent_point_distance_p50_normalized=safe_quantile([p['distance_p50'] for p in pairs if p.get('distance_p50') is not None],.5),
            projection_overlap_fraction=sum(p['overlap'] for p in pairs)/max(sum(p['projected'] for p in pairs),1),
            depth_metric_view_median_p10_m=safe_quantile([r['depth_metric_all_foreground_p50_m'] for r in frame_rows],.1),
            depth_metric_view_median_p50_m=safe_quantile([r['depth_metric_all_foreground_p50_m'] for r in frame_rows],.5),
            depth_metric_view_median_p90_m=safe_quantile([r['depth_metric_all_foreground_p50_m'] for r in frame_rows],.9),
            interpretation='Saved depth-channel consistency only. Self-projection is algebraic, not GT accuracy. Native depth confidence and same-world-confidence isolation are separate controls. No scale fit or output changes.')
        table_write(bundle.path/f'depth_diagnostic_{branch}_per_frame.csv',frame_rows)
        table_write(bundle.path/f'depth_diagnostic_{branch}_pairs.csv',pairs)
        results.append(summary)
    json_write(bundle.path/'depth_unprojection_diagnostic.json',results)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,default=EXPERIMENT);args=parser.parse_args()
    rows=[]
    for method in ('AMB3R_Base_OFF','AMB3R_Base_ON'):
        rows.extend(run(FrontendBundle(args.experiment/method)))
    table_write(args.experiment/'amb3r_depth_channel_comparison.csv',rows)
    text=['# AMB3R saved depth-channel diagnostic','',
        'These supplementary channels retain official world-frame depth unprojection and saved cameras. No new fusion, fit, metric conversion or primary-result replacement is performed. Separate native depth-confidence and same-world-confidence controls distinguish channel support from point-head confidence changes. Algebraic self-projection is not3D accuracy.','',
        '| Method | Confidence field | Foreground coverage | Edge coverage | Adjacent depth agreement | Normalized point distance |','| --- | --- | --- | --- | --- | --- |']
    for r in rows:text.append('| '+ ' | '.join(str(r[k]) for k in ('method','confidence_field','foreground_coverage','edge_coverage','adjacent_depth_agreement_fraction','adjacent_point_distance_p50_normalized'))+' |')
    (args.experiment/'amb3r_depth_channel_comparison.md').write_text('\n'.join(text))
    print('depth-channel diagnostic complete',flush=True)


if __name__=='__main__':main()
