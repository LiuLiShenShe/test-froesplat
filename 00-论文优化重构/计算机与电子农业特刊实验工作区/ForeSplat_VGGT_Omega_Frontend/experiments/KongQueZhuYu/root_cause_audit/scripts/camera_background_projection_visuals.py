"""Actual source-depth to independent target RGB matches, fixed plot settings."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OMEGA=ROOT.parents[2]
DATA=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
CONTROL=ROOT/'controlled_fixes/background_control_v1'

def project(source,target,p1,a_source,a_target,depth):
    xyz=np.c_[p1,np.ones(len(p1))]@np.linalg.inv(a_source).T
    pixel=(xyz@np.asarray(source['pixel_transform_pixel_center']).T)[:,:2]
    z=cv2.remap(depth,pixel[:,0].astype(np.float32)[None],pixel[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
    rays=np.c_[pixel,np.ones(len(pixel))]@np.linalg.inv(source['intrinsic_model']).T
    c2w=np.asarray(source['c2w']);w2c=np.asarray(target['extrinsic_model_w2c'])
    world=(rays*z[:,None])@c2w[:3,:3].T+c2w[:3,3]
    dest=world@w2c[:3,:3].T+w2c[:3,3]
    k=a_target@np.linalg.inv(target['pixel_transform_pixel_center'])@np.asarray(target['intrinsic_model'])
    uvw=dest@k.T
    uv=uvw[:,:2]/uvw[:,2,None]
    return uv,(np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)&(dest[:,2]>0))

def main():
    raw=json.loads((DATA/'metadata/preprocess_transforms.json').read_text())['frames']
    ref={int(Path(r['image_name']).stem.split('_')[-1]):np.asarray(r['pixel_transform']) for r in raw}
    results=[]
    for sid,tid in [(0,1),(95,96)]:
        with np.load(CONTROL/f'evaluation/fixed_foreground_matches/{sid:04d}_{tid:04d}.npz',allow_pickle=False) as m:
            assert bool(m['accepted'])
            p1=m['source_points'];p2=m['target_points']
        rgb=np.load(DATA/f'per_frame/model_rgb_crop_{tid:04d}.npy',allow_pickle=False)
        if rgb.shape[0]==3:rgb=rgb.transpose(1,2,0)
        if rgb.dtype!=np.uint8:rgb=np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
        h,w=rgb.shape[:2]
        indices=np.rint(np.linspace(0,len(p1)-1,min(24,len(p1)))).astype(int)
        fig,axes=plt.subplots(2,2,figsize=(9,13))
        for row,model in enumerate(['Omega','Standard']):
            for col,condition in enumerate(['RAP_baseline','background_restored']):
                path=CONTROL/f'{model}_{condition}_24'
                cams=json.loads((path/'cameras.json').read_text())['frames']
                lookup={int(r['source_index']):r for r in cams}
                depth=np.load(path/f'per_frame/depth_crop_{sid:04d}.npy',mmap_mode='r',allow_pickle=False)
                uv,valid=project(lookup[sid],lookup[tid],p1,ref[sid],ref[tid],depth)
                error=np.linalg.norm(uv-p2,axis=1)
                ax=axes[row,col];ax.imshow(rgb)
                for j in indices:
                    if not valid[j]:continue
                    ax.plot([p2[j,0],uv[j,0]],[p2[j,1],uv[j,1]],color='white',alpha=.65,linewidth=.7)
                ax.scatter(p2[indices,0],p2[indices,1],s=28,facecolors='none',edgecolors='#23ed57',linewidths=.9)
                selected=indices[valid[indices]]
                ax.scatter(uv[selected,0],uv[selected,1],s=22,color='#fa2aff',marker='x',linewidths=.9)
                ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.axis('off')
                median=float(np.median(error[valid]))
                ax.set_title(f'{model} {condition}\nAll {int(valid.sum())}/{len(p1)} positive-z matches; median {median:.2f} px',fontsize=10)
                results.append(dict(source=sid,target=tid,model=model,condition=condition,
                                    matched_points=len(p1),positive_z_points=int(valid.sum()),median_projection_error_shared_px=median))
        fig.suptitle(f'Source {sid:04d} saved depth projected into target {tid:04d} RGB\nGreen: same observed RGB matches; magenta: predicted projections\nSame 24 displayed indices; identical target RAP image, axes and markers',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.955))
        fig.savefig(CONTROL/f'evaluation/source_{sid:04d}_to_{tid:04d}_projection_overlay.png',dpi=150)
        plt.close(fig)
    (CONTROL/'evaluation/projection_overlay_protocol.json').write_text(json.dumps(
        dict(results=results,shared_grid_wh=[w,h],point_indices='24 uniform ranks in fixed foreground RGB-F inliers, independent of error/condition',
             raw_data_modified=False,new_inference=False,limits='Observed RGB correspondence is not 3D GT. Display clips out-of-image predictions; statistics keep all positive-z points.'),indent=2)+'\n')
    print(json.dumps(results))

if __name__=='__main__':main()
