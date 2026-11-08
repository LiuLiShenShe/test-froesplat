"""Read-only final Gaussian means in the first-camera frame; no pruning or alignment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
EXPS=['E1','E5','E7','E8','E9','E10']

def main():
    data={};protocol=[];max_bound=1.
    for e in EXPS:
        c=np.load(ROOT/'gaussian_analysis'/f'{e}_final_scalar_cache.npy',mmap_mode='r')
        records=json.loads((ROOT/'render_comparison'/e/'camera_runtime.json').read_text())['records']
        camera=next(r for r in records if r['source_id']==0)
        view=np.asarray(camera['world_view_transform']).T
        centers=np.array([r['camera_center'] for r in records]);extent=np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1
        xyz=np.asarray(c[:,:3]);local=(xyz@view[:3,:3].T+view[:3,3])/extent
        center=np.median(local,axis=0);local=local-center
        finite=np.isfinite(local).all(axis=1)
        valid=np.flatnonzero(finite);index=valid[np.linspace(0,len(valid)-1,min(100000,len(valid))).astype(int)]
        bound=float(np.abs(local[finite]).max());max_bound=max(max_bound,bound)
        data[e]=(local[index],np.asarray(c[index,3:6]))
        protocol.append(dict(experiment=e,total=len(c),shown=len(index),nonfinite_means=int((~finite).sum()),
            native_camera_extent=float(extent),center_in_first_camera_extent_units=center.tolist(),all_means_max_abs_coordinate=bound,
            all_means_outside_central_box1_fraction=float((np.abs(local[finite]).max(axis=1)>1).mean())))
    out=ROOT/'geometry_comparison'
    for mode,bound in [('all_bounds',max_bound*1.01),('central_box1',1.)]:
        fig=plt.figure(figsize=(15,10))
        for i,e in enumerate(EXPS):
            ax=fig.add_subplot(2,3,i+1,projection='3d');xyz,color=data[e]
            ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=color,s=.15,linewidths=0,depthshade=False)
            ax.view_init(elev=15,azim=-60);ax.set_xlim(-bound,bound);ax.set_ylim(bound,-bound);ax.set_zlim(-bound,bound)
            ax.set_box_aspect((1,1,1));ax.set_title(e+' final Gaussian means');ax.set_xlabel('X / extent');ax.set_ylabel('Y / extent');ax.set_zlabel('Z / extent')
        fig.suptitle('Unfiltered final Gaussian means; first-camera frame, median-centered display only; SH DC color proxy')
        fig.tight_layout();fig.savefig(out/f'gaussian_means_{mode}.png',dpi=180);plt.close(fig)
    (out/'gaussian_means_protocol.json').write_text(json.dumps(dict(records=protocol,
        common_all_bounds=max_bound*1.01,common_central_bounds=1.,same_display_rotation=[15,-60],
        selection='Deterministic evenly spaced finite rows, at most100000 per model; no alpha/color/scale selection or model pruning',
        color='Clamped SH DC-only proxy; these are Gaussian means, not opaque surfaces or full SH3 rendered colors',
        scope='All-bounds view includes every finite mean within plot limits. Central crop is explicitly paired with all-bounds view and outside fraction. Display normalization does not change inputs/checkpoints or fit scan scale.'),indent=2)+'\n')
    print('six final Gaussian mean clouds saved without alpha/scale pruning')

if __name__=='__main__':main()
