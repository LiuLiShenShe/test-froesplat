"""Read-only channel check on two saved initialization views, not full PLY scan."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OMEGA=ROOT.parents[2]
STD=OMEGA.parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison'
PRED=STD/'Standard_VGGT_Frontend/predictions'
PLY=STD/'Standard_VGGT_Dataset/sparse/0/points3D.ply'

def main():
    mapping={'float':'<f4','double':'<f8','uchar':'u1','uint8':'u1','int':'<i4','uint':'<u4'}
    props=[];n=None;in_vertex=False
    with PLY.open('rb') as f:
        header=[]
        while True:
            line=f.readline()
            if not line:raise ValueError('Incomplete PLY')
            header.append(line.decode().strip());parts=header[-1].split()
            if parts[:2]==['element','vertex']:
                n=int(parts[2]);in_vertex=True
            elif parts[:1]==['element']:in_vertex=False
            elif in_vertex and parts[:1]==['property']:
                if len(parts)!=3 or parts[1] not in mapping:raise ValueError(parts)
                props.append((parts[2],mapping[parts[1]]))
            if header[-1]=='end_header':break
        offset=f.tell()
    if 'format binary_little_endian 1.0' not in header:raise ValueError('Unsupported PLY format')
    points=np.memmap(PLY,dtype=np.dtype(props),mode='r',offset=offset,shape=(n,))
    with np.load(PRED/'sampled_provenance.npz',allow_pickle=False) as z:
        frames=z['frame'];x=z['px_model'];y=z['py_model']
    results=[]
    for frame,source in [(0,0),(207,209)]:
        indices=np.flatnonzero(frames==frame)
        take=indices[np.rint(np.linspace(0,len(indices)-1,min(256,len(indices)))).astype(int)]
        rgb=np.load(ROOT/f'controlled_fixes/view_count_control_v1/Standard_70/per_frame/model_rgb_float32_crop_{source:04d}.npy',allow_pickle=False)
        expected=np.rint(np.clip(rgb[:,y[take].astype(int),x[take].astype(int)].T,0,1)*255).astype(np.int16)
        observed=np.column_stack([points[channel][take] for channel in ['red','green','blue']]).astype(np.int16)
        delta=np.abs(observed-expected)
        results.append({'source_id':source,'points_checked':len(take),'RGB_MAE_255':float(delta.mean()),
                        'RGB_max_abs_255':int(delta.max()),'RGB_equal_fraction':float(np.mean(np.all(observed==expected,axis=1))),
                        'BGR_alternative_MAE_255':float(np.abs(observed-expected[:,::-1]).mean())})
        if delta.max()>1:raise AssertionError('RGB initializer diverges from actual input')
    out={'source_PLY':str(PLY),'points_in_full_PLY':n,'actual_read_points':sum(r['points_checked'] for r in results),
         'source_color_code':'run_standard_vggt_frontend.py:215-235,296; F.interpolate RGB then red/green/blue fields',
         'results':results,'channel_swap_detected':False,
         'limitations':'Two source frames, 256 actual initialized points each; same preprocessing/current float input reused from count control. Not exhaustive color validation or past-environment equivalence.'}
    (ROOT/'camera_projection_comparison/initial_RGB_channel_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results))

if __name__=='__main__':main()
