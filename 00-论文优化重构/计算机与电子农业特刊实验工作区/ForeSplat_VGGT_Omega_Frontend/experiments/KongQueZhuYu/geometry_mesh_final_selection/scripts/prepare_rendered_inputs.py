"""Depth-visible surface observations use the same normal/sampling protocol."""
import argparse,json,time
import numpy as np
from surface_protocol import ROOT,PROTOCOL,scale_for,normals_from_grid,unproject,unique_observations,merge_sampled,write,save_npz_fast

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['E8','E10','E11'],required=True);a=p.parse_args();exp=a.experiment
    out=ROOT/'surface_inputs'/f'{exp.lower()}_{"2dgs" if exp=="E11" else "gggs"}_surface'
    if (out/'complete.json').exists():return
    info=json.loads((out/'render_complete.json').read_text());scale=scale_for('omega' if exp=='E8' else 'standard');index=json.loads((out/'render_index.json').read_text());parts=[];records=[];start=time.time()
    for row in index:
        with np.load(row['path']) as d:
            depth=d['depth_surface'];alpha=d['alpha'];mask=d['foreground_mask'];rgb=(np.clip(d['render_raw'],0,1)*255).round().astype(np.uint8)
            xyz,sensor=unproject(depth,row['camera']);valid=mask&(alpha>=PROTOCOL['surface_alpha_min'])&np.isfinite(depth)&(depth>0)&np.isfinite(xyz).all(-1)
            normal,good=normals_from_grid(xyz,depth,valid,sensor);y,x=np.nonzero(valid);n=len(x)
            native=d['native_normal']
            if info['normal_coordinates']=='camera':native=native@np.asarray(row['camera']['world_view_transform']).T[:3,:3]
            full=dict(xyz=xyz[valid],rgb=rgb[valid],normal=normal[valid],normal_valid=good[valid],alpha=alpha[valid],depth=depth[valid],native_normal_world=native[valid],
                pixel_xy=np.column_stack([x,y]).astype(np.int32),source_id=np.full(n,row['source_id'],dtype=np.int16),sensor=np.broadcast_to(sensor,(n,3)).copy(),foreground_mask=np.ones(n,dtype=bool))
        dest=out/'per_frame'/f'{row["camera"]["name"]}.npz';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():save_npz_fast(dest,**full)
        filtered={k:v[full['normal_valid']] for k,v in full.items() if k not in ('normal_valid','foreground_mask','native_normal_world')}
        parts.append(unique_observations(filtered,scale));records.append(dict(source_id=row['source_id'],visible_mask_alpha_depth_points=n,valid_normal_points=int(good.sum()),sampled_frame_points=len(parts[-1]['xyz'])))
        print(exp,row['source_id'],'visible surface',n,'valid normal',int(good.sum()),flush=True)
    count=merge_sampled(parts,scale,out/'nksr_input.npz')
    write(out/'complete.json',dict(status='complete',experiment=exp,nksr_points=count,raw_surface_points=sum(r['visible_mask_alpha_depth_points'] for r in records),
        valid_normal_total=sum(r['valid_normal_points'] for r in records),frames=records,scale=scale,protocol=PROTOCOL,elapsed_seconds=time.time()-start,normal_strategy='Same depth finite differences as direct inputs; native rendered normals separately retained for audit'))

if __name__=='__main__':main()
