"""Independent saved RGB correspondences, never model-selected image matches.

Brightness changes are candidates: exposure, shadows and feature mistakes remain
possible explanations. A nearest primitive association is not ray visibility.
"""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MPLBACKEND='Agg')
import argparse,csv,json,hashlib,time
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.ndimage import map_coordinates,uniform_filter
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parents[1];PLANT=OUT.parent
HIST=PLANT/'geometry_mesh_final_selection';DUAL=PLANT/'geometry_rgb_dual_frontend'
BASE=OUT.parents[2];DEST=OUT/'comparison/reflection_analysis';DEST.mkdir(parents=True,exist_ok=True)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sample(arr,uv,order=1):
    if arr.ndim==2:return map_coordinates(arr,[uv[:,1],uv[:,0]],order=order,mode='constant',cval=np.nan)
    return np.column_stack([sample(arr[...,i],uv,order) for i in range(arr.shape[2])])
@lru_cache(maxsize=8)
def truth(sid):
    index=json.loads((HIST/'surface_inputs/e10_gggs_surface/render_index.json').read_text())
    item=next(i for i in index if i['source_id']==sid)
    with np.load(item['path']) as z:rgb=z['gt_rgb'];fg=z['foreground_mask']
    lum=uniform_filter(rgb.mean(-1),size=5)
    yy,xx=np.nonzero(fg);leaf=fg.copy();leaf[int(yy.min()+.60*(yy.max()-yy.min())):]=False
    return rgb,fg,lum,leaf,item
def prepare():
    path=DEST/'RGB_QUERY_MANIFEST.csv'
    if path.exists():return list(csv.DictReader(path.open()))
    trans=json.loads((BASE/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset/metadata/preprocess_transforms.json').read_text())['frames']
    trans={int(Path(t['image_name']).stem.split('_')[-1]):t for t in trans}
    records=[]
    for fp in sorted((DUAL/'geometry_comparison/fixed_foreground_matches').glob('*.npz')):
        sid,tid=map(int,fp.stem.split('_'))
        with np.load(fp) as z:
            if not bool(z['accepted']):continue
            a,b=z['source_points'],z['target_points']
        uv=[]
        for points,i in [(a,sid),(b,tid)]:
            t=trans[i];A=np.asarray(t['pixel_transform']);original=np.c_[points,np.ones(len(points))]@np.linalg.inv(A).T
            # Frozen full source2160x3840 -> training1080x1920, integer centres.
            assert (t['original_width'],t['original_height'])==(2160,3840)
            uv.append((original[:,:2]+.5)*.5-.5)
        u,v=uv;s=truth(sid);t=truth(tid)
        lum1=sample(s[2],u);lum2=sample(t[2],v)
        keep=(sample(s[3].astype(float),u,0)>.5)&(sample(t[3].astype(float),v,0)>.5)
        rgb1=sample(s[0],u);rgb2=sample(t[0],v)
        bright=np.maximum(lum1,lum2);delta=np.abs(lum1-lum2)
        # Green/dark leaf texture neighbourhood; silver variegation can pass.
        leafcolor=((rgb1[:,1]>.8*rgb1[:,0])&(rgb1[:,1]>.8*rgb1[:,2]))|((rgb2[:,1]>.8*rgb2[:,0])&(rgb2[:,1]>.8*rgb2[:,2]))
        for k in np.flatnonzero(keep&leafcolor):
            records.append(dict(query_id=len(records),source_id=sid,target_id=tid,x=float(u[k,0]),y=float(u[k,1]),target_x=float(v[k,0]),target_y=float(v[k,1]),source_luminance=float(lum1[k]),target_luminance=float(lum2[k]),luminance_change=float(delta[k]),brightness_candidate=bool(bright[k]>=.55 and delta[k]>=.12),match_archive=str(fp.resolve()),match_archive_sha256=hashlib.sha256(fp.read_bytes()).hexdigest()))
    assert records
    csvwrite(path,records)
    write(DEST/'QUERY_PROTOCOL.json',dict(status='prepared',queries=len(records),changing_brightness_queries=sum(r['brightness_candidate'] for r in records),matches='Existing image-only RAP SIFT mutual Lowe.75 / RANSAC1.5shared-grid px; no new matches, no model depth selection',pixel_transform='Saved perimage Omega actual half-pixel affine inverse -> original -> (pixel+.5)*.5-.5 for training1080x1920',leaf_region='Both endpoints original RAP upper60percent bbox and either endpoint green-relative RGB; cannot guarantee every sample is a leaf',brightness='GT5pixel mean RGB luminance; maximum >=.55, absolute change>=.12; candidate for reflection/shadow/exposure/registration, not certified specularity',denominator='Same fixed RGB_QUERY_MANIFEST for every model. Alpha/depth validity separately retained; missing samples never improve median secretly',ground_truth='No scan or calibrated physical3D truth'))
    for r in sorted((r for r in records if r['brightness_candidate']),key=lambda r:r['luminance_change'],reverse=True)[:16]:
        ims=[]
        for sid,x,y in [(r['source_id'],r['x'],r['y']),(r['target_id'],r['target_x'],r['target_y'])]:
            rgb=truth(sid)[0];im=Image.fromarray((rgb.clip(0,1)*255).astype('uint8'));x,y=int(x),int(y);crop=im.crop((x-64,y-64,x+64,y+64));d=ImageDraw.Draw(crop);d.ellipse((60,60,68,68),outline='red',width=2);ims.append(crop)
        panel=Image.new('RGB',(256,148),'black');panel.paste(ims[0],(0,20));panel.paste(ims[1],(128,20));ImageDraw.Draw(panel).text((2,3),f"RGB match {r['query_id']} {r['source_id']} -> {r['target_id']}",fill='white');panel.save(DEST/f"RGB_candidate_{r['query_id']:06d}.png")
    return records
def evaluate(args,records):
    index={int(r['source_id']):r for r in json.loads((args.input/'render_index.json').read_text())};assert len(index)==208
    @lru_cache(maxsize=6)
    def frame(sid):
        item=index[sid]
        with np.load(item['path']) as z:
            depth=z['depth_surface'];normal=z['native_normal'];alpha=z['alpha']
            if args.aux:
                depth=z['depth_expected_center_aux'];normal=z['normal_smallest_axis_aux']
        E=np.asarray(item['camera']['world_view_transform']).T
        if args.normal=='camera':normal=normal@E[:3,:3]
        normal=normal/np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-10)
        c=item['camera'];K=np.array([[c['Fx'],0,c['Cx']],[0,c['Fy'],c['Cy']],[0,0,1]])
        return depth,normal,alpha,E,K
    tree=None;primitive=None;tree_native_ids=None;nonfinite_primitive_centers=0
    if args.ply:
        from plyfile import PlyData
        ply=PlyData.read(args.ply)['vertex'].data;xyz=np.column_stack([ply[k] for k in ['x','y','z']])
        center_finite=np.isfinite(xyz).all(1);tree_native_ids=np.flatnonzero(center_finite)
        nonfinite_primitive_centers=int((~center_finite).sum())
        assert len(tree_native_ids),'No finite native primitive centers for spatial analysis'
        tree=cKDTree(xyz[center_finite])
        scales=np.column_stack([ply[k] for k in ply.dtype.names if k.startswith('scale_')]);primitive={'scale_fields':scales.shape[1]}
        if scales.shape[1]==6:
            s=np.exp(scales[:,:3])*np.tanh(scales[:,3:]);primitive.update(tangent=np.maximum(np.abs(s[:,0]),np.abs(s[:,1]))/np.maximum(np.minimum(np.abs(s[:,0]),np.abs(s[:,1])),1e-12),a=s[:,2]*np.sign(s[:,0])/np.maximum(s[:,0]**2,1e-20),b=s[:,2]*np.sign(s[:,1])/np.maximum(s[:,1]**2,1e-20))
        else:
            s=np.exp(scales);primitive.update(tangent=s.max(1)/np.maximum(s.min(1),1e-12))
    rows=[];diagonal=json.loads((HIST/'configs/standard_spatial_scale.json').read_text())['diagonal'];start=time.time()
    # Pair batches permit cached sequential frames, low memory and exact fixed queries.
    pairs={}
    for r in records:pairs.setdefault((int(r['source_id']),int(r['target_id'])),[]).append(r)
    for (sid,tid),rr in pairs.items():
        u=np.asarray([[float(r['x']),float(r['y'])] for r in rr]);v=np.asarray([[float(r['target_x']),float(r['target_y'])] for r in rr]);d,n,al,E,K=frame(sid);dt,nt,at,Et,Kt=frame(tid)
        z=sample(d,u);a=sample(al,u);za=sample(dt,v);aa=sample(at,v)
        rays=np.c_[u,np.ones(len(u))]@np.linalg.inv(K).T;C=np.linalg.inv(E);world=(rays*z[:,None])@C[:3,:3].T+C[:3,3];local=world@Et[:3,:3].T+Et[:3,3];h=local@Kt.T;proj=h[:,:2]/h[:,2,None]
        reproj=np.linalg.norm(proj-v,axis=1);rel=np.abs(local[:,2]-za)/np.maximum(za,1e-8)
        valid=np.isfinite(reproj)&np.isfinite(z)&(z>0)&np.isfinite(za)&(za>0)&(a>=.5)&(aa>=.5)&(local[:,2]>0)
        ns=sample(n,u);nts=sample(nt,v);angle=np.degrees(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i',ns,nts)),-1,1)))
        localj=[]
        for shift in [(3,0),(-3,0),(0,3),(0,-3)]:
            adj=sample(n,u+shift);localj.append(np.degrees(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i',ns,adj)),-1,1))))
        jump=np.nanmean(localj,axis=0)
        distances=np.full(len(rr),np.nan);ids=np.zeros(len(rr),dtype=int)
        if tree:
            finite=np.isfinite(world).all(1);distances[finite],local_ids=tree.query(world[finite],workers=8)
            ids[finite]=tree_native_ids[local_ids]
        for j,r in enumerate(rr):
            row=dict(r,method=args.method,valid=bool(valid[j]),alpha_source=float(a[j]),alpha_target=float(aa[j]),source_depth=float(z[j]),target_depth=float(za[j]),source_world_x=float(world[j,0]),source_world_y=float(world[j,1]),source_world_z=float(world[j,2]),RGB_match_reprojection_px=float(reproj[j]) if valid[j] else '',matched_relative_depth_error=float(rel[j]) if valid[j] else '',normal_crossview_abs_angle_deg=float(angle[j]) if valid[j] else '',local_normal_jump3px_deg=float(jump[j]) if valid[j] else '',depth_type='auxiliary3D centerZ moment' if args.aux else 'native surface depth')
            if tree:
                near=valid[j] and distances[j]<diagonal/512*5;row.update(nearest_primitive_distance=float(distances[j]),nearest_primitive_id=int(ids[j]),nearest_primitive_association_within5vox=bool(near),nearest_primitive_anisotropy=float(primitive['tangent'][ids[j]]) if near else '',primitive_anisotropy_type='QGS signed tangential' if primitive['scale_fields']==6 else '2D tangential' if primitive['scale_fields']==2 else '3D max/min sigma')
                if primitive['scale_fields']==6:row.update(nearest_native_quadratic_a=float(primitive['a'][ids[j]]) if near else '',nearest_native_quadratic_b=float(primitive['b'][ids[j]]) if near else '')
            rows.append(row)
        print('reflection',args.method,sid,tid,len(rows),flush=True)
    csvwrite(DEST/f'{args.method}_spatial_correspondence.csv',rows)
    summary=[]
    for candidate in [True,False]:
        group=[r for r in rows if str(r['brightness_candidate']).lower()==str(candidate).lower()];ok=[r for r in group if r['valid']]
        metrics={}
        for key in ['RGB_match_reprojection_px','matched_relative_depth_error','normal_crossview_abs_angle_deg','local_normal_jump3px_deg','nearest_primitive_anisotropy']:
            vals=[float(r[key]) for r in ok if r.get(key,'')!='' and np.isfinite(float(r[key]))]
            metrics[key+'_median']=float(np.median(vals)) if vals else None;metrics[key+'_p90']=float(np.quantile(vals,.9)) if vals else None
        summary.append(dict(method=args.method,brightness_candidate=candidate,queries=len(group),valid_queries=len(ok),valid_fraction=len(ok)/max(len(group),1),**metrics))
    write(DEST/f'{args.method}_summary.json',dict(status='complete',rows=summary,elapsed_seconds=time.time()-start,input=str(args.input.resolve()),primitive_source=str(args.ply.resolve()) if args.ply else None,normal_coordinates=args.normal,auxiliary=args.aux,nonfinite_native_centers_excluded_from_tree=nonfinite_primitive_centers,original_native_model_unchanged=True,primitive_association='nearest finite model center within5vox of source rendered surface point; original native primitive indices retained; NOT renderer visibility/contribution evidence',no_scan_accuracy_claim=True))
parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--method');parser.add_argument('--input',type=Path);parser.add_argument('--normal',choices=['camera','world']);parser.add_argument('--ply',type=Path);parser.add_argument('--aux',action='store_true');args=parser.parse_args()
records=prepare()
if not args.prepare_only:
    assert args.input and args.method and args.normal
    evaluate(args,records)
