"""CPU, no inference: shared source-pixel queries and predicted-occlusion audit."""
import io,hashlib
from PIL import Image
from scipy.ndimage import binary_erosion
from geometry_common import *

OFFSETS=[1,5,15,30,60]
ROIS={'leaf_window':[540,570,1080,1230],'leaf_edge_window':[510,630,720,1110],
      'pot_rim_window':[800,1680,1700,2000]}

def world_from_depth(depth,c2w,k):
    y,x=np.indices(depth.shape,dtype=np.float64)
    local=np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones_like(x)),axis=-1)*depth[...,None]
    return (local@c2w[:3,:3].T+c2w[:3,3]).astype(np.float32)

def bilinear_1024_to_518(rgb):
    # Replay official F.interpolate bilinear/align_corners=False on CPU NumPy.
    u=(np.arange(518,dtype=np.float32)+.5)*(1024/518)-.5;i=np.floor(u).astype(int);w=(u-i).astype(np.float32)
    a=rgb.astype(np.float32)/255.;lo=a[i[:,None],i[None,:]];hi_x=a[i[:,None],(i+1)[None,:]]
    lo_y=a[(i+1)[:,None],i[None,:]];hi=a[(i+1)[:,None],(i+1)[None,:]]
    out=lo*(1-w[:,None,None])*(1-w[None,:,None])+hi_x*(1-w[:,None,None])*w[None,:,None]+lo_y*w[:,None,None]*(1-w[None,:,None])+hi*w[:,None,None]*w[None,:,None]
    return np.rint(np.clip(out,0,1)*255).astype(np.uint8)

def standard_input(name,record):
    image_path=STANDARD/'Standard_VGGT_Dataset/images'/name;mask_path=STANDARD/'Standard_VGGT_Dataset/masks'/name
    raw=image_path.read_bytes();mask_raw=mask_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['rgb_sha256'],name+' RGB source changed'
    assert hashlib.sha256(mask_raw).hexdigest()==record['mask_sha256'],name+' mask source changed'
    image=Image.open(io.BytesIO(raw))
    if image.mode=='RGBA':image=Image.alpha_composite(Image.new('RGBA',image.size,(0,0,0,255)),image)
    image=image.convert('RGB');w,h=image.size;dimension=max(w,h);left=(dimension-w)//2;top=(dimension-h)//2
    square=Image.new('RGB',(dimension,dimension),(0,0,0));square.paste(image,(left,top))
    rgb=bilinear_1024_to_518(np.asarray(square.resize((1024,1024),Image.Resampling.BICUBIC)))
    mask=Image.open(io.BytesIO(mask_raw)).convert('L');square=Image.new('L',(dimension,dimension),0);square.paste(mask,(left,top))
    model_mask=np.asarray(square.resize((518,518),Image.Resampling.NEAREST))>=127
    return rgb,model_mask

def ply_raw(directory,source):
    path=Path(directory)/'pointcloud/global_raw.ply'
    with path.open('rb') as f:
        header=[]
        while True:
            line=f.readline();header.append(line)
            if line==b'end_header\n':break
        n=int(next(line for line in header if line.startswith(b'element vertex ')).split()[-1]);offset=f.tell()
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    t=source['transforms'];counts=[(r['input_size_wh'][0]-r['padding_ltrb'][0]-r['padding_ltrb'][2])*(r['input_size_wh'][1]-r['padding_ltrb'][1]-r['padding_ltrb'][3]) for r in t]
    assert n==sum(counts)==source['bundle']['cloud_counts']['raw'];assert path.stat().st_size==offset+n*dtype.itemsize
    return np.memmap(path,mode='r',offset=offset,dtype=dtype,shape=(n,)),counts

def cache_frames(name,source):
    frames=[];packed=None;offset=0
    if name=='Standard_depth':
        manifest={r['image_name']:r for r in json.loads((STANDARD/'Standard_VGGT_Frontend/metadata/input_manifest.json').read_text())['frames']}
    elif name!='Omega_depth':packed,counts=ply_raw(source['path'],source)
    for i,n in enumerate(source['names']):
        stem=Path(n).stem;rgb=None
        if name=='Omega_depth':
            depth=np.load(OMEGA_DATA/'per_frame'/f'depth_{stem}.npy').squeeze();conf=np.load(OMEGA_DATA/'per_frame'/f'confidence_{stem}.npy').squeeze()
            mask=np.load(OMEGA_DATA/'per_frame'/f'model_mask_{stem}.npy').squeeze()>0;rgb=np.load(OMEGA_DATA/'per_frame'/f'model_rgb_{stem}.npy')
            xyz=world_from_depth(depth,source['c2w'][i],source['k_model'][i])
        elif name=='Standard_depth':
            depth=np.load(STANDARD/'Standard_VGGT_Frontend/depth'/f'{stem}.npy').squeeze();conf=np.load(STANDARD/'Standard_VGGT_Frontend/confidence'/f'{stem}.npy').squeeze()
            rgb,mask=standard_input(n,manifest[n]);xyz=world_from_depth(depth,source['c2w'][i],source['k_model'][i])
        else:
            p=Path(source['path']);w,h=source['size_wh'][i];left,top,right,bottom=source['transforms'][i]['padding_ltrb']
            part=packed[offset:offset+counts[i]];offset+=counts[i];xyz=np.full((h,w,3),np.nan,np.float32)
            xyz[top:h-bottom,left:w-right]=np.column_stack([part[k] for k in ['x','y','z']]).reshape(h-top-bottom,w-left-right,3)
            with np.load(io.BytesIO((p/'confidence'/f'{stem}.npz').read_bytes())) as f:conf=f['world_points_conf'].reshape(h,w)
            mask=np.asarray(Image.open(p/'mask'/f'{stem}.png'))>0
            depth=xyz@source['w2c'][i][2,:3]+source['w2c'][i][2,3]
            if i==0:
                # Exact raw-source probe prevents silently treating a fused or
                # downsampled PLY as complete row-major pointmaps.
                with np.load(io.BytesIO((p/'raw_predictions'/f'{stem}.npz').read_bytes())) as f:original=f['world_points']
                for y,x in [(top,left),((h+top-bottom)//2,(w+left-right)//2),(h-bottom-1,w-right-1)]:assert np.array_equal(original[y,x],xyz[y,x])
        assert depth.shape==conf.shape==mask.shape
        frames.append(dict(name=n,source_id=source_id(n),depth=depth,xyz=xyz,conf=conf,mask=mask,rgb=rgb,
            w2c=source['w2c'][i],k=source['k_model'][i],a=source['a_actual'][i],index=i))
        if (i+1)%32==0:print(name,'cached',i+1,'/',len(source['names']),flush=True)
    if packed is not None:del packed
    return frames

def references(omega_frames):
    result={};rng=np.random.default_rng(107)
    for frame in omega_frames:
        y,x=np.indices(frame['mask'].shape);orig=np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(frame['a']).T
        edge=frame['mask']&~binary_erosion(frame['mask']);regions={'foreground':frame['mask'],'foreground_boundary_common_grid':edge}
        for name,(left,top,right,bottom) in ROIS.items():regions[name]=frame['mask']&(orig[...,0]>=left)&(orig[...,0]<right)&(orig[...,1]>=top)&(orig[...,1]<bottom)
        out={}
        for region,selected in regions.items():
            points=orig[selected,:2]
            if len(points)>1500:
                seed=frame['source_id']*101+list(regions).index(region)
                points=points[np.random.default_rng(seed).choice(len(points),1500,replace=False)]
            out[region]=points
        result[frame['name']]=out
    return result

def prepare_queries(frame,original):
    uv=np.column_stack((original,np.ones(len(original))))@frame['a'].T
    uv=np.rint(uv[:,:2]).astype(int);h,w=frame['mask'].shape
    inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    clipped=np.column_stack((np.clip(uv[:,0],0,w-1),np.clip(uv[:,1],0,h-1)))
    x,y=clipped.T;xyz=frame['xyz'][y,x];conf=frame['conf'][y,x];finite=inside&frame['mask'][y,x]&np.isfinite(xyz).all(-1)&np.isfinite(conf)&(frame['depth'][y,x]>0)
    return dict(original=original,uv=clipped,xyz=xyz,conf=conf,finite=finite)

def pair_metrics(query,source,target,control):
    n=len(query['xyz']);valid=query['finite']&(query['conf']>1 if control=='confidence_gt1' else True)
    point=query['xyz'];camera=point@target['w2c'][:3,:3].T+target['w2c'][:3,3];z=camera[:,2]
    uv=camera[:,:2]/np.maximum(z[:,None],1e-15);uv=uv@target['k'][:2,:2].T+target['k'][:2,2]
    finite=np.isfinite(uv).all(-1)&np.isfinite(z)&(z>0)&valid
    pix=np.rint(np.nan_to_num(uv,nan=-1e9,posinf=-1e9,neginf=-1e9)).astype(np.int64);h,w=target['mask'].shape
    inside=finite&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h);ids=np.flatnonzero(inside)
    base=dict(queries=n,source_valid=int(valid.sum()),projected_inside=int(inside.sum()))
    if not len(ids):return dict(base,target_overlap=0,source_zbuffer_occluded=0,target_predicted_occluded=0,supported_2pct=0,supported_5pct=0,supported_10pct=0,visible_candidates=0)
    px,py=pix[ids].T;tz=target['depth'][py,px];txyz=target['xyz'][py,px];tc=target['conf'][py,px]
    good=target['mask'][py,px]&np.isfinite(tz)&(tz>0)&np.isfinite(txyz).all(-1)&np.isfinite(tc)
    if control=='confidence_gt1':good&=tc>1
    # Source projected z-buffer, retaining equal-depth duplicate query pixels.
    linear=py*w+px;unique,inverse=np.unique(linear,return_inverse=True);minimum=np.full(len(unique),np.inf)
    np.minimum.at(minimum,inverse,z[ids]);self_visible=z[ids]<=minimum[inverse]*1.005
    relative=(z[ids]-tz)/np.maximum(tz,1e-15);behind=good&self_visible&(relative>.05)
    visible=good&self_visible&~behind;overlap=good&self_visible
    distance=np.linalg.norm(point[ids]-txyz,axis=1)/np.maximum(tz,1e-15)
    target_back=txyz@source['w2c'][:3,:3].T+source['w2c'][:3,3]
    back=target_back[:,:2]/np.maximum(target_back[:,2,None],1e-15);back=back@source['k'][:2,:2].T+source['k'][:2,2]
    inverse_a=np.linalg.inv(source['a'])[:2,:2]
    cycle=(back-query['uv'][ids])@inverse_a.T;cycle_error=np.linalg.norm(cycle,axis=1)
    return dict(base,target_overlap=int(good.sum()),source_zbuffer_occluded=int((good&~self_visible).sum()),
        target_predicted_occluded=int(behind.sum()),visible_candidates=int(visible.sum()),
        supported_2pct=int((overlap&(np.abs(relative)<=.02)).sum()),supported_5pct=int((overlap&(np.abs(relative)<=.05)).sum()),supported_10pct=int((overlap&(np.abs(relative)<=.10)).sum()),
        raw_overlap_relative_depth_error_p50=q(np.abs(relative[good]),.5),raw_overlap_relative_depth_error_p90=q(np.abs(relative[good]),.9),
        visible_relative_depth_error_p50=q(np.abs(relative[visible]),.5),visible_relative_depth_error_p90=q(np.abs(relative[visible]),.9),
        visible_signed_depth_error_p50=q(relative[visible],.5),visible_cycle_error_original_px_p50=q(cycle_error[visible],.5),visible_cycle_error_original_px_p90=q(cycle_error[visible],.9),
        visible_world_distance_over_target_depth_p50=q(distance[visible],.5))

def summarize(rows):
    results=[]
    keys=sorted({(r['method'],r['control'],r['region'],abs(r['source_offset'])) for r in rows})
    for method,control,region,offset in keys:
        pairs=[r for r in rows if (r['method'],r['control'],r['region'],abs(r['source_offset']))==(method,control,region,offset)]
        denominator=sum(r['queries'] for r in pairs);raw_overlap=sum(r['target_overlap'] for r in pairs)
        row=dict(method=method,control=control,region=region,source_offset=offset,bidirectional_pairs=len(pairs),queries=denominator,
            pairs_with_reference_queries=sum(r['queries']>0 for r in pairs),no_reference_query_pairs=sum(r['queries']==0 for r in pairs),
            source_valid_fraction=sum(r['source_valid'] for r in pairs)/max(denominator,1),projection_inside_fraction=sum(r['projected_inside'] for r in pairs)/max(denominator,1),
            target_foreground_overlap_fraction=raw_overlap/max(denominator,1),source_zbuffer_occluded_fraction=sum(r['source_zbuffer_occluded'] for r in pairs)/max(denominator,1),
            target_predicted_occluded_fraction=sum(r['target_predicted_occluded'] for r in pairs)/max(denominator,1),visible_fraction=sum(r['visible_candidates'] for r in pairs)/max(denominator,1))
        for percent in [2,5,10]:
            row[f'support_{percent}pct_fraction']=sum(r[f'supported_{percent}pct'] for r in pairs)/max(denominator,1)
            row[f'per_pair_support_{percent}pct_p10']=q([r[f'supported_{percent}pct']/r['queries'] for r in pairs if r['queries']>0],.1)
            row[f'per_pair_support_{percent}pct_p50']=q([r[f'supported_{percent}pct']/r['queries'] for r in pairs if r['queries']>0],.5)
        for field in ['raw_overlap_relative_depth_error_p50','raw_overlap_relative_depth_error_p90','visible_relative_depth_error_p50','visible_relative_depth_error_p90','visible_cycle_error_original_px_p50','visible_cycle_error_original_px_p90','visible_world_distance_over_target_depth_p50']:
            row[field+'_across_pair_median']=q([r.get(field) for r in pairs if r.get(field) is not None],.5)
        results.append(row)
    return results

def check_saved_standard_points(frames):
    path=STANDARD/'Standard_VGGT_Frontend/predictions/sampled_provenance.npz'
    with np.load(io.BytesIO(path.read_bytes())) as p:data={k:p[k] for k in ['frame','px_model','py_model','depth','xyz']}
    errors=[];depth_mismatch=0
    for i,frame in enumerate(frames):
        indices=np.flatnonzero(data['frame']==i);x=data['px_model'][indices].astype(int);y=data['py_model'][indices].astype(int)
        depth_mismatch+=int(np.sum(frame['depth'][y,x]!=data['depth'][indices]))
        errors.append(np.linalg.norm(frame['xyz'][y,x]-data['xyz'][indices],axis=1))
    error=np.concatenate(errors)
    write_json(AUDIT/'geometry_comparison/standard_saved_point_export_check.json',dict(saved_points=len(error),saved_depth_value_mismatches=depth_mismatch,
        reconstructed_xyz_error_p50=q(error,.5),reconstructed_xyz_error_p90=q(error,.9),reconstructed_xyz_error_max=q(error,1),
        full_point_head_saved=False,full_point_head_run=False,point_head_status='Unresolved: producer never ran/saved point_head; saved sampled points are depth unprojection',
        scope='Actual full1,252,383 saved sampled XYZ and depths checked against full208depth and recovered exported cameras; no model inference'))

def main():
    sources=camera_sources();omega=cache_frames('Omega_depth',sources['Omega_depth']);refs=references(omega);all_rows=[];input_rows=[]
    for method,source in sources.items():
        frames=omega if method=='Omega_depth' else cache_frames(method,source)
        if method=='Standard_depth':check_saved_standard_points(frames)
        lookup={r['source_id']:r for r in frames};rows=[]
        for frame in frames:
            queries={region:prepare_queries(frame,points) for region,points in refs[frame['name']].items()}
            if method not in ['Omega_depth','Standard_depth']:
                queries={k:v for k,v in queries.items() if k in ['foreground','foreground_boundary_common_grid','leaf_window','pot_rim_window']}
            for offset in OFFSETS:
                for sign in [-1,1]:
                    target=lookup.get(frame['source_id']+offset*sign)
                    if target is None:continue
                    for region,query in queries.items():
                        for control in ['mask_finite','confidence_gt1']:
                            rows.append(dict(method=method,source=frame['name'],target=target['name'],source_offset=offset*sign,region=region,control=control,**pair_metrics(query,frame,target,control)))
            if method in ['Omega_depth','Standard_depth']:
                selected=frame['mask']&np.isfinite(frame['conf']);edge=frame['mask']&~binary_erosion(frame['mask'])
                input_rows.append(dict(method=method,name=frame['name'],source_id=frame['source_id'],height=frame['mask'].shape[0],width=frame['mask'].shape[1],foreground_pixels=int(frame['mask'].sum()),
                    foreground_finite_positive_depth=int((selected&np.isfinite(frame['depth'])&(frame['depth']>0)).sum()),confidence_floor_fraction=float(((frame['conf']<=1)&selected).sum()/max(selected.sum(),1)),
                    foreground_confidence_p10=q(frame['conf'][selected],.1),foreground_confidence_p50=q(frame['conf'][selected],.5),foreground_confidence_p90=q(frame['conf'][selected],.9),
                    boundary_confidence_p50=q(frame['conf'][edge],.5),foreground_depth_p50=q(frame['depth'][selected],.5)))
            if (frame['index']+1)%32==0:print(method,'paired',frame['index']+1,'/',len(frames),flush=True)
        write_csv(AUDIT/'geometry_comparison'/f'multiview_{method}_pairs.csv',rows);all_rows.extend(rows)
        if method=='Standard_depth':
            # Small replay input cache for separate blur/mask/source checks.
            cache=Path('/tmp/foresplat_root_cause_geometry_input');cache.mkdir(exist_ok=True)
            np.savez(cache/'standard_model_inputs.npz',rgb=np.array([f['rgb'] for f in frames]),masks=np.array([f['mask'] for f in frames]),names=np.array(source['names']))
        if method not in ['Omega_depth']:del frames
    summary=summarize(all_rows);write_csv(AUDIT/'geometry_comparison/multiview_summary.csv',summary);write_json(AUDIT/'geometry_comparison/multiview_summary.json',summary)
    write_csv(AUDIT/'geometry_comparison/primary_depth_confidence_per_frame.csv',input_rows)
    write_json(AUDIT/'geometry_comparison/multiview_protocol.json',dict(offsets_original_source_id=OFFSETS,views=208,
        shared_query_reference='Actual Omega model-mask pixels mapped by saved A^-1 to common original-source pixel coordinates; same query coordinates before each model eligibility test; at most1500 source-number-seeded samples/ROI',
        regions=ROIS,boundary='One pixel inner boundary on common384x688 reference grid, includes plant/pot/blue, evaluation only',
        confidence_controls=['mask_finite','confidence_gt1'],occlusion='Source projected z-buffer1.005; target predicted occlusion when projected source depth >1.05 target depth. Occlusion can also reflect erroneous predictions; all-query support and raw-overlap errors retain that failure evidence.',
        sampling='Queries only; all208saved depth/pointmaps and cameras read; no output point averaging, no raw prediction or camera modifications',
        relative_errors='depth error divided by target depth; cycle error in original pixels; no metric/GT scale correction',
        limitations='Predictions across different input views provide internal multiview evidence, not absolute geometry accuracy. Independent RGB epipolar evidence is in camera_projection_comparison/epipolar*. Saved full point_head does not exist for Omega/standard; do not infer learned point-head accuracy from depth unprojection.'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in sources:
        subset=[r for r in summary if r['method']==method and r['control']=='mask_finite' and r['region']=='foreground']
        axes[0].plot([r['source_offset'] for r in subset],[r['support_5pct_fraction'] for r in subset],marker='o',linewidth=.8,label=method)
        axes[1].plot([r['source_offset'] for r in subset],[r['raw_overlap_relative_depth_error_p50_across_pair_median'] for r in subset],marker='o',linewidth=.8)
    axes[0].set_ylabel('Common FG query support at5% relative depth');axes[1].set_ylabel('Raw overlap depth error / target depth');axes[1].set_yscale('log')
    for ax in axes:ax.set_xlabel('Original source frame offset')
    axes[0].legend(fontsize=7,ncol=2);fig.suptitle('Actual208saved views; bidirectional multi-offset geometry\nPredicted occlusion reported separately; no GT accuracy or scale claim');fig.tight_layout();fig.savefig(AUDIT/'geometry_comparison/multiview_support_by_offset.png',dpi=160);plt.close(fig)
    print('Multiview audit complete:',len(all_rows),'pairs/regions/controls',flush=True)

if __name__=='__main__':main()
