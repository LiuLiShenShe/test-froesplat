"""Saved depth projection maps with identical source pixels and error colors."""
from geometry_common import *
from geometry_multiview_audit import world_from_depth,prepare_queries

def main():
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    sources=camera_sources();selected={}
    with np.load('/tmp/foresplat_root_cause_geometry_input/standard_model_inputs.npz') as f:masks=f['masks']
    def load(method,sid):
        if (method,sid) in selected:return selected[(method,sid)]
        s=sources[method];n=f'crop_{sid:04d}.png';i=s['names'].index(n);stem=Path(n).stem
        if method=='Omega_depth':
            depth=np.load(OMEGA_DATA/'per_frame'/f'depth_{stem}.npy').squeeze();conf=np.load(OMEGA_DATA/'per_frame'/f'confidence_{stem}.npy').squeeze();mask=np.load(OMEGA_DATA/'per_frame'/f'model_mask_{stem}.npy').squeeze()>0
            rgb=np.load(OMEGA_DATA/'per_frame'/f'model_rgb_{stem}.npy')
        else:
            depth=np.load(STANDARD/'Standard_VGGT_Frontend/depth'/f'{stem}.npy').squeeze();conf=np.load(STANDARD/'Standard_VGGT_Frontend/confidence'/f'{stem}.npy').squeeze();mask=masks[i];rgb=None
        frame=dict(mask=mask,depth=depth,conf=conf,xyz=world_from_depth(depth,s['c2w'][i],s['k_model'][i]),w2c=s['w2c'][i],k=s['k_model'][i],a=s['a_actual'][i],rgb=rgb)
        selected[(method,sid)]=frame;return frame
    pairs=[(0,5),(86,87),(162,163)];fig,axes=plt.subplots(len(pairs),3,figsize=(9,13));normal=Normalize(-.25,.25);cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('.75');records=[]
    for row,(sid,tid) in enumerate(pairs):
        omega=load('Omega_depth',sid);y,x=np.nonzero(omega['mask']);original=np.column_stack((x,y,np.ones(len(x))))@np.linalg.inv(omega['a']).T
        axes[row,0].imshow(omega['rgb']);axes[row,0].set_title(f'Actual saved Omega source RGB\n{sid:04d} -> {tid:04d}',fontsize=9)
        for col,method in enumerate(['Omega_depth','Standard_depth'],1):
            source=load(method,sid);target=load(method,tid);query=prepare_queries(source,original[:,:2]);camera=query['xyz']@target['w2c'][:3,:3].T+target['w2c'][:3,3];z=camera[:,2];uv=camera[:,:2]/np.maximum(z[:,None],1e-15);uv=uv@target['k'][:2,:2].T+target['k'][:2,2]
            pix=np.rint(np.nan_to_num(uv,nan=-1e9,posinf=-1e9,neginf=-1e9)).astype(int);h,w=target['mask'].shape;inside=query['finite']&(z>0)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h);ids=np.flatnonzero(inside);px,py=pix[ids].T;tz=target['depth'][py,px]
            good=target['mask'][py,px]&np.isfinite(tz)&(tz>0);ids=ids[good];tz=tz[good];relative=(z[ids]-tz)/tz
            display=np.full(omega['mask'].shape,np.nan);display[y[ids],x[ids]]=relative;axes[row,col].imshow(display,cmap=cmap,norm=normal)
            overlap=len(ids)/len(x);support=float(np.sum(np.abs(relative)<=.05)/len(x));axes[row,col].set_title(f'{method}\nFG overlap {overlap:.1%}; within5% {support:.1%}',fontsize=9)
            records.append(dict(method=method,source_id=sid,target_id=tid,reference_foreground_queries=len(x),overlap_fraction=overlap,within5pct_all_query_fraction=support,
                scope='Dense visualization at all common Omega FG source pixels, no z-buffer eligibility. Main statistical table uses seeded<=1500 query samples and explicit predicted-occlusion/z-buffer accounting; numbers are distinct controls.'))
        for ax in axes[row]:ax.axis('off')
    fig.suptitle('Saved-depth projection residuals on identical source foreground pixels\nGray: no positive-depth target foreground overlap; signed errors include predicted occlusion, no GT claim');fig.tight_layout(rect=(0,0.04,1,.96));fig.colorbar(plt.cm.ScalarMappable(norm=normal,cmap=cmap),ax=axes.ravel().tolist(),orientation='horizontal',fraction=.025,pad=.02,label='(Projected source depth - saved target depth) / saved target depth; clipped +/-25%')
    fig.savefig(AUDIT/'geometry_comparison/cross_view_depth_projection_maps.png',dpi=170);plt.close(fig);write_csv(AUDIT/'geometry_comparison/cross_view_depth_projection_maps.csv',records)
    print('Saved-depth common-source projection maps complete')

if __name__=='__main__':main()
