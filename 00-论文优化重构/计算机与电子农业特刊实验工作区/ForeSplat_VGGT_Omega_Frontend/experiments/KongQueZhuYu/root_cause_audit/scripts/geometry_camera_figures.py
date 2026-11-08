"""Scientific plots from saved camera metadata and original failure-view photos."""
from PIL import Image
from geometry_common import *

def main():
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    sources=camera_sources();matches=json.loads((AUDIT/'geometry_comparison/camera_trajectory_Sim3.json').read_text())
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,(method,s) in zip(axes.ravel(),sources.items()):
        p=s['c2w'];xyz=(p[:,:3,3]-p[0,:3,3])@p[0,:3,:3]
        scale=float(np.ptp(np.quantile(xyz,[.025,.975],axis=0),axis=0).max());xyz/=max(scale,1e-15)
        ax.plot(xyz[:,0],xyz[:,2],color='.7',linewidth=.55);ax.scatter(xyz[:,0],xyz[:,2],c=[source_id(n) for n in s['names']],cmap='viridis',vmin=0,vmax=209,s=7)
        ax.set_aspect('equal',adjustable='datalim');ax.set_title(method,fontsize=10);ax.set_xlabel('First camera x / own robust extent');ax.set_ylabel('First camera z / own robust extent')
    fig.suptitle('Independent208saved camera paths, identical source-ID colors\nEach method uses its own first-camera coordinates and display-only extent; no cross-method alignment');fig.tight_layout();fig.savefig(AUDIT/'geometry_comparison/camera_paths_independent.png',dpi=170);plt.close(fig)
    reference=sources['Omega_depth'];rp=reference['c2w'];center=rp[:, :3,3];scale=float(np.ptp(np.quantile(center,[.025,.975],axis=0),axis=0).max())
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for method in ['Standard_depth','AMB3R_SfM']:
        m=next(r for r in matches if r['method']==method);p=sources[method]['c2w'];aligned=m['similarity_scale']*p[:,:3,3]@np.asarray(m['rotation']).T+np.asarray(m['translation'])
        ax=axes[0 if method=='Standard_depth' else 1];a=(center-rp[0,:3,3])@rp[0,:3,:3]/scale;b=(aligned-rp[0,:3,3])@rp[0,:3,:3]/scale
        ax.plot(a[:,0],a[:,2],color='black',linewidth=.7,label='Omega reference');ax.scatter(a[:,0],a[:,2],color='black',s=6)
        ax.plot(b[:,0],b[:,2],color='tab:blue',linewidth=.7,label=method);ax.scatter(b[:,0],b[:,2],c=[source_id(n) for n in reference['names']],cmap='viridis',s=8)
        for i in range(0,len(a),8):ax.plot([a[i,0],b[i,0]],[a[i,2],b[i,2]],color='tab:red',linewidth=.35,alpha=.6)
        ax.set_title(method+' | '+('relative Sim3 supported' if m['reliable_similarity_comparison'] else 'reliable global Sim3 NOT established'),fontsize=10)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Omega first camera x / Omega extent');ax.set_ylabel('Omega first camera z / Omega extent');ax.legend(fontsize=8)
    fig.suptitle('Exact source-ID camera correspondence:156fit /52held-out\nDiagnostic relative camera agreement; no dataset transformation or GT pose claim');fig.tight_layout();fig.savefig(AUDIT/'geometry_comparison/camera_paths_Sim3_diagnostic.png',dpi=170);plt.close(fig)
    groups=[('Standard saved-camera jumps',[86,87,88]),('Omega saved-camera jumps',[162,163,164,165]),('Ordinary nearby views',[0,1])]
    fig,axes=plt.subplots(3,4,figsize=(11,13));rows=[]
    for row,(label,ids) in enumerate(groups):
        for col in range(4):
            ax=axes[row,col];ax.axis('off')
            if col>=len(ids):continue
            sid=ids[col];path=Path('/data/fj/F2DMAS/02-FFT/KongQueZhuYu')/f'{sid:04d}.jpg';image=Image.open(path).convert('RGB');image.thumbnail((540,960));ax.imshow(image)
            text=f'{sid:04d}'
            for method,prefix in [('Omega_depth','Omega'),('Standard_depth','Std')]:
                s=sources[method];i=s['names'].index(f'crop_{sid:04d}.png')
                angle=float(camera_angles(s['c2w'][i-1,:3,:3].T@s['c2w'][i,:3,:3])) if i else None
                if angle is not None:text+=f'\n{prefix} step {angle:.1f} deg'
                rows.append(dict(source_id=sid,method=method,rotation_from_previous_usable_view_deg=angle,original_rgb_path=str(path)))
            ax.set_title(text,fontsize=8)
        axes[row,0].text(-.15,.5,label,rotation=90,va='center',ha='center',transform=axes[row,0].transAxes,fontsize=10)
    fig.suptitle('Actual original02-FFT photos around large predicted camera jumps\nDirect visual inspection aids diagnosis; image similarity alone does not determine true camera motion');fig.tight_layout();fig.savefig(AUDIT/'geometry_comparison/camera_failure_source_photos.png',dpi=150);plt.close(fig)
    write_csv(AUDIT/'geometry_comparison/camera_failure_source_photos.csv',rows)
    print('Camera metadata/actual-source figures complete')

if __name__=='__main__':main()
