"""Show independent matched RGB points and model-predicted epipolar lines."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from camera_numeric_audit import OMEGA,STANDARD,OUT,colmap
from camera_epipolar_audit import fundamental,sampson

def load_rgb(ds,name):
    a=np.load(ds/'per_frame'/('model_rgb_'+Path(name).stem+'.npy'))
    if a.shape[0]==3:a=a.transpose(1,2,0)
    return np.clip(a,0,1) if a.dtype!=np.uint8 else a/255.

def main():
    ds=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
    frames=json.loads((ds/'per_frame/cameras.json').read_text())['frames'];A={f['image_name']:np.array(f['pixel_transform']) for f in frames}
    cams={'VGGT_Omega':colmap(ds),'Standard_VGGT':colmap(STANDARD/'Standard_VGGT_Dataset')}
    coords=json.loads((OUT/'epipolar_match_coordinates.json').read_text());chain=json.loads((OUT/'camera_matrix_chain.json').read_text())
    controls=[]
    for pair in ['0_1','30_31','75_76','135_136','180_181','0_5','75_80']:
        if pair not in coords:continue
        q=coords[pair];n1,n2=q['source1'],q['source2'];p1=np.array(q['p1']);p2=np.array(q['p2']);rgb1=load_rgb(ds,n1);rgb2=load_rgb(ds,n2)
        fig,ax=plt.subplots(1,3,figsize=(10,6))
        ax[0].imshow(np.hstack([rgb1,rgb2]));selection=np.linspace(0,len(p1)-1,min(14,len(p1))).astype(int)
        colors=plt.cm.tab20(np.arange(len(selection))/max(len(selection),1))
        for idx,color in zip(selection,colors):
            ax[0].plot([p1[idx,0],p2[idx,0]+384],[p1[idx,1],p2[idx,1]],color=color,lw=.5);ax[0].scatter([p1[idx,0],p2[idx,0]+384],[p1[idx,1],p2[idx,1]],s=8,color=color)
        ax[0].set_title('Same RGB-only RANSAC matches')
        for a,(method,cc) in zip(ax[1:],cams.items()):
            c1,c2=cc[n1],cc[n2];K1=A[n1]@c1['K'];K2=A[n2]@c2['K'];F=fundamental(K1,K2,c1['E'],c2['E']);err=sampson(F,p1,p2)
            a.imshow(rgb2)
            for idx,color in zip(selection,colors):
                line=F@np.r_[p1[idx],1]
                if abs(line[1])>1e-8:a.plot([0,383],[-line[2]/line[1],-(line[0]*383+line[2])/line[1]],lw=.7,color=color)
                else:a.axvline(-line[2]/line[0],color=color,lw=.7)
                a.scatter(p2[idx,0],p2[idx,1],s=15,color=color,edgecolor='white',linewidth=.4)
            a.set_title(method+'\nmedian %.2f shared-grid px'%np.median(err));a.set_xlim(0,383);a.set_ylim(687,0)
            # Only change camera principal point to match the actual pixel-centre
            # transform; E/focal remain exactly fixed.
            K1true=A[n1]@np.array(chain[method+'/'+n1]['K_colmap']);K2true=A[n2]@np.array(chain[method+'/'+n2]['K_colmap'])
            if method=='Standard_VGGT':
                for K in [K1true,K2true]:K[:2,2]+=(1/(518/3840)-1)/2*np.array([384/2160,688/3840])
            efix=sampson(fundamental(K1true,K2true,c1['E'],c2['E']),p1,p2)
            for region,mask in [('all',np.ones(len(p1),bool)),('upper_plant', (p1[:,1]<430)&(p2[:,1]<430)),('pot_lower',(p1[:,1]>=430)&(p2[:,1]>=430))]:
                controls.append(dict(pair=pair,method=method,region=region,n=int(mask.sum()),before=float(np.median(err[mask])) if mask.any() else None,half_pixel_corrected=float(np.median(efix[mask])) if mask.any() else None))
        for a in ax:a.axis('off')
        fig.suptitle(n1+' → '+n2+'; matched target points and saved-camera epipolar lines');fig.tight_layout();fig.savefig(OUT/f'epipolar_overlay_{pair}.png',dpi=180);plt.close(fig)
    (OUT/'epipolar_principal_point_control.json').write_text(json.dumps(controls,indent=2));print(json.dumps(controls[:12],indent=2))

if __name__=='__main__':main()
