"""Saved-depth cross-view overlay with actual training-resolution camera K.

Depth-derived support is endogenous; these images are coordinate diagnostics.
Independent RGB correspondences are measured separately by epipolar audit.
"""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from camera_numeric_audit import OMEGA,STANDARD,OUT

def main():
    chain=json.loads((OUT/'camera_matrix_chain.json').read_text())
    ods=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
    names=sorted([k.split('/')[1] for k in chain if k.startswith('VGGT_Omega/')])
    pairs=[(0,5),(26,41),(78,108),(130,156)]
    result=[]
    for i,j in pairs:
        fig,ax=plt.subplots(2,3,figsize=(10,11))
        for row,method in enumerate(['VGGT_Omega','Standard_VGGT']):
            n1,n2=names[i],names[j]; c1,c2=chain[method+'/'+n1],chain[method+'/'+n2]
            E1=np.array(c1['W2C_colmap']);E2=np.array(c2['W2C_colmap']);K1=np.array(c1['K_model']);A1=np.array(c1['A_recorded']);A2=np.array(c2['A_recorded'])
            if method=='VGGT_Omega':
                depth=np.load(ods/'per_frame'/('depth_'+Path(n1).stem+'.npy'));mask=np.load(ods/'per_frame'/('model_mask_'+Path(n1).stem+'.npy')).astype(bool)
                raw=json.loads((ods/'per_frame/cameras.json').read_text());Eraw=np.eye(4);Eraw[:3,:4]=raw['frames'][i]['extrinsic_model_w2c'];pose=np.linalg.inv(Eraw);alignment=np.array(raw['scene_alignment_matrix'])
            else:
                depth=np.load(STANDARD/'Standard_VGGT_Frontend/depth'/ (Path(n1).stem+'.npy'))
                with Image.open(ods/'masks'/n1) as mm:
                    padded=Image.new('L',(3840,3840),0);padded.paste(mm.convert('L'),(840,0));mask=np.array(padded.resize((518,518),Image.Resampling.NEAREST))>127
                pose=np.linalg.inv(E1);alignment=np.eye(4)
            depth=depth.squeeze();y,x=np.nonzero(mask & np.isfinite(depth) & (depth>0));keep=np.arange(0,len(x),3);x,y=x[keep],y[keep]
            ray=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(K1).T;Xcam=ray*depth[y,x,None];Xraw=Xcam@pose[:3,:3].T+pose[:3,3];Xworld=Xraw@alignment[:3,:3].T+alignment[:3,3]
            Xdest=Xworld@E2[:3,:3].T+E2[:3,3];positive=Xdest[:,2]>0
            gt=np.array(Image.open(ods/'images'/n2).convert('RGB').resize((1080,1920)))
            source=np.array(Image.open(ods/'images'/n1).convert('RGB').resize((1080,1920)))
            orig=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(A1).T;uvsrc=orig[:,:2]*.5-.25;si=np.rint(uvsrc).astype(int);colors=source[np.clip(si[:,1],0,1919),np.clip(si[:,0],0,1079)]/255.
            for col,kname in enumerate(['K_train_saved','K_gggs','K_train_true_pixel_transform']):
                K=np.array(c2[kname]);p=Xdest@K.T;uv=p[:,:2]/p[:,2:3];valid=positive&(uv[:,0]>=0)&(uv[:,0]<1080)&(uv[:,1]>=0)&(uv[:,1]<1920)&np.isfinite(uv).all(axis=1)
                ax[row,col].imshow(gt);ax[row,col].scatter(uv[valid,0],uv[valid,1],s=.3,c=colors[valid],alpha=.7,rasterized=True);ax[row,col].set_title(method+'\n'+kname);ax[row,col].axis('off')
                result.append(dict(method=method,source=n1,target=n2,K_scope=kname,projected_in_image=int(valid.sum()),source_sample_points=len(x)))
        fig.suptitle(names[i]+' → '+names[j]+'; source-depth RGB points over target training GT\nShared point size, image grid, colors; support is not independent camera ground truth');fig.tight_layout();fig.savefig(OUT/f'cross_view_overlay_{i:03d}_{j:03d}.png',dpi=160);plt.close(fig)
    (OUT/'projection_overlay_counts.json').write_text(json.dumps(result,indent=2));print('overlays_complete',len(pairs))

if __name__=='__main__':main()
