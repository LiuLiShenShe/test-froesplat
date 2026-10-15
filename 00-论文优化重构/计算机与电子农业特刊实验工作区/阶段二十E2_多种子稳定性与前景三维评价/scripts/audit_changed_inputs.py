"""Read-only audit and faithful display of the four frozen changed training views."""
from __future__ import annotations
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


def main():
    selected=read_csv(E0/'FULL_VIEW_R0_R1_SELECTION.csv')
    inputs=read_csv(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv')
    candidates=read_csv(E0/'FULL_VIEW_CANDIDATE_MANIFEST.csv')
    raw_masks=read_csv(E0/'FULL_VIEW_R0_R1_MASK_MANIFEST.csv')
    changed=[r for r in selected if r['pixelwise_identity']=='False' and r['split']=='train']
    assert {r['frame_id'] for r in changed}=={'WanNianQing1_0037','WanNianQing1_0038','WanNianQing1_0165','CaoMei2_0139'}
    output=[]
    for row in changed:
        frame=row['frame_id']; item={'frame_id':frame,'sample':row['sample'],'registered_name':row['registered_name'],'split':'train','ROI_is_independent_GT':False}
        group_rows=[next(r for r in inputs if r['frame_id']==frame and r['group']==g) for g in GROUPS]
        reference=np.array(Image.open(group_rows[0]['common_rgb_path']).convert('RGB'))
        images=[];masks=[]
        for g,r in zip(GROUPS,group_rows):
            assert sha256(r['loader_image_path'])==r['image_sha256']
            assert sha256(r['mask_path'])==r['mask_sha256']
            assert sha256(r['common_rgb_path'])==r['common_rgb_sha256']
            image=np.array(Image.open(r['loader_image_path']));mask=np.array(Image.open(r['mask_path']))
            assert image.shape[2]==4 and np.array_equal(image[:,:,3],mask)
            assert set(np.unique(mask))<={0,255}
            inside=mask>127
            assert np.array_equal(image[:,:,:3][inside],reference[inside])
            assert not image[:,:,:3][~inside].any()
            images.append(image[:,:,:3]);masks.append(mask)
            item.update({f'{g}_image_path':r['loader_image_path'],f'{g}_image_sha256':r['image_sha256'],
                         f'{g}_mask_path':r['mask_path'],f'{g}_mask_sha256':r['mask_sha256'],
                         f'{g}_foreground_fraction':float(inside.mean()),f'{g}_candidate_id':row['R0_candidate_id' if g=='B_R0' else 'R1_candidate_id']})
        item['frozen_training_mask_changed_fraction']=float((masks[0]!=masks[1]).mean())
        item['canonical_rgb_path']=group_rows[0]['common_rgb_path'];item['canonical_rgb_sha256']=group_rows[0]['common_rgb_sha256']
        fig,axes=plt.subplots(1,4,figsize=(9,4),constrained_layout=True)
        for ax,array,title in zip(axes,[reference,*images,(masks[0]!=masks[1]).astype(float)],['Canonical undistorted RGB','B_R0 frozen black composite','C_R1 frozen black composite','Frozen mask difference']):
            ax.imshow(array,cmap='gray',vmin=0,vmax=1,interpolation='nearest') if array.ndim==2 else ax.imshow(array,interpolation='nearest')
            ax.set_title(title,fontsize=8);ax.axis('off')
        fig.suptitle(f'{frame} -> {row["registered_name"]}; TRAIN input only, not heldout evidence',fontsize=10)
        figure=STAGE/f'figures/INPUT_{frame}.png';fig.savefig(figure,dpi=130);plt.close(fig)
        item['input_figure_path']=str(figure);item['input_figure_sha256']=sha256(figure)
        cc=[r for r in candidates if r['frame_id']==frame]
        fig,axes=plt.subplots(1,len(cc),figsize=(2.2*len(cc),4),squeeze=False,constrained_layout=True)
        for ax,r in zip(axes[0],cc):
            assert sha256(r['mask_path'])==r['mask_sha256']
            assert sha256(r['metadata_path'])==r['metadata_sha256']
            candidate=np.array(Image.open(r['mask_path']))
            assert set(np.unique(candidate))<={0,255}
            policy=[p['policy'] for p in raw_masks if p['frame_id']==frame and p['candidate_id']==r['candidate_id']]
            for p in raw_masks:
                if p['frame_id']==frame and p['candidate_id']==r['candidate_id']:
                    assert sha256(p['mask_path'])==p['mask_sha256']==r['mask_sha256']
            ax.imshow(candidate,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.axis('off')
            ax.set_title(r['candidate_id'].rsplit('_',1)[-1]+' '+','.join(policy),fontsize=10)
        fig.suptitle(f'{frame}: original raw P6 candidate masks; no independent foreground GT',fontsize=9)
        figure=STAGE/f'figures/CANDIDATES_{frame}.png';fig.savefig(figure,dpi=130);plt.close(fig)
        item['candidate_figure_path']=str(figure);item['candidate_figure_sha256']=sha256(figure)
        output.append(item)
    write_csv(STAGE/'reports/CHANGED_TRAINING_INPUT_AUDIT.csv',output)
    write_json(STAGE/'reports/CHANGED_TRAINING_INPUT_AUDIT.json',{'status':'PASS','timestamp':now(),'changed_training_views':len(output),
        'original_candidate_masks_read':sum(1 for r in candidates if r['frame_id'] in {s['frame_id'] for s in changed}),
        'canonical_inside_RGB_exact':True,'outside_composite_RGB_exactly_zero':True,'alpha_equals_frozen_mask':True,
        'source_masks_modified':False,'views_are_training_not_heldout':True,'all_visualizations_are_actual_inputs':True,
        'raw0139_registered_mapping':'CaoMei2_0139 -> crop_0124.png','independent_foreground_GT':False})
    print('CHANGED_INPUT_AUDIT_PASS',len(output),flush=True)

if __name__=='__main__':main()
