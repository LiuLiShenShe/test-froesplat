"""Scientific displays from actual saved tensors; never used to compute metrics."""
from __future__ import annotations
import math
import statistics
from common import *
from metric_core import resize_reference,load_roi


def main():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':120})
    rows=read_csv(STAGE/'MULTISEED_PER_VIEW_METRICS.csv')
    lookup={(r['sample'],r['group'],int(r['seed']),r['loader_image_name']):r for r in rows}
    fixed=read_csv(STAGE/'configs/FIXED_FIGURE_VIEWS.csv')
    exploratory=read_csv(STAGE/'configs/EXPLORATORY_FIGURE_VIEWS.csv')
    print('FIXED_SELECTION_COLUMNS',list(fixed[0]),flush=True)
    choices={}
    for row in fixed+exploratory:
        key=row['sample'],row['loader_image_name']
        choices.setdefault(key,[]).append(row)
    index=[];omissions=[]
    for (sample,name),selections in choices.items():
        models={};records=[]
        for group in GROUPS:
            for seed in SEEDS:
                row=lookup[sample,group,seed,name];p=Path(row['float_render_path'])
                assert sha256(p)==row['float_render_sha256']
                models[group,seed]=torch.load(p,map_location='cpu',weights_only=True)
                records.append({'group':group,'seed':seed,'float_render_path':str(p),'sha256':row['float_render_sha256']})
        example=lookup[sample,'B_R0',0,name]
        width,height=int(example['render_width']),int(example['render_height'])
        reference=resize_reference(example['reference_path'],width,height).permute(1,2,0).numpy()
        roi=load_roi(example['ROI_path'],width,height).numpy()
        selected_fixed=any(r in fixed for r in selections)
        label='PREDECLARED CAMERA ORDER' if selected_fixed else 'EXPLORATORY: SEED 0 FOREGROUND RANKING'
        roles='; '.join(r.get('display_role',r.get('selection','fixed_camera')) for r in selections)
        def save(fig,kind,**extra):
            p=STAGE/f'figures/{sample}_{name}_{kind}.png'
            fig.suptitle(f'{sample} / {name} | {label}\n{roles}',fontsize=12)
            fig.savefig(p,bbox_inches='tight',facecolor='white');plt.close(fig)
            index.append({'sample':sample,'loader_image_name':name,'figure_kind':kind,'path':str(p),
                          'sha256':sha256(p),'selection_basis':label,'roles':roles,
                          'float_sources':json.dumps(records,ensure_ascii=False),'independent_GT':False,
                          'display_resampling':'matplotlib only; no input enhancement or metric from PNG',**extra})
        import json
        fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
        axes[0,0].imshow(reference,interpolation='none');axes[0,0].set_title('Canonical raw RGB')
        axes[1,0].imshow(roi,cmap='gray',vmin=0,vmax=1,interpolation='none');axes[1,0].set_title('Fixed R0 proxy ROI (non-GT)')
        for gi,group in enumerate(GROUPS):
            for seed in SEEDS:
                ax=axes[gi,seed+1];ax.imshow(models[group,seed]['rgb'].permute(1,2,0).numpy(),interpolation='none')
                ax.set_title(f'{group} seed {seed}')
        for ax in axes.ravel():ax.set_axis_off()
        save(fig,'RGB')
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        clipped=[]
        for seed,ax in zip(SEEDS,axes):
            difference=(models['C_R1',seed]['rgb']-models['B_R0',seed]['rgb']).abs().mean(0).numpy()
            im=ax.imshow(difference,cmap='magma',vmin=0,vmax=.1,interpolation='none')
            clipped.append(float((difference>.1).mean()))
            ax.set_title(f'Seed {seed}: mean |C - B|');ax.set_axis_off()
        fig.colorbar(im,ax=axes,label='Mean absolute RGB error',shrink=.7)
        save(fig,'BC_ABSOLUTE_DIFFERENCE',display_limits='[0,0.1] common to every seed/view',display_clipped_pixel_fractions=json.dumps(clipped))
        for kind in ('ALPHA','DEPTH','NORMAL'):
            fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
            if kind=='DEPTH':
                valid_depths=[]
                for tensor in models.values():
                    alpha=tensor['geometry_replay_alpha']
                    support=(alpha>.5)&torch.isfinite(tensor['depth'])&(tensor['depth']>0)
                    valid_depths.append(tensor['depth'][support].numpy())
                pooled=np.concatenate(valid_depths)
                if not len(pooled):
                    plt.close(fig)
                    omissions.append({'sample':sample,'loader_image_name':name,'figure_kind':'DEPTH','reason':'No positive finite alpha-supported depth in any of the six actual models'})
                    continue
                low,high=np.quantile(pooled,[.01,.99])
                if high<=low:high=low+1e-6
            for gi,group in enumerate(GROUPS):
                for seed in SEEDS:
                    t=models[group,seed];ax=axes[gi,seed]
                    if kind=='ALPHA':
                        im=ax.imshow(t['alpha'].numpy(),cmap='gray',vmin=0,vmax=1,interpolation='none')
                    elif kind=='DEPTH':
                        array=t['depth'].numpy().copy();alpha=t['geometry_replay_alpha'].numpy()
                        array[(alpha<=.5)|~np.isfinite(array)|(array<=0)]=np.nan
                        im=ax.imshow(array,cmap='viridis',vmin=low,vmax=high,interpolation='none')
                    else:
                        normal=t['normal'];norm=torch.linalg.vector_norm(normal,dim=0)
                        valid=(t['geometry_replay_alpha']>.5)&torch.isfinite(normal).all(0)&(norm>1e-6)
                        array=((normal/norm.clamp_min(1e-12)[None]+1)*.5).permute(1,2,0).numpy()
                        array[~valid.numpy()]=0
                        ax.imshow(array.clip(0,1),interpolation='none')
                    ax.set_title(f'{group} seed {seed}');ax.set_axis_off()
            extra={}
            if kind=='DEPTH':
                fig.colorbar(im,ax=axes,label='Inherited arbitrary COLMAP units; shared 1%-99% display limits',shrink=.7)
                extra={'display_limits':f'{low:.9g},{high:.9g}','invalid_depth_display':'blank; alpha <= 0.5 excluded'}
            elif kind=='ALPHA':fig.colorbar(im,ax=axes,label='Rendered alpha',shrink=.7)
            else:extra={'normal_RGB_mapping':'(unit world-space rendered normal + 1)/2; alpha <=0.5 black'}
            save(fig,kind,**extra)
        print('FIGURES_COMPLETE',sample,name,flush=True)
    # Show all paired effects as three training repeats, without treating views as repeats.
    effects=read_csv(STAGE/'PAIRED_SEED_EFFECTS.csv')
    for metric,ylabel in [('PSNR','Full raw RGB PSNR difference (dB)'),('foreground_PSNR','Fixed R0 proxy foreground PSNR difference (dB)')]:
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        for scene_index,scene in enumerate(SCENES):
            part=sorted([r for r in effects if r['sample']==scene],key=lambda r:int(r['seed']))
            values=[float(r[f'delta_{metric}']) for r in part]
            xs=[scene_index+offset for offset in (-.13,0,.13)]
            ax.scatter(xs,values,s=70)
            for x,v,r in zip(xs,values,part):ax.annotate(f'S{r["seed"]}',(x,v),xytext=(5,6),textcoords='offset points')
            ax.errorbar(scene_index,statistics.mean(values),yerr=statistics.stdev(values),fmt='ks',capsize=6,label='Mean +/- sample SD' if scene_index==0 else None)
        ax.axhline(0,color='gray',linewidth=1);ax.set_xticks(range(2),SCENES);ax.set_ylabel(ylabel);ax.set_xlim(-.5,1.5)
        ax.set_title('R1 - R0 paired effects: n = 3 training seeds per scene');ax.legend()
        p=STAGE/f'figures/PAIRED_EFFECT_{metric}.png';fig.savefig(p);plt.close(fig)
        index.append({'sample':'PAIRED_TWO_SCENES','loader_image_name':'ALL_HELDOUT_AVERAGED_PER_TRAINING',
                      'figure_kind':'PAIRED_EFFECT','path':str(p),'sha256':sha256(p),'selection_basis':'ALL_PREDECLARED_SEEDS_NO_EXCLUSIONS','metric':metric,'independent_GT':False})
    fields=list(dict.fromkeys(k for row in index for k in row))
    write_csv(STAGE/'reports/FIGURE_INDEX.csv',index,fields)
    write_csv(STAGE/'reports/FIGURE_OMISSIONS.csv',omissions,['sample','loader_image_name','figure_kind','reason'])
    write_json(STAGE/'reports/FIGURE_GENERATION.json',{'status':'PASS','timestamp':now(),'views':len(choices),'figures':len(index),'omissions':omissions,'script_sha256':sha256(Path(__file__)),
                                                     'PNG_not_used_for_formal_metrics':True,'real_float_sources_only':True,'all_seed_pairs_displayed':True})

if __name__=='__main__':main()
