"""Fixed-camera comparisons from preserved floats, with identical display scales."""
from common import *

def main():
    import numpy as np,torch,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from figures import normalize_depth
    core=metric_module();manifest=[]
    for sample in SCENES:
        records={m:{r['loader_image_name']:r for r in rows(STAGE/f'metrics/{sample}_{m}_FULL.csv')} for m in ['M0','M1','M2']}
        ordered=sorted(records['M0']);selected=[ordered[i] for i in [0,len(ordered)//2,len(ordered)-1]]
        depth_max=read_json(STAGE/f'configs/{sample}_MESH.json')['depth_trunc']
        for name in selected:
            rec=records['M0'][name];w,h=int(rec['render_width']),int(rec['render_height'])
            reference=core.resize_reference(rec['reference_path'],w,h).permute(1,2,0).numpy()
            roi=core.load_roi(rec['ROI_path'],w,h).numpy()
            data={m:torch.load(records[m][name]['float_render_path'],map_location='cpu',weights_only=True) for m in records}
            fig,axes=plt.subplots(3,7,figsize=(18,11))
            for index,method in enumerate(['M0','M1','M2']):
                d=data[method];rgb=d['rgb'].permute(1,2,0).numpy()
                diff=np.abs(rgb-data['M0']['rgb'].permute(1,2,0).numpy()).mean(2)
                post=np.load(DATA/'mesh_reprojection'/sample/method/f'post_{name}.npz')
                normal=(d['normal'].permute(1,2,0).numpy()+1)/2
                normal[d['geometry_replay_alpha'].numpy()<=.5]=0
                panels=[('Canonical raw RGB',reference,'rgb'),(f'{method}: Gaussian RGB',rgb,'rgb'),('Gaussian alpha',d['alpha'].numpy(),'gray'),
                        ('Mean |RGB - M0| [0, 0.1]',diff,'diff'),('Gaussian depth',normalize_depth(d['depth'].numpy(),0,depth_max),'rgb'),
                        ('Post Mesh depth',normalize_depth(post['depth'],0,depth_max),'rgb'),('Gaussian normal',normal,'rgb')]
                for ax,(title,image,kind) in zip(axes[index],panels):
                    if kind=='diff':ax.imshow(image,cmap='magma',vmin=0,vmax=.1)
                    elif kind=='gray':ax.imshow(image,cmap='gray',vmin=0,vmax=1)
                    else:ax.imshow(image)
                    ax.set_title(title,fontsize=9);ax.axis('off')
            fig.suptitle(f'{sample} / fixed held-out {name} | M0, M1, M2 single-seed pilot\nIdentical image and depth scales; RGB difference is descriptive; ROI is a fixed prediction, no independent geometry GT')
            fig.tight_layout();path=STAGE/f'figures/{sample}_{name}_M0_M1_M2_COMPARISON.png';fig.savefig(path,dpi=130);plt.close(fig)
            manifest.append({'sample':sample,'camera':name,'path':str(path),'sha256':sha(path),'selection':'Sorted held-out first/middle/last; no metric ranking','difference_scale':[0,.1],'depth_range':[0,depth_max],
                             'float_paths':{m:records[m][name]['float_render_path'] for m in records},'ROI_pixels':int(roi.sum())})
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    for ax,sample in zip(axes,SCENES):
        for method,color in [('M1','#4477aa'),('M2','#cc6677')]:
            values=rows(STAGE/f'metrics/{sample}_{method}_PRUNING.csv');x=[int(r['iteration']) for r in values]
            before=np.array([int(r['Gaussian_count_before']) for r in values]);after=np.array([int(r['Gaussian_count_after']) for r in values])
            ax.plot(x,after,'o-',label=f'{method}: after extra prune',color=color)
            ax.vlines(x,after,before,color=color,alpha=.6)
            if method=='M2':ax.plot(x,[int(r['floor_count']) for r in values],':',label='M2 global floor',color='#228833')
        ax.set_title(sample);ax.set_xlabel('Iteration');ax.set_ylabel('Gaussian count');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Real extra-pruning events: vertical segments show removal; floor setting alone does not prove benefit')
    fig.tight_layout();fig.savefig(STAGE/'figures/M2M3_PRUNING_EVENTS.png',dpi=150);plt.close(fig)
    write_json(STAGE/'reports/COMPARISON_FIGURES.json',{'status':'PASS','selection':'Rule declared before formal M1/M2 outcomes','figures':manifest,'postprocessing':'Display scaling only; no model output editing'})
if __name__=='__main__':main()
