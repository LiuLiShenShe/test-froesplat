"""Single-seed, one-sequence descriptive variability and vector evidence figures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'; os.environ['MPLBACKEND']='Agg'
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
EXPS=['E1','E5','E7','E8','E9','E10']
COLORS=['#000000','#999999','#56B4E9','#0072B2','#E69F00','#D55E00']
METRICS=['full_psnr','fg_mask_psnr','inner_fg_edge5_psnr','ssim','lpips_alex','fg_gradient_retention','fg_gradient_mae']

def main():
    tables={e:list(csv.DictReader((ROOT/'render_comparison'/e/'per_view_metrics.csv').open())) for e in EXPS}
    arrays={e:{m:np.array([float(r[m]) for r in sorted(tables[e],key=lambda r:int(r['source_id']))]) for m in METRICS} for e in EXPS}
    rows=[]
    for e in EXPS:
        for m in METRICS:
            v=arrays[e][m];assert len(v)==208 and np.isfinite(v).all()
            rows.append(dict(experiment=e,metric=m,view_count=208,mean=float(v.mean()),view_sd=float(v.std(ddof=1)),
                median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),min=float(v.min()),max=float(v.max())))
    out=ROOT/'render_comparison';figures=out/'figures';figures.mkdir(exist_ok=True)
    with (out/'metric_view_variability.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for ax,m,label in zip(axes,METRICS[:3],['Full PSNR (dB)','True foreground PSNR (dB)','Inner foreground edge PSNR (dB)']):
        values=[arrays[e][m] for e in EXPS]
        b=ax.boxplot(values,patch_artist=True,showfliers=True,medianprops={'color':'black'})
        for patch,color in zip(b['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.7)
        ax.set_xticks(range(1,7),EXPS);ax.set_ylabel(label);ax.set_ylim(0,max(v.max() for v in values)*1.05);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figures/'six_model_view_distributions.pdf');fig.savefig(figures/'six_model_view_distributions.png',dpi=180);plt.close(fig)
    pairs=[('E1','E7'),('E5','E9'),('E7','E8'),('E9','E10')]
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for ax,m,label in zip(axes,METRICS[:3],['Full PSNR gain (dB)','Foreground PSNR gain (dB)','Inner edge PSNR gain (dB)']):
        values=[arrays[b][m]-arrays[a][m] for a,b in pairs]
        box=ax.boxplot(values,patch_artist=True,showfliers=True,medianprops={'color':'black'})
        for patch,color in zip(box['boxes'],['#0072B2','#D55E00','#56B4E9','#E69F00']):patch.set_facecolor(color);patch.set_alpha(.7)
        ax.axhline(0,color='black',linestyle='--',linewidth=.8);ax.set_xticks(range(1,5),[b+'−'+a for a,b in pairs],rotation=25);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figures/'geometry_and_interface_paired_gains.pdf');fig.savefig(figures/'geometry_and_interface_paired_gains.png',dpi=180);plt.close(fig)
    protocol=dict(seed=0,plants=1,views=208,unit='photographic view from one correlated acquisition sequence',
        stats='Descriptive mean, sample SD across views, median, p10/p90 and range. View SD is not seed SD or standard error.',
        significance_test='Not performed: one plant and one seed; neighboring photographic views are correlated. No iid confidence interval or independent-repetition p-value is claimed.',
        boxplot='Median line, Q1/Q3 box, whiskers1.5IQR and all plotted fliers; no observations removed.',
        figures='PDF vector charts, Okabe-Ito-derived colors and visible labels; PNG previews only. Native RGB evidence remains unresampled at leaf/edge crop scale.')
    (figures/'statistical_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (figures/'CAPTIONS.md').write_text('# Descriptive evidence captions\n\n'
        '`six_model_view_distributions.pdf`: All208 common training views, six actual30K checkpoints. Boxes show Q1/Q3 and the median, whiskers1.5IQR, dots retained outliers. These are correlated view distributions from one plant/seed0, not independent seed repetitions.\n\n'
        '`geometry_and_interface_paired_gains.pdf`: Differences on the same208 GT/Mask-hash-matched views. E7−E1 and E9−E5 describe original-background geometry gains under the old interface; E8−E7 and E10−E9 describe the separate interface gains. Historical checkpoints are re-rendered, not retrained. Boxes describe per-view variation; no significance test or iid confidence interval is asserted.\n')
    print('all208 descriptive variability and PDF figures saved; no independent-view significance claim')

if __name__=='__main__':main()
