#!/usr/bin/env python3
"""Plot verified completed GGGS measurements without rerendering or selecting a method."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--c5-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    table=args.comparison/'comparisons/ALL_GGGS_CLEANING_RESULTS.csv'
    rows=list(csv.DictReader(table.open()))
    methods=['C0','C1_raw','C1_adaptive','C2_raw','C2_adaptive','C3_pre_marker','C3','C4']
    selected=[next(r for r in rows if r['method']==method and r['plant']=='KongQueZhuYu') for method in methods]
    selected.append(next(r for r in rows if Path(r['experiment']).resolve()==args.c5_output.resolve()))
    original=selected[0]['input_sha256']
    records=[]
    for row in selected:
        if row['input_sha256']!=original or int(row['evaluated_views'])!=208:
            raise ValueError('Scientific comparison requires the same E10 input and all 208 cameras')
        if sha(Path(row['native_ply']))!=row['native_sha256']:
            raise ValueError('A plotted native model changed after completed measurement')
        report=json.loads(Path(row['report']).read_text())
        mean=report.get('metrics',report.get('final'))['mean']
        records.append(dict(method=row['method'],version=row['experiment_version'],
            native_ply=row['native_ply'],native_sha256=row['native_sha256'],report=row['report'],report_sha256=sha(Path(row['report'])),
            background_reduction=float(row['background_alpha_mass_reduction']),fg_mse=mean['fg_mse'],
            dark_fg_mse=mean['dark_foreground_mse'],fg_psnr=mean['fg_psnr'],
            black_cloud_reduction=float(row['black_largest_dark_bg_component_pixels_reduction']),
            white_cloud_reduction=float(row['white_largest_dark_bg_component_pixels_reduction'])))
    old=records[0]
    labels=['C0','C1 raw','C1 relative','C2 raw','C2 relative','C3 pre-marker','C3 final','C4','C5 '+records[-1]['version'].replace('C5-development-','dev ')]
    colors=['#a5a5a5','#d78738','#d78738','#d78738','#d78738','#8665a7','#8665a7','#a5a5a5','#2c8564']
    x=np.arange(len(records))
    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    values=[100*r['background_reduction'] for r in records]
    bars=axes[0,0].bar(x,values,color=colors)
    axes[0,0].bar_label(bars,fmt='%.2f',fontsize=9)
    axes[0,0].set_ylabel('Background alpha mass reduction (%)')
    axes[0,0].set_ylim(0,110)
    for ax,key,title in [(axes[0,1],'fg_mse','Full-mask foreground MSE change (%)'),
                         (axes[1,0],'dark_fg_mse','Target-dark foreground MSE change (%)')]:
        values=[100*(r[key]-old[key])/old[key] for r in records]
        bars=ax.bar(x,values,color=colors)
        ax.bar_label(bars,fmt='%.2f',fontsize=9)
        ax.axhline(0,color='#333333',lw=.8)
        ax.set_ylabel(title+'; positive is worse')
        ax.margins(y=.2)
    for offset,key,color,label in [(-.18,'black_cloud_reduction','#444444','Black display background'),
                                    (.18,'white_cloud_reduction','#94b8ca','White display background')]:
        axes[1,1].bar(x+offset,[100*r[key] for r in records],width=.36,color=color,label=label)
    axes[1,1].set_ylabel('Largest actual dark-background component reduction (%)')
    axes[1,1].set_ylim(0,110)
    axes[1,1].legend(fontsize=9,loc='upper left',ncol=2,frameon=False)
    for ax in axes.flat:
        ax.set_xticks(x,labels,rotation=32,ha='right')
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same original GGGS / 208 cameras / 960-pixel maximum side\nMeasured cloud suppression and foreground damage; development comparison',fontsize=15)
    args.output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf','svg']:
        fig.savefig(args.output/f'GGGS_CLEANING_TRADEOFF.{suffix}',dpi=180)
    plt.close(fig)
    write_json(args.output/'GGGS_CLEANING_TRADEOFF_SOURCE.json',dict(
        input_sha256=original,evaluated_cameras=208,source_table=str(table.resolve()),source_table_sha256=sha(table),
        c5_output=str(args.c5_output.resolve()),source_sha256=sha(Path(__file__)),measurements=records,
        scope='Completed measured comparison only; this plot never selects a global cleaner or certifies independent-test generalization',
        interpretation=['Average PSNR and mean MSE need not have the same direction; both raw MSE measures are shown.',
                        'C3 final includes historical marker exclusions; pre-marker is displayed separately.',
                        'Mean largest dark component reduction does not establish that all floating dots disappeared.',
                        'The plotted C5 source version is explicit; no interrupted D result is included.']))
    print(args.output.resolve())


if __name__=='__main__':
    main()
