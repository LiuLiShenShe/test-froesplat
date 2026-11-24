#!/usr/bin/env python3
"""Render 14 ForeSplat plates from the archived evidence without running models."""

import argparse
import json
from pathlib import Path
import shutil
import string
import zipfile

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from PIL import Image


INK, MUTED, RULE = '#192E32', '#4B6269', '#D9E3E3'
TEAL, BLUE, OCHRE, PLUM = '#007C76', '#366CA1', '#B26020', '#865679'
SOFT = '#B3C6C7'
CULTIVARS = {'Beefsteak': ('BFS', TEAL, 'o'), 'Big Boy Hybrid': ('BBH', OCHRE, '^'),
             'Celebrity Hybrid': ('CH', BLUE, 's')}
FRONT = Path('02_RAP_FSAM3_FRONTEND')
REP = FRONT/'Evaluation_Data/Representative4/visual_assets'
ABL = FRONT/'Ablations/Recovered_RAP_Chain/metrics'
PANEL = Path('08_PAPER_FIGURE_MATERIALS')
POTTED = Path('07_PHENOTYPING_EVALUATION/Potted_19/Per_Sample')
MASTER = Path('11_PORTABLE_RESEARCH_PACK/Research_Tables')
CODE = Path('10_REPRODUCIBILITY/Source_Code_References')
CASE = PANEL/'Tomato_Application/BFS_06262025'
CROP = CASE/'Leaf_Inset_Compound_006'
CHORD, WIDTH, AREA = 'base_to_tip_chord', 'midsection_transverse_width', 'observed_one_sided_leaf_area'
TRAIT = {CHORD: 'Chord', WIDTH: 'Width', AREA: 'Area'}

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.4, 'axes.labelsize': 6.4,
    'axes.titlesize': 7.8, 'axes.titleweight': 'medium', 'axes.titlelocation': 'left',
    'axes.edgecolor': SOFT, 'axes.linewidth': .55, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'legend.fontsize': 5.8, 'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


class Set:
    def __init__(self, root, out, dpi):
        self.root, self.out, self.dpi = root, out, dpi
        self.index, self.sources = [], []

    def start(self, key, title, scope, panel_names, claim, *, shape=(3, 2), size=(183, 165), ratios=None):
        self.key = key
        self.index.append(dict(id=key, title=title, scope=scope, panels=panel_names, claim=claim,
                               file=f'{key}.pdf'))
        fig = plt.figure(figsize=tuple(v/25.4 for v in size))
        fig.text(.058, .965, title, size=14.4, weight='bold', va='top')
        fig.text(.058, .918, scope, size=7.0, color=MUTED, va='top')
        fig.add_artist(Line2D([.058, .965], [.884, .884], transform=fig.transFigure, color=RULE, lw=.95))
        fig.text(.965, .965, key.split('_', 1)[0], va='top', ha='right', size=8, color=TEAL, weight='bold')
        if not shape:
            return fig
        grid = fig.add_gridspec(*shape, left=.105, right=.955, bottom=.09, top=.82, wspace=.50,
                               hspace=.90, height_ratios=ratios)
        axes = [fig.add_subplot(cell) for cell in grid]
        for i, (a, name) in enumerate(zip(axes, panel_names, strict=True)):
            a.set_title(f'{string.ascii_lowercase[i]}   {name}', pad=5.5, loc='left')
            a.set_axisbelow(True)
        return fig, axes

    def source(self, path, panel, role='evidence'):
        path = Path(path)
        if not (self.root/path).is_file():
            raise FileNotFoundError(path)
        self.sources.append(dict(figure=self.key, panel=panel, role=role, path=path.as_posix()))
        return self.root/path

    def csv(self, path, panel, columns=None, **kwargs):
        return pd.read_csv(self.source(path, panel, 'numeric inputs'), usecols=columns, **kwargs)

    def table(self, name, panel, columns=None, **kwargs):
        return self.csv(MASTER/f'{name}.csv', panel, columns, **kwargs)

    def dump(self, data, name):
        folder = self.out/'Source_Data'/self.key
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/f'{name}.csv'
        data.to_csv(path, index=False)
        self.sources.append(dict(figure=self.key, panel='see panel map', role='plotted source data',
                                 path=str(path.relative_to(self.out))))

    def finish(self, fig, note):
        fig.text(.058, .022, note, size=5.7, color=MUTED, va='bottom')
        self.index[-1]['note'] = note
        for fmt in ('png', 'pdf', 'svg'):
            fig.savefig(self.out/f'{self.key}.{fmt}', dpi=self.dpi, metadata={'Creator': 'ForeSplat evidence'}
                        if fmt != 'png' else {'Software': 'ForeSplat evidence'})
        plt.close(fig)
        print(self.key, flush=True)


def grid(a, axis='y'):
    a.grid(axis=axis, color=RULE, lw=.55, alpha=.8)


def cat(a, pos, labels, *, axis='x', rotate=0):
    (a.set_xticks if axis == 'x' else a.set_yticks)(pos, labels)
    if axis == 'x' and rotate:
        plt.setp(a.get_xticklabels(), rotation=rotate, ha='right', rotation_mode='anchor')


def simple_bar(a, labels, values, colors=TEAL, unit='', decimals=0):
    values = np.array(values, dtype=float)
    pos = np.arange(len(labels))
    a.barh(pos, values, color=colors, height=.58, zorder=2)
    cat(a, pos, labels, axis='y')
    a.invert_yaxis()
    a.set_xlabel(unit)
    grid(a, 'x')
    a.margins(x=.20)
    high = np.nanmax(values)
    for p, v in zip(pos, values):
        if np.isfinite(v):
            a.text(v + high*.022, p, f'{v:,.{decimals}f}', va='center', size=5.5)


def dotsets(a, data, group, key, order, colors, ylabel, *, limits=None, rotate=0):
    for i, (g, color) in enumerate(zip(order, colors, strict=True)):
        y = data.loc[data[group].eq(g), key].dropna().to_numpy()
        if not len(y):
            continue
        jitter = np.random.default_rng(i+617).uniform(-.22, .22, len(y))
        a.scatter(i+jitter, y, s=10, color=color, alpha=.66, edgecolors='white', linewidths=.20, zorder=2)
        a.hlines(np.median(y), i-.30, i+.30, color=INK, lw=1.15, zorder=3)
    cat(a, range(len(order)), order, rotate=rotate)
    a.set_xlim(-.55, len(order)-.45)
    a.set_ylabel(ylabel)
    if limits:
        a.set_ylim(*limits)
    grid(a)


def heat(a, mat, rows, cols, vlim=(0, 1), *, palette='YlGnBu', texts=None, rotate=0):
    data = np.asarray(mat)
    image = a.imshow(data, vmin=vlim[0], vmax=vlim[1], cmap=palette, aspect='auto', interpolation='none')
    cat(a, range(len(cols)), cols, rotate=rotate)
    cat(a, range(len(rows)), rows, axis='y')
    for i, j in np.ndindex(data.shape):
        if np.isfinite(data[i, j]) and (texts is not None or data.size <= 60):
            rgba = image.cmap(image.norm(data[i, j]))
            fg = 'white' if np.dot(rgba[:3], [.21, .72, .07]) < .49 else INK
            text = str(texts[i, j]) if texts is not None else f'{data[i, j]:.2f}'
            a.text(j, i, text, ha='center', va='center', color=fg, size=5.1)
    a.spines[['left', 'bottom']].set_visible(False)
    return image


def colors_cult(data, col):
    return data[col].map({k: v[1] for k,v in CULTIVARS.items()})


def scatter_cult(a, d, x, y, *, xlab=None, ylab=None, s=16, alpha=.83):
    for label, (short, color, marker) in CULTIVARS.items():
        block = d[d.cultivar.eq(label)]
        a.scatter(block[x], block[y], color=color, marker=marker, s=s, alpha=alpha,
                  linewidths=.28, edgecolors='white', label=short, zorder=3)
    a.set(xlabel=xlab or x, ylabel=ylab or y)
    grid(a)


def legend_cult(a, loc='best'):
    a.legend(ncols=3, loc=loc, columnspacing=.65, handletextpad=.18, borderpad=.2)


def identity(a, points):
    m = max(0.2, float(np.nanmax(points)))*1.07
    a.plot([0,m],[0,m], '--', lw=.65, color=MUTED, zorder=0)
    a.set(xlim=(0,m), ylim=(0,m))


def front_performance(p):
    fig, a = p.start('A01_Frontend_benchmark', 'Front-end mask benchmark',
        'Representative4: 4 plants, 20 annotated views | evaluation protocols differ by method',
        ['Foreground F1', 'Mean IoU', 'Archived boundary F1', '95th-percentile Hausdorff distance',
         'F1 by plant and method', 'Same-view RAP minus single-prompt SAM3'],
        'Distributions and paired differences characterize mask performance on the archived Representative4 benchmark.')
    d = p.csv(REP/'source_data/frame_metrics.csv', 'a–f', ['method','sample','frame','f1','miou','boundary_f1','hd95_px'])
    info = p.csv(REP/'source_data/paper_summary_metrics.csv', 'a–f', ['method','method_dir','protocol','eval_frames'])
    short = ['SEEM','CLIP','G-S1','G-S2','S2*','S3-P2','RAP','U-Net','DLv3+','SAM†']
    d = d.merge(info, left_on='method', right_on='method_dir', suffixes=('_key',''))
    d['short'] = d.method_dir.map(dict(zip(info.method_dir, short)))
    cols = [TEAL if k=='RAP' else BLUE if k=='S3-P2' else SOFT for k in short]
    for ax, metric, label in zip(a, ['f1','miou','boundary_f1','hd95_px'], ['F1','mIoU','BF1','HD95 (px)']):
        dotsets(ax, d, 'short', metric, short, cols, label, rotate=35)
    a[3].set_yscale('log')
    mat = d.pivot_table(index='sample',columns='short',values='f1').reindex(columns=short)
    heat(a[4],mat,mat.index,short,(.45,1),rotate=35)
    pair = d[d.short.isin(['RAP','S3-P2'])].pivot(index=['sample','frame'],columns='short',values=['f1','boundary_f1']).dropna()
    pair = pair.xs('RAP', level=1, axis=1)-pair.xs('S3-P2', level=1, axis=1)
    for i,(sample, block) in enumerate(pair.groupby(level=0)):
        a[5].scatter(block.f1, block.boundary_f1, marker='osp^'[i], color=[TEAL,BLUE,OCHRE,PLUM][i], s=19, label=sample)
    a[5].axvline(0,color=RULE,lw=.75); a[5].axhline(0,color=RULE,lw=.75)
    a[5].set(xlabel='Δ F1',ylabel='Δ BF1'); a[5].legend(ncols=2, loc='best', fontsize=5.0)
    p.dump(d[['sample','frame','method','short','protocol','eval_frames','f1','miou','boundary_f1','hd95_px']], 'annotated_frames')
    p.dump(pair.reset_index(), 'same_view_deltas')
    p.finish(fig, 'Dots: annotated views; dark marks: medians. * Oracle boxes; † historical SAM output (19 views).')


def front_ablation(p):
    fig,a = p.start('A02_Frontend_ablation', 'Prompt selection and refinement ablations',
        'Recovered RAP chain v3: 16 development frames from 4 plants | same frames across rungs',
        ['F1 with potted-plant target', 'Boundary F1 at three tolerances', 'Potted-target F1 by plant',
         'F1 under three annotation conventions', 'Change from single-P2 baseline', 'Execution and acceptance counts'],
        'Rung, annotation convention and scoring tolerance all affect interpretation on the development frames.')
    keys = ['R0s_single_P2_base','R1s_single_P2_cdr','R3n_multi_nogate_cdr','R2_multi_SGMPS_base',
            'R3_multi_SGMPS_cdr','R4b_R3_SPNP_boxes','R4p_R3_SPNP_points','R5_R4p_repair','R6_R5_GCFC']
    labels = ['P2','P2+CDR','Multi+CDR','Gate','Gate+CDR','Box','Points','Repair','GCFC']
    d = p.csv(ABL/'dev_ablation_per_frame_v3.csv', 'a–e', ['scene','config','frame','gt_convention','f1',
            'boundary_f1@2','boundary_f1@5','boundary_f1@10'])
    d = d[d.config.isin(keys)].copy(); d['rung'] = d.config.map(dict(zip(keys,labels)))
    d = d.rename(columns={'boundary_f1@2':'BF2','boundary_f1@5':'BF5','boundary_f1@10':'BF10'})
    plant = d[d.gt_convention.eq('potted_plant')]; col = [SOFT]+[TEAL]*(len(labels)-1)
    dotsets(a[0],plant,'rung','f1',labels,col,'F1',rotate=30)
    for k,c,mark in [('BF2',OCHRE,'^'),('BF5',TEAL,'o'),('BF10',BLUE,'s')]:
        y = plant.groupby('rung')[k].mean().reindex(labels)
        a[1].plot(range(len(labels)),y,marker=mark,color=c,lw=.8,ms=2.6,label=k[-2:].strip('F')+' px')
    cat(a[1],range(len(labels)),labels,rotate=30); grid(a[1]); a[1].set_ylabel('Mean BF1'); a[1].legend(ncols=3, loc='center left')
    matrix = plant.pivot_table(index='scene',columns='rung',values='f1').reindex(columns=labels)
    heat(a[2],matrix,matrix.index,labels,(.9,1),rotate=30)
    targets = d.pivot_table(index='gt_convention',columns='rung',values='f1').reindex(index=['potted_plant','plant_only','pot'],columns=labels)
    heat(a[3],targets,['With pot','Plant only','Pot only'],labels,(.40,1),rotate=30)
    paired = plant.pivot(index=['scene','frame'],columns='rung',values='f1').reindex(columns=labels).dropna()
    delta = paired.subtract(paired['P2'],axis='index').reset_index().melt(id_vars=['scene','frame'],var_name='rung',value_name='delta_f1')
    dotsets(a[4],delta,'rung','delta_f1',labels,col,'Δ F1 (same frame)',rotate=30)
    a[4].axhline(0,color=MUTED,lw=.7,ls='--')
    log = p.csv(ABL/'dev_ablation_stage_counts_v3.csv','f'); cols = ['sg_mps_executed','cdr_executed','sb_fbpr_accepted','gcfc_accepted']
    tab = log.groupby('config')[cols].sum().reindex(keys).T
    heat(a[5],tab,['Gate run','CDR run','SB accept','GC accept'],labels,(0,16),texts=tab.to_numpy(),rotate=30)
    p.dump(d,'frame_rungs'); p.dump(delta,'paired_f1_deltas'); p.dump(tab.T.reset_index(),'stage_counts')
    p.finish(fig, 'Selected rungs only; development descriptors, not an independent estimate of module gains. BF thresholds in pixels.')


def gaussian(p):
    order = ['3DGS','2DGS','GGGS','gsplat','MCMC-3DGS']; colors = [SOFT, BLUE, TEAL, SOFT, PLUM]
    fig,a = p.start('A03_Gaussian_selection', 'Five Gaussian methods on a shared scene',
        'KongQueZhuYu only | same COLMAP cameras | 183 train views, 27 test views | 30,000 iterations',
        ['Full-frame rendering', 'Foreground rendering', 'Structural agreement', 'Perceptual distance',
         'Training time and Gaussian count', 'Native RGB vs. attribute rendering'],
        'The single-plant experiment separates view-level appearance from training and rendering cost.')
    cols = ['name','psnr','foreground_psnr','ssim','lpips']
    logs = []
    for method in order:
        src = p.source(Path('03_COLMAP_GGGS_RECONSTRUCTION/Render_Quality')/method.replace('-','_')/'per_view.json','a–d','per-view metrics')
        logs.append(pd.DataFrame(json.loads(src.read_text()))[cols].assign(method=method))
    d = pd.concat(logs,ignore_index=True)
    assert d.groupby('method').name.nunique().eq(27).all() and d.groupby('name').method.nunique().eq(5).all()
    for ax,key,label in zip(a,cols[1:],['PSNR (dB)','Foreground PSNR (dB)','SSIM','LPIPS']):
        dotsets(ax,d,'method',key,order,colors,label)
    cols = ['method','training_wall_s','gaussian_count','native_rgb_fps','rgb_alpha_depth_fps']
    sums = p.table('Gaussian_Method_Comparison','e–f', cols).replace({'MCMC_3DGS':'MCMC-3DGS'}).set_index('method').reindex(order).reset_index()
    a[4].scatter(sums.training_wall_s/60,sums.gaussian_count/1e6,c=colors,s=26,zorder=3,edgecolors=INK,linewidths=.35)
    label_offsets = {'3DGS': (4, 4), '2DGS': (4, 4), 'GGGS': (4, -9), 'gsplat': (4, 4), 'MCMC-3DGS': (-48, 4)}
    for r in sums.itertuples():
        a[4].annotate(r.method,(r.training_wall_s/60,r.gaussian_count/1e6),xytext=label_offsets[r.method],textcoords='offset points',size=5.6)
    a[4].margins(x=.15,y=.22); a[4].set(xlabel='Training wall time (min)',ylabel='Gaussian count (million)'); grid(a[4]);
    for i,r in enumerate(sums.itertuples()):
        a[5].plot([r.rgb_alpha_depth_fps,r.native_rgb_fps],[i,i],lw=1,color=SOFT)
    x=np.arange(len(order)); a[5].scatter(sums.native_rgb_fps,x,c=TEAL,label='RGB only',s=14)
    a[5].scatter(sums.rgb_alpha_depth_fps,x,c=OCHRE,label='RGB + attributes',s=14,marker='s')
    cat(a[5],x,order,axis='y'); a[5].set_ylim(5.05,-.65); a[5].set_xlabel('Reported rendering throughput (FPS)'); grid(a[5],'x')
    a[5].legend(loc='lower right',ncols=2,fontsize=5.2)
    p.dump(d,'per_view_metrics'); p.dump(sums,'resources_and_throughput')
    p.finish(fig,'Dots: test cameras; dark marks: medians. Foreground uses the archived RAP mask. Throughput is protocol-specific.')


def mesh_selection(p):
    fig,a = p.start('A04_Mesh_selection', 'Mesh extraction: quality versus complexity',
        'KongQueZhuYu only | 7 extractor/renderer combinations | raw and archived postprocessing',
        ['Raw and post silhouette overlap', 'Post mesh size versus overlap', 'Postprocessing and connectivity',
         'Foreground coverage versus precision', 'Post mask-boundary correspondence', 'Extraction wall time'],
        'Face count, mask correspondence, connectivity and extraction cost should be considered together.')
    cols = ['model','method','stage','faces','components','silhouette_iou','largest_component_face_fraction',
            'foreground_coverage','foreground_precision','boundary_fscore_2px','extraction_wall_s']
    d = p.table('Mesh_Method_Comparison','a–f',cols)
    d['label'] = d.model+' '+d.method.replace({'Poisson':'Pois.','native':'native'})
    order = ['GGGS TSDF','2DGS TSDF','GGGS NKSR','2DGS NKSR','GGGS Pois.','2DGS Pois.','GGGS native']
    idx = np.arange(len(order)); raw = d[d.stage.eq('raw')].set_index('label').reindex(order)
    post = d[d.stage.eq('post')].set_index('label').reindex(order)
    colors = [TEAL, BLUE, SOFT, SOFT, SOFT, SOFT, PLUM]
    for ax,k,xlab in [(a[0],'silhouette_iou','Silhouette IoU (RAP proxy)'),(a[2],'largest_component_face_fraction','Fraction of faces in largest component')]:
        ax.hlines(idx,raw[k],post[k],color=SOFT,lw=1.05); ax.scatter(raw[k],idx,facecolors='white',edgecolors=MUTED,s=15,label='Raw',zorder=3)
        ax.scatter(post[k],idx,c=colors,s=16,label='Post',zorder=4); cat(ax,idx,order,axis='y'); ax.invert_yaxis()
        grid(ax,'x'); ax.set_xlabel(xlab)
    a[0].legend(loc='lower left',ncols=2,fontsize=5.2)
    for ax, x, y, xl, yl in [(a[1],'faces','silhouette_iou','Faces','Silhouette IoU'),
                              (a[3],'foreground_coverage','foreground_precision','Foreground coverage','Foreground precision')]:
        ax.scatter(post[x],post[y],c=colors,s=22,edgecolors=INK,lw=.3,zorder=3)
        if x=='faces': ax.set_xscale('log')
        for label, off in [('GGGS TSDF',(4,3)),('GGGS native',(4,-8)),('2DGS TSDF',(4,-8))]:
            ax.annotate(label,(post.loc[label,x],post.loc[label,y]),xytext=off,textcoords='offset points',size=5.3)
        ax.set(xlabel=xl,ylabel=yl); ax.margins(x=.24,y=.16); grid(ax)
    simple_bar(a[4],order,post.boundary_fscore_2px,colors,'BF1, tolerance 2 px',2)
    simple_bar(a[5],order,post.extraction_wall_s/60,colors,'Reported wall time (min)',1)
    p.dump(d,'mesh_raw_and_post')
    p.finish(fig,'Overlap and boundary scores use the archived RAP silhouette as a proxy, not independent 3D ground truth.')


def reconstruction(p):
    fig,a = p.start('A05_Reconstruction_coverage', 'Reconstruction and metric coverage',
        'Reconstruction: 19 potted plants and 42 TomatoPGT scans | Metric coverage: 19 tomato scans',
        ['Successes out of each input cohort','COLMAP registered fraction','Processing state per tomato scan',
         'Tomato sparse geometry quality','Plant-only mesh face counts','Metric mesh coverage'],
        'The full cohorts support reconstruction evaluation; panel f reports metric meshes within the measurement cohorts.')
    t = p.table('Tomato_42_Reconstruction','a–e',['scene','cultivar','queue_status','rgb','independent_colmap',
          'sparse_points','mean_sparse_error_working_px','gggs_30k','plant_faces','capture_date_iso','tsdf_mesh_archive'])
    h = p.table('Potted_19_Reconstruction','a,b,e,f',['plant_id','group','true_colmap','raw_views','registered_views',
          'gggs_30k','bounded_tsdf','metric_mesh','plant_only_faces'])
    metric = p.table('Tomato_Metric_19_Reconstruction','f',['sample_id','metric_mesh'])
    pots_ok = [h.true_colmap.sum(), h.gggs_30k.sum(), h.bounded_tsdf.sum()]
    t['tsdf_available'] = t.tsdf_mesh_archive.map(lambda path:isinstance(path,str) and (p.root/path).is_file())
    tom_ok = [t.independent_colmap.gt(0).sum(), t.gggs_30k.sum(), t.tsdf_available.sum()]
    x = np.arange(3)
    for counts,n,off,c,l in [(pots_ok,len(h),-.17,TEAL,'Potted'),(tom_ok,len(t),.17,BLUE,'Tomato')]:
        a[0].bar(x+off,np.asarray(counts)/n,width=.30,color=c,label=l)
        for i,k in enumerate(counts): a[0].text(i+off,k/n+.026,f'{k}/{n}',ha='center',size=5.5)
    cat(a[0],x,['COLMAP','GGGS','TSDF']); a[0].set_ylim(0,1.28); a[0].set_ylabel('Fraction of inputs')
    a[0].legend(ncols=2,loc='upper left',borderpad=.05); grid(a[0]);
    stats = pd.concat([pd.DataFrame({'cohort':'Potted','id':h.plant_id,'pct':100*h.registered_views/h.raw_views,'faces':h.plant_only_faces}),
                       pd.DataFrame({'cohort':'Tomato','id':t.scene,'pct':100*t.independent_colmap/t.rgb,'faces':t.plant_faces})])
    dotsets(a[1], stats,'cohort','pct',['Potted','Tomato'],[TEAL,BLUE],'Registered input RGB (%)')
    sd=t.sort_values(['cultivar','capture_date_iso']).copy(); masks=np.vstack([sd.independent_colmap.gt(0),sd.gggs_30k,sd.tsdf_available])
    a[2].imshow(masks,aspect='auto',interpolation='none',cmap=ListedColormap(['#ECE7DF',TEAL]),vmin=0,vmax=1)
    cat(a[2],range(3),['COLMAP','GGGS','TSDF'],axis='y')
    n=0; midpoints=[]
    for cv, g in sd.groupby('cultivar',sort=False):
        midpoints.append((n+(len(g)-1)/2,CULTIVARS[cv][0])); n+=len(g); a[2].axvline(n-.5,color='white',lw=1.5)
    cat(a[2],[k for k,_ in midpoints],[l for _,l in midpoints]); a[2].set_xlabel('Chronological within cultivar')
    scatter_cult(a[3],t,'sparse_points','mean_sparse_error_working_px',xlab='Sparse points',ylab='Mean reprojection error (working px)'); legend_cult(a[3],loc='upper right')
    dotsets(a[4],stats,'cohort','faces',['Potted','Tomato'],[TEAL,BLUE],'Plant-only triangle faces'); a[4].set_yscale('log')
    groups = [('Potted',h,TEAL),('Tomato subset',metric,BLUE)]
    for i,(_, cohort, color) in enumerate(groups):
        yes = cohort.metric_mesh.sum()
        a[5].barh(i,len(cohort),color=RULE,height=.50); a[5].barh(i,yes,color=color,height=.50)
        a[5].text(len(cohort)+.35,i,f'{yes}/{len(cohort)}',va='center',size=6)
    cat(a[5],range(len(groups)),[name for name,_,_ in groups],axis='y'); a[5].invert_yaxis()
    a[5].set(xlim=(0,max(len(h),len(metric))*1.18),xlabel='Samples in metric evaluation'); grid(a[5],'x')
    p.dump(h,'potted_reconstruction'); p.dump(t,'tomato_reconstruction'); p.dump(stats,'aligned_cohort_descriptors')
    p.dump(metric,'tomato_metric_meshes')
    p.finish(fig,'Mesh face count measures asset complexity, not reconstruction fidelity. Tomato columns represent scans, not independent plants.')


def scale(p):
    fig,a = p.start('A06_Scale_calibration', 'Metric calibration and quality control',
        'Potted: known 2 cm cube | TomatoPGT: planar target baselines, 19/19 in the metric subset',
        ['Potted calibration routes','Potted cube fit residuals','Potted fitted edge versus factor',
         'Tomato metric scans by cultivar','Tomato board observation support','Tomato scale factors and baseline QC'],
        'Scale and target quality control are reported for the defined metric cohorts.')
    d = p.table('Potted_19_Calibration','a–c',['sample_id','method','scale_cm_per_unit','fitted_cube_edge_units',
        'retained_view_count','cube_reprojection_error_px','boundary_rms_px'])
    d['route'] = np.where(d.method.str.startswith('AUTOMATIC'),'SAM3 cube masks','Historical assistance')
    cohort = p.table('Tomato_Metric_19_Reconstruction','d–f',['sample_id'])
    calibration = p.table('Tomato_42_Calibration','d–f',['sample_id','cultivar_or_type','metric_mesh','calibration_view_ids',
        'scale_cm_per_unit','max_relative_baseline_scale_deviation','status']).rename(columns={'cultivar_or_type':'cultivar'})
    t = cohort.merge(calibration,on='sample_id',how='left',validate='one_to_one')
    if not (t.metric_mesh.eq(True) & np.isfinite(t.scale_cm_per_unit) & t.scale_cm_per_unit.gt(0)).all():
        raise ValueError('Missing accepted scale in the TomatoPGT metric subset')
    potted_palette={'SAM3 cube masks':TEAL, 'Historical assistance':OCHRE}
    labels=list(potted_palette)
    simple_bar(a[0],['SAM3 masks','History-aided'],d.route.value_counts().reindex(labels),
               list(potted_palette.values()),'Potted plants (19 calibrated)')
    for label,color in potted_palette.items():
        block=d[d.route.eq(label)]
        a[1].scatter(block.retained_view_count, block.cube_reprojection_error_px, color=color, s=18,
                     marker='o', label='Cube reprojection' if color==TEAL else None)
        a[1].scatter(block.retained_view_count, block.boundary_rms_px, facecolors='none',edgecolors=color, s=18,
                     marker='s', label='Boundary RMS' if color==TEAL else None)
    a[1].legend(loc='lower center',ncols=2,fontsize=5.1); a[1].set(xlabel='Retained views',ylabel='Fit residual (px)',ylim=(.4,1.57)); grid(a[1])
    edge = np.linspace(d.fitted_cube_edge_units.min()*.92, d.fitted_cube_edge_units.max()*1.06,100)
    a[2].plot(edge,2/edge,color=MUTED,ls='--',lw=.75,label='2 cm / fitted edge')
    a[2].scatter(d.fitted_cube_edge_units,d.scale_cm_per_unit,c=d.route.map(potted_palette),s=21,zorder=3)
    a[2].set(xlabel='Fitted cube edge (SfM units)',ylabel='Scale (cm per SfM unit)'); a[2].legend(); grid(a[2]);
    short = [v[0] for v in CULTIVARS.values()]
    colors = [v[1] for v in CULTIVARS.values()]
    counts = t.cultivar.value_counts().reindex(CULTIVARS.keys(),fill_value=0)
    simple_bar(a[3],short,counts,colors,'Calibrated scans in metric subset')
    t['cultivar_code'] = t.cultivar.map({k:v[0] for k,v in CULTIVARS.items()})
    t['target_support_views'] = t.calibration_view_ids.fillna('').map(lambda ids:len(set(ids.split())))
    dotsets(a[4],t,'cultivar_code','target_support_views',short,colors,'Distinct target-support views')
    t['baseline_deviation_pct'] = t.max_relative_baseline_scale_deviation*100
    good = t.dropna(subset=['baseline_deviation_pct'])
    scatter_cult(a[5],good,'scale_cm_per_unit','baseline_deviation_pct',xlab='Scale (cm per SfM unit)',ylab='Maximum baseline deviation (%)')
    legend_cult(a[5],loc='upper right')
    p.dump(d,'potted_calibration'); p.dump(t,'tomato_calibration')
    p.finish(fig,'Tomato panels use the metric subset throughout. SfM units are arbitrary per capture; factor size is not an accuracy metric.')


def potted_leaves(p):
    fig,a = p.start('A07_Potted_leaf_instances', 'Potted plant leaves: instances to measurements',
        '19 potted plants | 798 stable visible candidates | 57 human-confirmed handwritten leaf identities',
        ['Stable visible candidates per plant','Fragmentation among candidates','Available dimensions on tagged leaves',
         'Observed leaf widths by plant type','Observed leaf areas by plant type','Length versus width where both exist'],
        'The manually confirmed identity subset has uneven trait availability and no independent paired manual values.')
    h = p.table('Potted_19_Reconstruction','a–b',['plant_id','plant_type','group','stable_leaf_count','fragmented_leaf_candidates','plant_only_faces'])
    d = p.table('Potted_57_Trait_Measurements','c–f',['plant_id','handwritten_number','leaf_id','human_confirmed',
        'manual_leaf_length_cm','manual_leaf_width_cm','automatic_leaf_length_cm','automatic_leaf_width_cm','automatic_leaf_area_cm2','measurement_status'])
    assert len(d)==57 and d.human_confirmed.all() and h.stable_leaf_count.sum()==798
    h['scope']=h.group.map({'development':'Dev.','heldout_validation':'Holdout'})
    h=h.sort_values('stable_leaf_count'); codes={k:k for k in h.plant_type.unique()}
    codes.update({'KongQueZhuYu':'KQ','ChangShouHua':'CSH','XiangPiShu':'XPS','CaoMei':'CM','DouBanLv':'DBL',
                 'HongZhang':'HZ','WanNianQing':'WNQ','WangWenCao':'WWC','XianKeLai':'XKL'})
    h['plant_short'] = [x.replace(y,codes[y]) for x,y in zip(h.plant_id,h.plant_type)]
    a[0].bar(np.arange(len(h)),h.stable_leaf_count,color=h.scope.map({'Dev.':OCHRE,'Holdout':TEAL}),width=.65)
    cat(a[0],range(len(h)),h.plant_short,rotate=52); a[0].set_ylabel('Visible candidates'); grid(a[0])
    a[0].scatter([],[],c=OCHRE,marker='s',label='Development'); a[0].scatter([],[],c=TEAL,marker='s',label='Holdout'); a[0].legend(loc='upper left')
    for typ,c,mark in [('Dev.',OCHRE,'s'),('Holdout',TEAL,'o')]:
        x=h[h.scope.eq(typ)]; a[1].scatter(x.stable_leaf_count,x.fragmented_leaf_candidates,color=c,marker=mark,s=22)
    a[1].set(xlabel='Stable visible candidates',ylabel='Marked fragmented'); grid(a[1]);
    d = d.merge(h[['plant_id','plant_type','scope']],on='plant_id',validate='many_to_one')
    u = ['automatic_leaf_length_cm','automatic_leaf_width_cm','automatic_leaf_area_cm2']
    vals = [d[k].count() for k in u]; catlabs = ['Length candidate','Width candidate','Observed area','Manual pairs']
    for i,v in enumerate(vals+[0]):
        a[2].barh(i,57,color=RULE,height=.62); a[2].barh(i,v,color=[TEAL,TEAL,TEAL,SOFT][i],height=.62)
        a[2].text(58.5,i,f'{v}/57',va='center',size=6)
    cat(a[2],range(4),catlabs,axis='y'); a[2].invert_yaxis(); a[2].set(xlim=(0,68),xlabel='Tagged leaves'); grid(a[2],'x')
    order=sorted(h.plant_type.unique()); d['type_code']=d.plant_type.map(codes); type_codes=[codes[k] for k in order]
    dotsets(a[3],d,'type_code',u[1],type_codes,[TEAL]*len(order),'Automatic width (cm)',rotate=30)
    dotsets(a[4],d,'type_code',u[2],type_codes,[BLUE]*len(order),'Observed one-sided area (cm²)',rotate=30); a[4].set_yscale('log')
    both=d.dropna(subset=u[:2]); a[5].scatter(both[u[0]],both[u[1]],c=both.scope.map({'Dev.':OCHRE,'Holdout':TEAL}),s=17,alpha=.8)
    a[5].set(xlabel='Automatic length candidate (cm)',ylabel='Automatic width (cm)'); grid(a[5]);
    a[5].text(.98,.05,f'n = {len(both)} leaves',transform=a[5].transAxes,ha='right',color=MUTED)
    p.dump(h,'per_plant_candidates'); p.dump(d,'57_tagged_leaves')
    p.finish(fig,'Tagged identity confirms correspondence, not 3D segmentation accuracy. Potted length candidates are preliminary surface paths.')


def tomato_correspondence(p):
    fig,a = p.start('A08_Tomato_compound_mapping', 'Tomato compound-leaf correspondence',
        '42 scans, 3 cultivars, 245 official compound instances | 70 official instances in metric scans',
        ['Metric qualification over daily captures','Predicted compound candidates per scan','Match status by scale group',
         'Reference versus prediction coverage','Match score by assignment status','Official records and assigned models'],
        'Assignment status and coverage characterize correspondence without treating the matching score as segmentation accuracy.')
    t = p.table('Tomato_42_Reconstruction','a,b',['scene','cultivar','capture_date_iso','metric_mesh','compound_candidates',
        'leaf_candidates','official_compounds_from_point_labels'])
    d = p.table('Tomato_245_Compound_Instances','c–f',['scene','cultivar','metric_scan','official_point_instance','candidate_id','status',
        'overlap_score','reference_coverage','prediction_coverage','runner_up_gap'])
    t['date']=pd.to_datetime(t.capture_date_iso)
    for i,(cv,(short,c,mark)) in enumerate(CULTIVARS.items()):
        x=t[t.cultivar.eq(cv)]; yes=x[x.metric_mesh]; no=x[~x.metric_mesh]
        a[0].scatter(no.date,[i]*len(no),s=17,marker=mark,facecolors='white',edgecolors=c,lw=.7)
        a[0].scatter(yes.date,[i]*len(yes),s=17,marker=mark,color=c)
    cat(a[0],range(3),[v[0] for v in CULTIVARS.values()],axis='y'); a[0].set_ylim(-.55,2.55); a[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2)); a[0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    a[0].set_xlabel('Acquisition date (2025; filled = metric)'); grid(a[0],'x')
    ref=t[t.official_compounds_from_point_labels.gt(0)]
    scatter_cult(a[1],ref,'official_compounds_from_point_labels','compound_candidates',xlab='Official compound instances',ylab='Predicted compound candidates'); grid(a[1]); legend_cult(a[1],loc='lower right')
    order=['strong_suggestion','review_suggestion','ambiguous_assignment','no_reliable_match','no_candidate']
    labels=['Strong','Review','Ambiguous','No reliable','No candidate']; colors=[TEAL,BLUE,PLUM,OCHRE,'#B9AD9E']
    data = pd.crosstab(d.metric_scan,d.status).reindex(index=[True,False],columns=order,fill_value=0)
    pos = np.arange(2); left=np.zeros(2)
    for i,k in enumerate(order):
        v=data[k].to_numpy(); a[2].barh(pos,v,left=left,color=colors[i],height=.52,label=labels[i]);
        for j,cnt in enumerate(v):
            if cnt>8: a[2].text(left[j]+cnt/2,j,str(cnt),ha='center',va='center',size=5.5,color='white' if i<4 else INK)
        left+=v
    cat(a[2],pos,['Metric (70)','SfM units (175)'],axis='y'); a[2].set_ylim(2.0,-.55); a[2].set_xlabel('Official compound instances'); a[2].legend(loc='lower left',ncols=3,fontsize=4.9,handlelength=.9,columnspacing=.8); grid(a[2],'x')
    b=d.dropna(subset=['reference_coverage','prediction_coverage']);
    for k,lab,c in zip(order,labels,colors):
        z=b[b.status.eq(k)]; a[3].scatter(z.reference_coverage,z.prediction_coverage,c=c,s=13,alpha=.68,label=lab)
    a[3].set(xlim=(0,1.04),ylim=(0,1.04),xlabel='Reference covered fraction',ylabel='Prediction covered fraction'); grid(a[3]);
    d['match_label']=d.status.map(dict(zip(order,labels)))
    dotsets(a[4],d,'match_label','overlap_score',labels[:3],colors[:3],'Distance-threshold match score')
    for i,(cv,(lab,c,_)) in enumerate(CULTIVARS.items()):
        b=d[d.cultivar.eq(cv)]; n=b.candidate_id.notna().sum(); a[5].barh(i,len(b),color=RULE,height=.55);a[5].barh(i,n,color=c,height=.55)
        a[5].text(len(b)+1.5,i,f'{n}/{len(b)}',va='center',size=6)
    cat(a[5],range(3),[v[0] for v in CULTIVARS.values()],axis='y'); a[5].set(xlabel='Compound records (filled = has candidate)',xlim=(0,d.cultivar.value_counts().max()+25)); a[5].invert_yaxis(); grid(a[5],'x')
    p.dump(t,'scan_dates_and_candidates'); p.dump(d,'official_instance_matches'); p.dump(data.reset_index(),'status_counts')
    p.finish(fig,'Panel b includes only scans with positive official instance counts from point labels. Match score is not standard 3D instance IoU.')


def tomato_agreement(p):
    fig,a = p.start('A09_Tomato_leaf_agreement', 'Agreement of matched tomato leaf landmarks',
        'Scale-qualified compound leaves | official reference guides mesh landmark position | scan-clustered observations',
        ['Base-to-tip chord: model versus reference','Midsection width: model versus reference','Chord residual versus reference',
         'Width residual versus reference','Effective sample size across stages','Reasons for excluded trait pairs'],
        'The included subset supports reference-guided measurement agreement, not end-to-end automatic accuracy.')
    d = p.table('Tomato_Leaf_Trait_Pairs','a–f',['scene','cultivar','official_point_instance','candidate_id','trait','status','unit','model','reference','error_model_minus_reference'])
    stats = p.table('Tomato_Leaf_Trait_Errors','a–e',['trait','n_pairs','n_scans','mae','rmse','mean_bias','r2_against_identity']).set_index('trait')
    kept=d[d.status.eq('included_reference_guided_landmarks')]; assert len(kept)==38
    for i,k in enumerate([CHORD,WIDTH]):
        g=kept[kept.trait.eq(k)]; n,m=len(g),g.scene.nunique(); data=stats.loc[k]
        assert n==data.n_pairs and m==data.n_scans
        assert np.isclose(g.error_model_minus_reference.abs().mean(),data.mae) and np.isclose(np.mean(g.error_model_minus_reference),data.mean_bias)
        identity(a[i],g[['model','reference']]); scatter_cult(a[i],g,'reference','model',xlab='Official reference (cm)',ylab='Mesh measurement (cm)')
        a[i].text(.035,.96,f'{n} leaves / {m} scans\nMAE {data.mae:.3f} cm    RMSE {data.rmse:.3f} cm\n$R^2_{{identity}}$ {data.r2_against_identity:.3f}',
                transform=a[i].transAxes,va='top',fontsize=5.6,bbox=dict(fc='white',ec='none',alpha=.86,pad=1.0))
        ax=a[i+2]; scatter_cult(ax,g,'reference','error_model_minus_reference',xlab='Official reference (cm)',ylab='Mesh − reference (cm)')
        ax.axhline(0,color=MUTED,lw=.70,ls='--'); ax.axhline(data.mean_bias,color=OCHRE,lw=.65,ls=':')
        ax.text(.98,.07,f'Mean bias {data.mean_bias:+.3f} cm',ha='right',transform=ax.transAxes,fontsize=5.3,color=OCHRE)
    legend_cult(a[1],loc='lower right')
    status_counts=d.groupby('trait').size().reindex([CHORD,WIDTH,AREA]); pos=np.arange(3)
    a[4].barh(pos,status_counts,height=.58,color=RULE,label='Official in metric scans');a[4].barh(pos,stats.reindex([CHORD,WIDTH,AREA]).n_pairs,height=.58,color=TEAL,label='Effective pairs')
    for i,k in enumerate([CHORD,WIDTH,AREA]):a[4].text(71,i,f'{int(stats.loc[k,"n_pairs"])}/70',va='center',size=5.5)
    cat(a[4],pos,['Chord','Width','Area'],axis='y'); a[4].set(ylim=(3.0,-.6),xlim=(0,84),xlabel='Compound leaves per trait'); a[4].legend(loc='lower left',ncols=2,fontsize=5.1,handlelength=.85); grid(a[4],'x')
    reasons=['no_model_landmark_pair_passing_existing_qc','missing_reference_landmarks','scale_review','ambiguous_assignment','no_matching_mesh']
    lab=['No eligible model points','No reference points','Scale review','Ambiguous assignment','No matching mesh']
    mat = pd.crosstab(d.status,d.trait).reindex(index=reasons,columns=[CHORD,WIDTH],fill_value=0)
    heat(a[5],mat,lab,['Chord','Width'],(0,mat.to_numpy().max()),texts=mat.to_numpy());
    p.dump(d,'all_trait_rows'); p.dump(stats.reset_index(),'archived_metrics'); p.dump(mat.reset_index(),'excluded_chord_width_counts')
    p.finish(fig,'Identity line: y = x. Residual panels: dashed = zero; dotted = mean signed bias. Area lacks a comparable reference.')


def plant_traits(p):
    fig,a = p.start('A10_Canopy_and_observed_area', 'Whole-plant dimensions and observed leaf area',
        'Potted metric cohort (19 plants) and TomatoPGT metric subgroup (19 scans) | descriptive, not accuracy estimates',
        ['Potted canopy span on fixed axes','Potted visible height and maximum spread','Tomato visible height at capture',
         'Exploratory tomato height comparison','Observed area of tomato compound meshes','Tomato area record availability'],
        'The evidence supports descriptive size measures distinct from validated length and width agreement.')
    d = pd.concat([p.csv(path.relative_to(p.root),'a,b',['plant_id','group','plant_height_cm','canopy_span_a_cm',
        'canopy_span_b_cm','canopy_width_cm','measurement_status']) for path in sorted((p.root/POTTED).glob('*/plant_measurements.csv'))],ignore_index=True)
    assert d.plant_id.nunique()==19
    groups={'development':('Development',OCHRE,'s'),'heldout_validation':('Holdout',TEAL,'o')}
    for k,(label,c,mark) in groups.items():
        z=d[d.group.eq(k)]; a[0].scatter(z.canopy_span_a_cm,z.canopy_span_b_cm,c=c,marker=mark,s=22,label=label)
        a[1].scatter(z.plant_height_cm,z.canopy_width_cm,c=c,marker=mark,s=22,label=label)
    identity(a[0],d[['canopy_span_a_cm','canopy_span_b_cm']]); a[0].set(xlabel='Axis A canopy span (cm)',ylabel='Axis B canopy span (cm)'); grid(a[0]); a[0].legend(ncols=2,loc='lower right',fontsize=5.1)
    a[1].set(xlabel='Visible height candidate (cm)',ylabel='Maximum horizontal span (cm)'); grid(a[1]);
    t = p.table('Tomato_Metric_19_Reconstruction','c',['scene','cultivar','capture_date_iso','visible_height_cm'])
    t['date'] = pd.to_datetime(t.capture_date_iso)
    scatter_cult(a[2],t,'date','visible_height_cm',xlab='Acquisition date (2025)',ylab='Visible vertical span (cm)');
    a[2].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2)); a[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d')); legend_cult(a[2],loc='upper left')
    h = p.csv(Path('07_PHENOTYPING_EVALUATION/Tomato_Metric_19/Initial_visible_height_comparison.csv'),'d')
    identity(a[3],h[['official_cm','prediction_cm']]); scatter_cult(a[3],h,'official_cm','prediction_cm',xlab='Official true_height (cm)',ylab='Visible mesh span (cm)')
    a[3].text(.06,.93,f'{len(h)} scans; different definitions',transform=a[3].transAxes,color=OCHRE,va='top',size=5.7)
    area = p.table('Tomato_70_Metric_Compounds','e,f',['scene','cultivar','candidate_id','official_point_instance','model_observed_mesh_area_cm2','status'])
    area['code']=area.cultivar.map({k:v[0] for k,v in CULTIVARS.items()}); crops=list(CULTIVARS.values())
    area_key='model_observed_mesh_area_cm2'
    dotsets(a[4],area,'code',area_key,[k[0] for k in crops],[k[1] for k in crops],'Observed one-sided area (cm²)'); a[4].set_yscale('log')
    for i,(label,c,_) in enumerate(crops):
        block=area[area.code.eq(label)]; n=block[area_key].count()
        a[5].barh(i,len(block),height=.58,color=RULE); a[5].barh(i,n,height=.58,color=c)
        a[5].text(len(block)+.5,i,f'{n}/{len(block)}',va='center',size=5.8)
    cat(a[5],range(len(crops)),[k[0] for k in crops],axis='y'); a[5].set(xlim=(0,area.code.value_counts().max()+8),xlabel='Official records (filled = model area)'); a[5].invert_yaxis(); grid(a[5],'x')
    p.dump(d,'potted_visible_dimensions'); p.dump(t,'tomato_metric_height_candidates'); p.dump(h,'exploratory_height_pairs'); p.dump(area,'tomato_observed_areas')
    p.finish(fig,'Visible height and official true_height are not definition-matched. No comparable reference is available for one-sided area.')


def canvas(fig):
    ax=fig.add_axes([.056,.07,.90,.78]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    return ax


def tile(ax,x,y,w,h,title,body,accent=TEAL,bg='#F4F8F7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.004,rounding_size=.011', facecolor=bg,edgecolor=RULE,lw=.65))
    ax.plot([x+.012,x+w-.012],[y+h-.015]*2,color=accent,lw=1.1,solid_capstyle='round')
    ax.text(x+.013,y+h-.042,title,weight='bold',size=7.8,va='top')
    ax.text(x+.013,y+h-.105,body,color=MUTED,size=6.3,va='top',linespacing=1.35)


def section(ax,x,y,name):
    ax.text(x,y,name,weight='bold',size=8.2,color=TEAL,va='bottom')


def arrow(ax,x,y,tx,ty,*,color=TEAL,style='-|>'):
    ax.add_patch(FancyArrowPatch((x,y),(tx,ty),arrowstyle=style,mutation_scale=7,color=color,lw=1.05,shrinkA=0,shrinkB=0))


def img(p,fig,bbox,src,title,panel,*,crop=None):
    path=p.source(src,panel,'figure raster input' if crop is None else f'figure raster input; crop xyxy={crop}')
    ax=fig.add_axes(bbox); ax.axis('off')
    with Image.open(path) as image:
        if crop is not None: image=image.crop(crop)
        # Match page sampling to avoid embedding the full-resolution image in PDF/SVG.
        image.thumbnail((1800,1800))
        ax.imshow(image.convert('RGB'),interpolation='antialiased')
    if title: ax.set_title(title,loc='left',size=6.6,pad=3)
    return ax


def method_img(p,fig,ax,box,src,title,panel,**kwargs):
    x,y,w,h=ax.get_position().bounds; i,j,a,b=box
    return img(p,fig,[x+i*w,y+j*h,a*w,b*h],src,title,panel,**kwargs)


def mechanism_front(p):
    names=['Frame and proposal generation','Prompt choice and geometric refinement','Archived image and masks']
    fig=p.start('M01_RAP_FSAM3', 'RAP-FSAM3: reconstruction-aware masks',
                'Mechanism diagram | candidate generation, decision, refinement and output',names,
                'The front-end obtains one reconstruction mask per observed view by selecting and refining the prompt-conditioned instance.',
                shape=None,size=(183,140))
    ax=canvas(fig)
    section(ax,.008,.968,'a   Candidate generation')
    x=[.008,.265,.516,.767]; w=.223; y=.705; h=.223
    specs=[('Multiview RGB','Frames retain their\nper-sequence order.'),('FFT quality filter','Radial high-frequency\nscore screens blur.'),('Five text prompts','P2: entire potted plant\nP1/P3–P5: alternatives'),('SAM3 candidates','Per-prompt masks,\nboxes and scores.')]
    for j,(xx,(title,body)) in enumerate(zip(x,specs)):
        tile(ax,xx,y,w,h,title,body,TEAL if j<2 else BLUE)
        if j<3: arrow(ax,xx+w+.007,y+h/2,x[j+1]-.007,y+h/2)
    section(ax,.008,.644,'b   Selection and refinement')
    for j,(xx,(title,body)) in enumerate(zip(x,[('SG-MPS','Use heuristic supports\nand penalties to gate\nacross prompt groups.'),
            ('CDR / R1','Rerank instances\nwithin the selected\nprompt group.'),('SB-FBPR','Sample positive interior\nand balanced negative\npoints; re-query SAM3.'),
            ('GCFC / output','Guarded geometric\ncorrection in full chain;\nkeep selected masks.')])):
        tile(ax,xx,.362,w,.238,title,body,TEAL if j<2 else PLUM)
        if j<3: arrow(ax,xx+w+.007,.481,x[j+1]-.007,.481)
    section(ax,.008,.287,'c   Real view: KongQueZhuYu · 0000')
    common=PANEL/'Frontend/Representative4/selected_frames/KongQueZhuYu_0000.jpg'
    for j,(src,title) in enumerate([(common,'RGB view ID 0000'),(REP/'gt_overlays/KongQueZhuYu_0000_gt_overlay.png','Annotated target overlay'),
                (REP/'overlays/RAP-FSAM3-v2/KongQueZhuYu_0000_RAP-FSAM3-v2_overlay.png','Archived RAP output'),
                (REP/'error_maps/RAP-FSAM3-v2/KongQueZhuYu_0000_RAP-FSAM3-v2_error.png','Archived error overlay')]):
        method_img(p,fig,ax,(x[j],.009,w,.229),src,title,'c')
    p.source(FRONT/'Method/FFT/blur_filter.py','a','implementation'); p.source(FRONT/'Method/Recovered_RAP_Chain/01_PAPER_CODE_MAPPING.md','a–b','implementation and scope')
    p.finish(fig,'Schematic: execution order of the recovered pipeline. Lower panels are archived outputs; displayed image scales may differ.')


def mechanism_metric(p):
    fig=p.start('M02_Metric_GGGS_TSDF','Metric geometry from multiview RGB',
                'Mechanism diagram | independent camera recovery, scale quality control, and depth fusion',
                ['Native geometry','Independent metric anchors','Reconstruction intermediates'],
                'Known target geometry sets the metric similarity scale while masked GGGS depth supplies the TSDF observations.',
                shape=None,size=(183,143))
    ax=canvas(fig)
    section(ax,.008,.960,'a   Native geometry and fusion path')
    steps=[('COLMAP SfM','RGB correspondences\nK, R, t + sparse points'),('Native GGGS','Train in native gauge\n30K, SH3, SG0'),('Metric gate','Scale geometry and\ntranslations once'),('Filtered depth','Median camera-Z\nMask + alpha ≥ 0.5'),('Bounded TSDF','Fuse training views\nExtract zero surface')]
    for i,(title,body) in enumerate(steps):
        x=.008+i*.199; tile(ax,x,.710,.180,.209,title,body,BLUE if i<2 else TEAL)
        if i<4: arrow(ax,x+.185,.813,x+.193,.813)
    section(ax,.008,.640,'b   Two target routes')
    cube='Fit a 3D cube to multiview masks.\n'+r'$s=2\,\mathrm{cm}/\ell_{\mathrm{cube}}$'+'  ·  19/19'
    tile(ax,.008,.392,.285,.206,'Potted plants: known cube',cube,BLUE)
    tile(ax,.346,.392,.294,.206,'Retain qualified scale','Accepted: use centimeters.\nOtherwise: retain SfM units.',OCHRE)
    metric = p.table('Tomato_Metric_19_Reconstruction','b',['sample_id','cm_per_unit'])
    scaled = (np.isfinite(metric.cm_per_unit) & metric.cm_per_unit.gt(0)).sum()
    board = f'Triangulate board baselines.\nMetric subset: {scaled}/{len(metric)} scans.'
    tile(ax,.695,.392,.288,.206,'TomatoPGT: planar board',board,TEAL)
    arrow(ax,.501,.606,.501,.701); arrow(ax,.300,.495,.338,.495,color=MUTED); arrow(ax,.685,.495,.646,.495,color=MUTED)
    section(ax,.008,.320,'c   Real intermediates: KongQueZhuYu · shared preview camera')
    folder=PANEL/'TSDF_Mesh/KongQueZhuYu'
    for i,(name,label) in enumerate([('GGGS_Render.png','GGGS render'),('Rendered_Depth.png','Cached depth'),
                                      ('Valid_Depth_Mask.png','Valid depth'),('TSDF_Mesh_Render.png','Bounded TSDF mesh'),
                                      ('Plant_Only_Mesh_Render.png','Plant-only mesh')]):
        method_img(p,fig,ax,(.009+i*.198,.012,.181,.262),folder/name,label,'c')
    for path in ['batch_mesh.py','batch_calibrate.py','tomatopgt_targets.py','auto_metric_scale.py']:
        p.source(CODE/'ForeSplat_AutoGaussianCleaner/phenotyping'/path,'a–b','implementation');
    p.source(folder/'Depth_Display_Protocol.json','c','depth display range')
    p.finish(fig,'TSDF: same bounded extractor; target source differs by cohort. Depth color is display-only; see the archived display protocol.')


def mechanism_organs(p):
    fig=p.start('M03_Organ_and_traits','From projected masks to organ traits',
                'Mechanism diagram | semantic mesh separation, stable instances and trait definitions',
                ['2D to 3D association','Paired length and width landmarks','Cohort-specific validation rules'],
                'Projection transfers multicamera semantic evidence to mesh faces and builds organ identities used for metric traits.',
                shape=None,size=(183,141))
    ax=canvas(fig)
    section(ax,.008,.963,'a   Transfer image evidence to mesh faces')
    nodes=[('SAM3 per view','Plant / pot / target\nOrgan proposals'),('COLMAP projection','Visible pixels index\nTSDF mesh triangles'),
           ('Fuse evidence','Aggregate semantics;\ngroup organ support'),('Independent meshes','Plant-only and pot;\nleaf / compound PLY')]
    for i,(title,body) in enumerate(nodes):
        x=.008+i*.253; tile(ax,x,.704,.218,.216,title,body,BLUE if i<2 else TEAL)
        if i<3: arrow(ax,x+.224,.810,x+.246,.810)
    section(ax,.008,.638,'b   Define the same landmarks before comparing measurements')
    # The silhouette is a schematic, separate from any measured leaf.
    curve = np.array([[.058,.419],[.152,.578],[.312,.554],[.433,.432],[.340,.343],[.171,.299],[.058,.419],[.058,.419]])
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    closed=PlotPath(curve,[PlotPath.MOVETO]+[PlotPath.CURVE4]*6+[PlotPath.CLOSEPOLY])
    ax.add_patch(PathPatch(closed,facecolor='#E2F0EB',edgecolor=TEAL,lw=1.15))
    ax.plot([.059,.432],[.419,.432],color=BLUE,lw=1.2,ls='--',zorder=2)
    ax.scatter([.059,.432],[.419,.432],c=BLUE,s=8,zorder=3)
    ax.plot([.239,.243],[.526,.349],color=OCHRE,lw=1.05); ax.scatter([.239,.243],[.526,.349],c=OCHRE,s=8,zorder=3)
    ax.text(.071,.546,'Chord  |  3D base → tip',color=BLUE,size=6.7)
    ax.text(.160,.301,'Width  |  same transverse section',color=OCHRE,size=6.4)
    ax.text(.053,.381,'base',size=5.4); ax.text(.435,.417,'tip',size=5.4)
    tile(ax,.544,.349,.438,.239,'Reference-guided tomato agreement','Project annotated reference landmarks\nto the matched leaf mesh and apply QC.\nMeasure paired Euclidean distances (cm).',PLUM)
    section(ax,.008,.240,'c   Keep validation targets distinct')
    tile(ax,.008,.025,.473,.178,'Potted: numbered leaves','57 manually confirmed identities; independent\nmatched manual values not yet populated.',BLUE)
    tile(ax,.515,.025,.467,.178,'Tomato: compound leaves','Separate compound leaves from leaflets. Observed\none-sided area has no comparable reference.',TEAL)
    for path in ['sam_inference.py','fuse_semantics.py','associate_leaves.py','extract_instances.py','measure_phenotypes.py']:
        p.source(CODE/'ForeSplat_AutoGaussianCleaner/phenotyping'/path,'a','implementation')
    p.source(CODE/'Entrypoints/pair_tomatopgt_compounds.py','b–c','matching implementation'); p.source(MASTER/'Tomato_Leaf_Trait_Pairs.csv','b–c','evidence for panel scope')
    p.finish(fig,'Landmark schematic, not a real silhouette. Identity matching is separate from 3D segmentation IoU and automatic trait accuracy.')


def read_triangle_ply(path):
    """Read the archive's binary little-endian XYZ/RGB triangular mesh layout."""
    types={'float':'<f4','double':'<f8','uchar':'u1','int':'<i4','uint':'<u4'}
    with path.open('rb') as f:
        assert f.readline().strip()==b'ply' and f.readline().strip()==b'format binary_little_endian 1.0'
        counts, fields, element = {}, [], ''
        face_dtype = None
        for line in iter(f.readline,b''):
            tags=line.decode('ascii').split()
            if tags==['end_header']: break
            if tags[0]=='element': element=tags[1]; counts[element]=int(tags[2])
            if tags[0]=='property' and element=='vertex': fields.append((tags[2],types[tags[1]]))
            if tags[:2]==['property','list'] and element=='face': face_dtype=np.dtype([('n',types[tags[2]]),('ix',types[tags[3]],3)])
        vertex=np.fromfile(f,dtype=np.dtype(fields),count=counts['vertex'])
        face=np.fromfile(f,dtype=face_dtype,count=counts['face'])
    assert len(face)==counts['face'] and np.all(face['n']==3)
    xyz=np.column_stack([vertex[k] for k in 'xyz']); rgb=np.column_stack([vertex[k] for k in ['red','green','blue']])/255
    return xyz,face['ix'],rgb


def case_study(p):
    names=['Original RGB','GGGS render','Metric TSDF','Plant-only mesh','Whole-plant instances',
           'Compound_006 overlay','Independent compound mesh','Official ↔ model registration','Leaf landmark comparison']
    fig=p.start('C01_Tomato_Beefsteak','Case study: from RGB to a tomato compound leaf',
                'TomatoPGT · Beefsteak · BFS_06262025 · metric reconstruction with board scale QC',names,
                'A verified-scale case links an input view to an independent leaf mesh and an accepted reference-guided chord pair.',shape=None,size=(183,163))
    p.source(CASE/'preview_view.json','a–e','common view provenance'); p.source(CROP/'instance.json','f–g','independent organ provenance')
    x=[.055,.282,.509,.736]; w=.212
    for i,(path,title) in enumerate(zip(['Original_RGB.png','GGGS_Render.png','TSDF_Mesh_Render.png','Plant_Only_Mesh_Render.png'],names[:4])):
        img(p,fig,[x[i],.652,w,.164],CASE/path,f'{string.ascii_lowercase[i]}   {title}',string.ascii_lowercase[i])
    fig.text(.057,.848,'Shared camera · DSC00057',color=TEAL,size=7.5,weight='bold')
    fig.add_artist(Line2D([.055,.95],[.628,.628],transform=fig.transFigure,lw=.75,color=RULE))
    img(p,fig,[.055,.398,.274,.192],CASE/'Instance_Overlay.png','e   Whole-plant instances','e')
    img(p,fig,[.353,.398,.274,.192],CROP/'Instance_Overlay.png','f   Compound_006 overlay (detail)','f',crop=(582,364,908,585))
    ax=fig.add_axes([.649,.391,.304,.226],projection='3d'); ax.set_axis_off(); ax.view_init(elev=72,azim=-75)
    xyz,faces,rgb=read_triangle_ply(p.source(CROP/'Leaf_Instance_Mesh_cm.ply','g','archived triangular mesh'))
    centered=xyz-xyz.mean(axis=0)
    _,basis=np.linalg.eigh(np.cov(centered,rowvar=False)); basis=basis[:,::-1]
    if np.linalg.det(basis)<0: basis[:,2]*=-1
    local=centered@basis
    ax.add_collection3d(Poly3DCollection(local[faces],facecolors=rgb[faces].mean(axis=1),edgecolor='#527161',linewidth=.10))
    ax.auto_scale_xyz(*local.T); ax.set_box_aspect(np.maximum(np.ptp(local,axis=0),.01),zoom=1.16); ax.set_proj_type('ortho')
    fig.text(.661,.605,'g   Independent compound mesh',size=6.6,weight='medium')
    fig.text(.800,.400,f'{len(faces):,} triangles · cm PLY',size=5.4,color=MUTED,ha='center')
    fig.add_artist(Line2D([.055,.95],[.375,.375],transform=fig.transFigure,lw=.75,color=RULE))
    img(p,fig,[.052,.139,.435,.170],CASE/'Official_Predicted_Registration.png','h   Whole-plant registration', 'h')
    img(p,fig,[.512,.139,.435,.170],CROP/'Measurement_or_Registration_Landmarks.png','i   Compound landmarks in aligned coordinates','i')
    tab=p.table('Tomato_Leaf_Plot_Data','i',['scene','candidate_id','trait','reference','model','error_model_minus_reference'])
    r=tab[(tab.scene.eq('BFS_06262025'))&(tab.candidate_id.eq('Compound_006'))&(tab.trait.eq(CHORD))]; assert len(r)==1
    p.dump(r,'accepted_case_chord'); r=r.iloc[0]
    fig.text(.057,.097,'OFFICIAL REFERENCE',color=BLUE,size=5.8,weight='bold'); fig.text(.224,.097,'MODEL',color=OCHRE,size=5.8,weight='bold');
    fig.text(.512,.097,f'Chord: {r.reference:.3f} cm reference   |   {r.model:.3f} cm mesh   |   {r.error_model_minus_reference:+.3f} cm signed difference',size=5.9)
    p.finish(fig,'Panels a–e share the scene camera; f is a crop of the instance reference. Panel g is orthographic. Registration is for correspondence.')


FIGURES=(mechanism_front,mechanism_metric,mechanism_organs,case_study,front_performance,front_ablation,gaussian,
         mesh_selection,reconstruction,scale,potted_leaves,tomato_correspondence,tomato_agreement,plant_traits)


def review_sheet(p):
    fig, axes = plt.subplots(4,4,figsize=(14,14),layout='constrained')
    for ax in axes.ravel(): ax.axis('off')
    for ax,item in zip(axes.ravel(),p.index):
        with Image.open(p.out/f'{item["id"]}.png') as image:
            image.thumbnail((660,660)); ax.imshow(image)
        ax.set_title(item['id'].replace('_','  ',1),fontsize=8,loc='left',weight='bold')
    fig.savefig(p.out/'Contact_Sheet.jpg',dpi=160,pil_kwargs={'quality':91})
    plt.close(fig)


def docs(p):
    pd.DataFrame(p.sources).to_csv(p.out/'Figure_Source_Manifest.csv',index=False)
    pd.DataFrame(p.index).assign(panels=lambda d:d.panels.map(' | '.join)).to_csv(p.out/'Figure_Index.csv',index=False)
    lines = ['# ForeSplat 图件试排：14 张', '', '原稿与模型未改动。图号 `M`、`C`、`A` 为本套内部编号。',
        'PDF/SVG 用于排版，PNG 为 600 dpi 预览。英文面板便于投稿，以下提供中英文读图索引。', '',
        '- [总览](Contact_Sheet.jpg) · [逐图索引](Figure_Index.csv) · [素材来源](Figure_Source_Manifest.csv) · [绘图脚本](render_foresplat_paper_figures.py)',
        '- `Source_Data/` 保存各分析图使用的数据。来源表中的原始路径相对于 ForeSplat 证据包，绘图数据路径相对于本目录。',
        '- 散点图每个点代表图中标明的视角、植株、扫描或叶；散点分布的深色横线为中位数（若有）。结果为描述性，没有将重复视角或同一扫描的器官视为独立生物重复进行推断。',
        '- 原照片和既有可视化没有局部调整亮度、对比度或擦除内容。C01f 为标注过出处与像素范围的局部底图；C01g 对归档三角网格做主轴定向正交显示，不改变几何或比例。',
        '- 番茄物理尺度恢复按公制表型子集报告：19/19 个扫描有有效尺度因子及公制网格（M02、A05f、A06）。全部 42 个番茄扫描用于重建覆盖统计（A05a–e）。',
        '- A01 仅代表 4 株 20 帧的已定位前端测试；A02 是另一组 4 株 16 开发帧；A03–04 均为孔雀竹芋单株选型。A09 是参考引导的一致性，A10 的株高对照仅探索，面积无独立真值。', '', '## 图件和面板说明', '']
    zh=['RAP-FSAM3 前端掩膜机制','独立 SfM、公制标定与 GGGS–TSDF 重建机制','多视角器官提取、配对与量测机制',
        'Beefsteak 番茄整株到独立复叶案例','前端掩膜对照','前端提示、排序与细化消融','五种 Gaussian 单株对照',
        '七组提取器的原始与后处理网格对照','两批重建分阶段覆盖','标定来源与尺度质控','盆栽叶片候选与 57 片量测可用性',
        '番茄复叶候选、对应和匹配覆盖','番茄复叶长宽的一致性','整株尺寸候选与单侧观测面积']
    for item, label in zip(p.index,zh,strict=True):
        name=item['id']; lines.extend([f'### {name.split("_")[0]} · {label}', '',
            f'[{name}.pdf]({name}.pdf) · [SVG]({name}.svg) · [PNG]({name}.png)', '',
            f'**Scope.** {item["scope"]}. {item["claim"]}', '',
            ' · '.join(f'**{letter}** {title}' for letter,title in zip(string.ascii_lowercase,item['panels'])), '',item['note'],''])
    lines.extend(['## 重绘', '', '用证据包自带的 Python 绘图环境运行下列命令（从本目录执行）。', '',
        '```bash', 'python render_foresplat_paper_figures.py --evidence .. --output ../12_PAPER_FIGURES_rerender', '```', ''])
    (p.out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    if Path(__file__).resolve()!=p.out/Path(__file__).name:
        shutil.copy2(Path(__file__),p.out/Path(__file__).name)


def bundle(p):
    name='ForeSplat_14_Figures.zip'
    with zipfile.ZipFile(p.out/name,mode='w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(p.out.rglob('*')):
            if f.is_file() and f.name!=name:
                z.write(f,f.relative_to(p.out))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dpi',type=int,default=600)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    pack=Set(args.evidence.resolve(),args.output.resolve(),args.dpi)
    for fn in FIGURES: fn(pack)
    assert len(pack.index)==14 and all(len(i['panels'])==6 for i in pack.index if i['id'].startswith('A'))
    review_sheet(pack); docs(pack); bundle(pack)


if __name__=='__main__':
    main()
