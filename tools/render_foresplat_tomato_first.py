#!/usr/bin/env python3
"""Compose manuscript figures from archived ForeSplat evidence, with TomatoPGT first."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import string

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from render_foresplat_paper_figures import (
    AREA, CASE, CHORD, CODE, CULTIVARS, FRONT, INK, MUTED, PANEL, POTTED, REP, RULE,
    WIDTH, Set, arrow, bundle, cat, dotsets, grid, identity, img, method_img,
    read_triangle_ply, review_sheet, section, simple_bar, tile,
)

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                     'pdf.fonttype':42,'svg.fonttype':'none'})

TOMATO = '#007A73'
POTTED_COLOR = '#797E89'
ACCENT = '#AB671F'
GROUPS = ['BFS', 'BBH', 'CH', 'Potted']
COLORS = [TOMATO, '#35799D', '#A05B27', POTTED_COLOR]
CODES = {key: values[0] for key, values in CULTIVARS.items()}
MARKS = ['o', '^', 's', 'D']
PANEL_IDS = string.ascii_lowercase
POTTED_ASSETS = PANEL/'Potted_MultiSpecies'
SUPPLEMENT = Path('09_SUPPLEMENTARY_MATERIALS')
SCAN_REPORTS = SUPPLEMENT/'All_42_Tomato_Scans'
POTTED_REPORTS = SUPPLEMENT/'All_19_Potted_Plants'
ORGAN_ARCHIVE = Path('06_PLANT_AND_ORGAN_SEGMENTATION')
TOMATO_PHENO = Path('07_PHENOTYPING_EVALUATION/TomatoPGT_42/Per_Sample')


@dataclass(frozen=True)
class OrganCase:
    name: str
    sample: str
    organ: str
    handwritten: int | None = None

    @property
    def scene(self):
        return (SCAN_REPORTS if self.handwritten is None else POTTED_REPORTS)/self.sample/'Figures'

    @property
    def asset(self):
        if self.handwritten is None:
            return ORGAN_ARCHIVE/'Tomato_Compound_Leaves'/self.sample/self.organ
        return ORGAN_ARCHIVE/'Potted_Leaf_Instances'/self.sample/f'Handwritten_{self.handwritten}_{self.organ}'


CASES = [OrganCase('Beefsteak','BFS_06262025','Compound_002'),
         OrganCase('Big Boy Hybrid','BBH_06122025','Compound_009'),
         OrganCase('Celebrity Hybrid','CH_06102025','Compound_010'),
         OrganCase('Rubber plant','XiangPiShu2','Leaf_002',1),
         OrganCase('Peacock plant','KongQueZhuYu','Leaf_002',3)]


class Plates(Set):
    def start(self, key, title, scope, panel_names, claim, *, shape=(3, 2), size=(183, 154)):
        self.key = key
        self.index.append(dict(id=key, title=title, scope=scope, panels=panel_names,
                               claim=claim, file=f'{key}.pdf'))
        fig = plt.figure(figsize=tuple(dim/25.4 for dim in size))
        if not shape:
            return fig
        layout = fig.add_gridspec(*shape, left=.115, right=.965, bottom=.077, top=.950,
                                  wspace=.48, hspace=.77)
        axes = [fig.add_subplot(cell) for cell in layout]
        for ax, letter, label in zip(axes, PANEL_IDS, panel_names):
            ax.set_title(f'{letter}   {label}', pad=5.5)
            ax.set_axisbelow(True)
        return fig, axes

    def finish(self, fig, note):
        self.index[-1]['note'] = note
        for ax in fig.axes:
            for axis in (ax.xaxis,ax.yaxis):
                if axis.get_scale() in ('log','symlog'):
                    axis.set_major_formatter(FuncFormatter(compact_tick))
        if os.environ.get('FORESPLAT_FIGURE_QA') and self.key.startswith('A'):
            from audit_panel_alignment import require_matplotlib_panel_alignment
            qa_path = self.out/'QA'
            qa_path.mkdir(exist_ok=True)
            require_matplotlib_panel_alignment(fig, json_out=qa_path/f'{self.key}_alignment.json')
        for fmt in ('png', 'pdf', 'svg'):
            fig.savefig(self.out/f'{self.key}.{fmt}', dpi=self.dpi)
        plt.close(fig)
        print(self.key, flush=True)


def compact_tick(value, _):
    if abs(value)>=1_000_000:
        return f'{value/1_000_000:g}M'
    if abs(value)>=1_000:
        return f'{value/1_000:g}k'
    return f'{value:g}'


def figure_axes(fig):
    ax = fig.add_axes([.045,.035,.919,.94], xlim=(0,1), ylim=(0,1))
    ax.axis('off')
    return ax


def cohort(p, panels):
    t = p.table('Tomato_42_Reconstruction', panels, ['sample_id','cultivar','rgb','independent_colmap','gggs_30k',
        'metric_mesh','plant_faces','leaf_candidates','compound_candidates','official_compounds_from_point_labels',
        'tsdf_mesh_archive','plant_only_archive','train_views_archive','test_views_archive','preview_rgb_archive'])
    t = t.rename(columns={'rgb':'input_views','independent_colmap':'registered_views'})
    t['group'] = t.cultivar.map(CODES)
    t['colmap'] = t.registered_views.gt(0)
    h = p.table('Potted_19_Reconstruction', panels, ['sample_id','plant_type','group','raw_views','registered_views',
        'true_colmap','gggs_30k','metric_mesh','plant_only_faces','stable_leaf_count','fragmented_leaf_candidates',
        'tsdf_mesh_archive','plant_only_archive','train_views_archive','test_views_archive','preview_rgb_archive'])
    h = h.rename(columns={'group':'potted_split','raw_views':'input_views','true_colmap':'colmap',
                          'plant_only_faces':'plant_faces','stable_leaf_count':'leaf_candidates'})
    h['group'] = 'Potted'
    data = pd.concat([t,h], ignore_index=True)
    data['registered_pct'] = 100*data.registered_views/data.input_views
    return data


def metric_cohort(p, panels):
    t = p.table('Tomato_Metric_19_Reconstruction', panels, ['sample_id','cultivar','metric_mesh','cm_per_unit'])
    t['group'] = t.cultivar.map(CODES)
    h = p.table('Potted_19_Reconstruction', panels, ['sample_id','plant_type','group','metric_mesh','scale_cm_per_unit'])
    h = h.rename(columns={'group':'potted_split','scale_cm_per_unit':'cm_per_unit'})
    h['group'] = 'Potted'
    assert len(t)==len(h)==19
    return pd.concat([t,h],ignore_index=True)


def scatter(ax, data, x, y, xlabel, ylabel, *, groups=GROUPS, legend=False):
    for group in groups:
        i = GROUPS.index(group)
        part = data.loc[data.group.eq(group)]
        ax.scatter(part[x],part[y],s=17,c=COLORS[i],marker=MARKS[i],label=group,
                   alpha=.77,linewidth=.34,edgecolor='white',zorder=3)
    ax.set(xlabel=xlabel,ylabel=ylabel)
    grid(ax)
    if legend:
        ax.legend(loc='best',ncols=2,handletextpad=.32,columnspacing=.72)


def dots(ax, data, key, ylabel, *, groups=GROUPS):
    dotsets(ax,data,'group',key,groups,[COLORS[GROUPS.index(g)] for g in groups],ylabel)


def ratios(ax, labels, counts, totals, colors, xlabel):
    pos = np.arange(len(labels))
    values = np.divide(counts,totals)
    ax.barh(pos,np.ones_like(values),height=.55,color=RULE)
    ax.barh(pos,values,height=.55,color=colors,zorder=2)
    for i,(count,total) in enumerate(zip(counts,totals)):
        ax.text(1.025,i,f'{count}/{total}',va='center',fontsize=5.7)
    cat(ax,pos,labels,axis='y')
    ax.set(xlim=(0,1.24),xlabel=xlabel,xticks=[0,.25,.5,.75,1])
    ax.invert_yaxis()
    grid(ax,'x')


def group_ratio(ax, data, key, xlabel):
    blocks = [data[data.group.eq(group)] for group in GROUPS]
    ratios(ax,GROUPS,[int(b[key].sum()) for b in blocks],[len(b) for b in blocks],COLORS,xlabel)


def proportions(ax, groups, xlabel):
    labels = [label for label,_,_,_ in groups]
    ratios(ax,labels,[k for _,k,_,_ in groups],[n for _,_,n,_ in groups],
           [color for *_,color in groups],xlabel)


def ecdf(ax, data, key, xlabel):
    for group,color in zip(GROUPS,COLORS):
        vals = np.sort(data.loc[data.group.eq(group),key].dropna().to_numpy())
        ax.step(np.r_[vals[0],vals],np.arange(len(vals)+1)/len(vals),where='post',color=color,label=group,lw=1.05)
    ax.set(xlabel=xlabel,ylabel='Cumulative fraction',ylim=(0,1.04))
    ax.legend(ncols=2,handlelength=1.4,loc='lower right')
    grid(ax)


def show(ax,p,path,panel,*,box=(0,0,1,1)):
    picture=ax.inset_axes(box)
    with Image.open(p.source(path,panel,'archived image')) as image:
        image.thumbnail((1550,1550))
        picture.imshow(image.convert('RGB'),interpolation='antialiased')
    picture.set_axis_off()
    ax.set_axis_off()


def organ_meta(p,case,panel):
    meta=json.loads(p.source(case.asset/'instance.json',panel,'selected organ provenance').read_text())
    assert meta['plant']==case.sample and meta['leaf_id']==case.organ and meta['unit']=='cm'
    return meta


def organ_reference(ax,p,case,panel,*,inset=False):
    meta=organ_meta(p,case,panel)
    view=meta['reference_view']
    path=p.source(case.asset/'reference.png',panel,f'selected organ overlay; camera {view}')
    mask_path=p.source(case.asset/'masks'/f'{view}.png',panel,'selected organ mask for crop bounds')
    with Image.open(path) as image, Image.open(mask_path) as mask:
        assert image.size==mask.size
        box=mask.convert('L').getbbox()
        assert box
        x0,y0,x1,y1=box
        pad=max(12,round(.23*max(x1-x0,y1-y0)))
        crop=(max(0,x0-pad),max(0,y0-pad),min(image.width,x1+pad),min(image.height,y1+pad))
        picture=ax.inset_axes([0,0,1,1]) if inset else ax
        picture.imshow(image.crop(crop).convert('RGB'),interpolation='antialiased')
        picture.set_axis_off()
        p.sources.append(dict(figure=p.key,panel=panel,role=f'overlay crop xyxy={crop}',path=str(case.asset/'reference.png')))
    ax.set_axis_off()


def capture(p):
    fig,a = p.start('A01_Capture_registration','RGB input and view coverage',
        '42 TomatoPGT scans and 19 potted plants. BFS, BBH and CH denote the three tomato cultivars.',
        ['Input RGB views','Registered fraction','Input vs registered views','Held-out RGB views',
         'Reconstruction RGB views','Held-out fraction'],
        'Shows capture and reconstruction split coverage in both cohorts.')
    d = cohort(p,'a–f')
    for phase in ('train','test'):
        d[f'{phase}_count'] = [len(p.source(path,'d–f',f'{phase} split').read_text().split())
                               for path in d[f'{phase}_views_archive']]
    d['test_pct'] = 100*d.test_count/(d.test_count+d.train_count)
    dots(a[0],d,'input_views','RGB views per sample')
    dots(a[1],d,'registered_pct','Registered views (%)')
    identity(a[2],d[['input_views','registered_views']])
    scatter(a[2],d,'input_views','registered_views','Input views','Registered views',legend=True)
    dots(a[3],d,'test_count','Held-out views')
    dots(a[4],d,'train_count','Reconstruction views')
    dots(a[5],d,'test_pct','Held-out views (%)')
    p.dump(d,'capture_and_splits'); p.finish(fig,'One point per sample. Dark lines mark medians. The held-out fraction uses the archived train and test lists.')


def front(p):
    fig,a = p.start('A02_Foreground_evidence','Foreground evidence in the two cohorts',
        'Tomato held-out mesh–SAM3 agreement and the separate hand-annotated potted diagnostic.',
        ['Tomato RGB','Tomato input mask','Potted RGB','Potted input mask','Tomato held-out silhouettes','Potted mask diagnostic'],
        'Pairs comparable visual stages with protocol-specific evidence for each cohort.')
    pot = POTTED_ASSETS/'KongQueZhuYu'
    for i,(folder,name) in enumerate([(CASE,'Original_RGB.png'),(CASE,'Input_Mask.png'),
                                      (pot,'Original_RGB.png'),(pot,'Input_Mask.png')]):
        show(a[i],p,folder/name,PANEL_IDS[i])
    views = []
    scans = cohort(p,'e')
    for row in scans[scans.group.ne('Potted')].itertuples():
        path=SCAN_REPORTS/row.sample_id/'Figures/heldout_silhouettes.csv'
        views.append(p.csv(path,'e',['view','iou','sam3_coverage']).assign(sample_id=row.sample_id,group=row.group))
    views = pd.concat(views,ignore_index=True)
    summary = views.groupby(['sample_id','group'],as_index=False).agg(iou=('iou','median'), valid_views=('iou','count'))
    dots(a[4],summary,'iou','Median mesh–SAM3 IoU',groups=GROUPS[:3]); a[4].set_ylim(0,1.03)
    h = p.csv(REP/'source_data/frame_metrics.csv','f',['sample','frame','method','miou','f1'])
    names = {'RAP-FSAM3-v2':'RAP v2','SAM3_P2':'SAM3 P2'}
    h['label'] = h.method.map(names)
    subset = h.dropna(subset=['label'])
    assert subset.label.value_counts().to_dict() == {'RAP v2':20,'SAM3 P2':20}
    dotsets(a[5],subset,'label','miou',['RAP v2','SAM3 P2'],[POTTED_COLOR,'#B6B9C1'],'Mask IoU vs annotation')
    common_floor = np.floor(10*min(summary.iou.min(),subset.miou.min()))/10
    for ax in a[4:]: ax.set_ylim(common_floor,1.025)
    p.dump(views,'tomato_heldout_view_scores'); p.dump(summary,'tomato_scan_medians'); p.dump(subset,'potted_annotated_frames')
    p.finish(fig,'Tomato IoU compares a projected mesh with SAM3 on views held out from training and TSDF fusion. Potted IoU compares RAP and single-prompt SAM3 masks with manual annotations from 4 plants (20 views).')


def mesh_faces(path):
    with path.open('rb') as stream:
        assert stream.readline().strip()==b'ply',path
        for line in stream:
            if line.startswith(b'element face '):
                return int(line.split()[2])
            if line.strip()==b'end_header':
                break
    raise ValueError(f'Missing mesh face count in {path}')


def reconstruction(p):
    fig,a = p.start('A03_Reconstruction_coverage','Reconstruction completion and mesh coverage',
        'All 42 tomato scans and 19 potted plants for reconstruction. The metric panel uses both 19-item measurement cohorts.',
        ['Completed reconstruction stages','Metric meshes','TSDF mesh faces','Plant-only mesh faces',
         'TSDF vs plant-only mesh','Plant-only mesh distribution'],
        'Shows complete reconstruction coverage and the range of mesh sizes in both cohorts.')
    d = cohort(p,'a,c–f')
    for key,column,panels in [('tsdf_faces','tsdf_mesh_archive','a,c,e'),('plant_archive_faces','plant_only_archive','a,d–f')]:
        d[key] = [mesh_faces(p.source(path,panels,'archived mesh header')) for path in d[column]]
    x = np.arange(4)
    for off,(label,rows,color) in zip((-.17,.17),[('Tomato',d[d.group.ne('Potted')],TOMATO),('Potted',d[d.group.eq('Potted')],POTTED_COLOR)]):
        counts = [rows.colmap.sum(),rows.gggs_30k.sum(),rows.tsdf_faces.gt(0).sum(),rows.plant_archive_faces.gt(0).sum()]
        a[0].bar(x+off,np.asarray(counts)/len(rows),width=.29,color=color,label=label)
        for i,count in enumerate(counts):
            a[0].text(i+off,count/len(rows)+.025,f'{count}/{len(rows)}',ha='center',size=5.2)
    cat(a[0],x,['COLMAP','GGGS','TSDF','Plant only']); a[0].set(ylim=(0,1.31),ylabel='Fraction of inputs')
    a[0].legend(loc='upper left',ncols=2); grid(a[0])
    m = metric_cohort(p,'b')
    groups = [('Tomato',m[m.group.ne('Potted')],TOMATO),('Potted',m[m.group.eq('Potted')],POTTED_COLOR)]
    proportions(a[1],[(label,int(rows.metric_mesh.sum()),len(rows),color) for label,rows,color in groups],
                'Fraction of metric cohort')
    dots(a[2],d,'tsdf_faces','Triangle faces'); a[2].set_yscale('log')
    dots(a[3],d,'plant_archive_faces','Triangle faces'); a[3].set_yscale('log')
    scatter(a[4],d,'tsdf_faces','plant_archive_faces','TSDF triangle faces','Plant-only triangle faces',legend=True)
    a[4].set(xscale='log',yscale='log')
    ecdf(a[5],d,'plant_archive_faces','Plant-only triangle faces'); a[5].set_xscale('log')
    a[5].grid(False)
    p.dump(d,'reconstruction_samples'); p.dump(m,'metric_samples')
    p.finish(fig,'Triangle count describes mesh size. Coverage counts refer to archived nonempty meshes. Each tomato scan is an observation at capture.')


def semantics(p):
    fig,a = p.start('A04_Plant_pot_separation','Plant and pot separation',
        '42 tomato scans and 19 potted plants, with matched-stage visual examples.',
        ['Tomato scene mesh','Tomato plant and organ detail','Potted scene mesh','Potted plant and organ detail',
         'Plant and pot triangle counts','Plant share of labeled faces'],
        'Shows the mesh products from plant–pot separation across the tomato and generalization cohorts.')
    for i,case in [(0,CASES[0]),(2,CASES[3])]:
        show(a[i],p,case.scene/'mesh.png',PANEL_IDS[i])
        show(a[i+1],p,case.scene/'plant_only.png',PANEL_IDS[i+1],box=[0,0,.62,1])
        obj=a[i+1].inset_axes([.64,.20,.35,.69],projection='3d')
        independent(obj,p,case.asset/'leaf_cm.ply',PANEL_IDS[i+1])
        a[i+1].text(.815,.16,case.organ,transform=a[i+1].transAxes,ha='center',va='top',size=5.5,color=MUTED)
    d = cohort(p,'e–f')
    pots = []
    for row in d.itertuples():
        plant = Path(row.plant_only_archive)
        pot = plant.with_name(plant.name.replace('plant_only_','pot_'))
        exists = (p.root/pot).is_file()
        pots.append(mesh_faces(p.source(pot,'e–f','pot mesh header')) if exists else 0)
    d['pot_faces'] = pots
    d['plant_face_share'] = 100*d.plant_faces/(d.plant_faces+d.pot_faces)
    scatter(a[4],d,'plant_faces','pot_faces','Plant triangle faces','Pot triangle faces',legend=True)
    a[4].set(xscale='symlog',yscale='symlog')
    a[4].grid(False)
    dots(a[5],d,'plant_face_share','Plant / (plant + pot) faces (%)')
    p.dump(d,'semantic_mesh_counts'); p.finish(fig,'The examples are Beefsteak BFS_06262025 and rubber plant XiangPiShu2. Insets show their selected organs at independent display magnifications. Plant share uses the two exported semantic meshes; zero pot faces denotes no exported pot mesh.')


def calibration(p):
    fig,a = p.start('A05_Metric_scale_QC','Physical scale and calibration quality',
        '19/19 tomato scans in the metric subset and 19/19 potted plants.',
        ['Physical scale recovered','Scale conversion factors','Calibration routes',
         'Target view support','Tomato board baseline QC','Potted cube fit QC'],
        'Uses each target route to recover metric geometry with route-specific quality checks.')
    m = metric_cohort(p,'a–f')
    h = p.table('Potted_19_Calibration','b–d,f',['sample_id','method','scale_cm_per_unit','retained_view_count',
        'cube_reprojection_error_px','boundary_rms_px'])
    t = p.table('Tomato_42_Calibration','b–e',['sample_id','method','scale_cm_per_unit','calibration_view_ids',
        'max_relative_baseline_scale_deviation'])
    t['retained_view_count']=t.calibration_view_ids.fillna('').map(lambda ids:len(set(ids.split())))
    h['route']=np.where(h.method.str.startswith('AUTOMATIC'),'SAM3 cube','Assisted cube')
    t['route']=np.where(t.method.str.startswith('Published'),'Published board','Transferred board')
    d=m.merge(pd.concat([t,h],ignore_index=True),on='sample_id',validate='one_to_one')
    assert len(d)==38 and np.allclose(d.cm_per_unit,d.scale_cm_per_unit)
    scaled=np.isfinite(d.scale_cm_per_unit) & d.scale_cm_per_unit.gt(0)
    tomato=d.group.ne('Potted')
    proportions(a[0],[('Tomato',int((scaled & tomato).sum()),int(tomato.sum()),TOMATO),
                      ('Potted',int((scaled & ~tomato).sum()),int((~tomato).sum()),POTTED_COLOR)],
                'Fraction of metric cohort')
    dots(a[1],d,'scale_cm_per_unit','Scale (cm per SfM unit)')
    routes=['Transferred board','Published board','SAM3 cube','Assisted cube']
    simple_bar(a[2],routes,d.route.value_counts().reindex(routes),[TOMATO,TOMATO,POTTED_COLOR,POTTED_COLOR],'Scans or plants')
    dots(a[3],d,'retained_view_count','Reported target views')
    d['baseline_error_pct']=100*d.max_relative_baseline_scale_deviation
    dots(a[4],d,'baseline_error_pct','Maximum baseline deviation (%)',groups=GROUPS[:3])
    z=d[d.group.eq('Potted')]
    a[5].scatter(z.retained_view_count,z.cube_reprojection_error_px,color=POTTED_COLOR,s=15,label='Reprojection')
    a[5].scatter(z.retained_view_count,z.boundary_rms_px,facecolor='none',edgecolor=POTTED_COLOR,s=15,label='Boundary RMS')
    a[5].set(xlabel='Retained views',ylabel='Cube fit residual (px)'); a[5].legend(loc='best',ncols=2)
    p.dump(d,'metric_scale_and_qc'); p.finish(fig,'Target view count uses distinct board-support views in tomato and retained mask views in potted plants. SfM scale factors have arbitrary starting units. Baseline deviation and cube residual are route-specific.')


def organs(p):
    fig,a = p.start('A06_Organ_instances','Visible organ meshes across cohorts',
        'All tomato scans and potted plants. Candidate collections use the instance definitions of each cohort.',
        ['Tomato selected compound','Potted selected leaf','Stable leaf candidates','Tomato compounds per scan',
         'Candidates vs plant mesh size','Potted fragmented candidates'],
        'Shows organ extraction in the tomato cultivars and across the potted plants.')
    organ_reference(a[0],p,CASES[2],'a',inset=True)
    organ_reference(a[1],p,CASES[3],'b',inset=True)
    d=cohort(p,'c–f')
    dots(a[2],d,'leaf_candidates','Candidates per scan or plant')
    dots(a[3],d,'compound_candidates','Compound candidates per scan',groups=GROUPS[:3])
    scatter(a[4],d,'plant_faces','leaf_candidates','Plant triangle faces','Stable leaf candidates',legend=True)
    a[4].set_xscale('log')
    z=d[d.group.eq('Potted')].copy()
    z['fragmented_pct']=100*z.fragmented_leaf_candidates/z.leaf_candidates
    for key,label,color,marker in [('heldout_validation','Held out',POTTED_COLOR,'D'),('development','Development','#A08CAC','s')]:
        part=z[z.potted_split.eq(key)]
        a[5].scatter(part.leaf_candidates,part.fragmented_pct,s=16,c=color,marker=marker,label=label)
    a[5].set(xlabel='Stable leaf candidates',ylabel='Marked fragmented (%)'); a[5].legend()
    p.dump(d,'scan_plant_candidates'); p.dump(z,'potted_fragmentation')
    p.finish(fig,'Panels a and b zoom in on selected masks for the Celebrity Hybrid compound and rubber plant leaf. Candidate counts include the original cohorts and represent their distinct organ protocols.')


def matched_records(p, panels, *, shapes=False):
    fields = ['scene','cultivar','metric_scan','official_point_instance','candidate_id','status']
    if shapes:
        fields += ['overlap_score','reference_coverage','prediction_coverage']
    tomato = p.table('Tomato_245_Compound_Instances',panels,fields)
    tomato['group'] = tomato.cultivar.map(CODES)
    potted = p.table('Potted_57_Leaf_Matching',panels,['plant_id','handwritten_number','leaf_id','human_confirmed','reference_view'])
    assert len(tomato)==245 and len(potted)==57
    return tomato,potted


def matching(p):
    fig,a = p.start('A07_Organ_correspondence','Assigning organ identities',
        '245 official tomato compound records and 57 potted leaves with handwritten identifiers.',
        ['Tomato records with candidates','Potted numbered identities','Tomato assignment statuses','Tomato projected coverage',
         'Tomato independent organ','Potted numbered organ'],
        'Shows correspondence evidence under the two label protocols.')
    t,h = matched_records(p,'a–d',shapes=True)
    t['has_candidate'] = t.candidate_id.notna()
    tg=[t[t.group.eq(g)] for g in GROUPS[:3]]
    proportions(a[0],[(group,int(d.has_candidate.sum()),len(d),c) for group,d,c in zip(GROUPS,tg,COLORS)],'Fraction with candidate')
    hg=[h[h.handwritten_number.eq(i)] for i in (1,2,3)]
    proportions(a[1],[(f'No. {i}',int(d.human_confirmed.sum()),len(d),POTTED_COLOR) for i,d in enumerate(hg,1)],
        'Fraction with confirmed identity')
    statuses=['strong_suggestion','review_suggestion','ambiguous_assignment','no_reliable_match','no_candidate']
    labels=['Strong','Review','Ambiguous','No reliable','No candidate']
    counts=t.status.value_counts().reindex(statuses,fill_value=0)
    simple_bar(a[2],labels,counts,[TOMATO,'#35799D','#A05B27','#AF91A6','#B6B9C1'],'Official compound records')
    a[2].grid(False)
    scatter(a[3],t,'reference_coverage','prediction_coverage','Reference covered fraction','Prediction covered fraction',groups=GROUPS[:3],legend=True)
    a[3].set(xlim=(0,1.04),ylim=(0,1.04))
    a[3].grid(False)
    independent(a[4],p,CASES[1].asset/'leaf_cm.ply','e'); independent(a[5],p,CASES[3].asset/'leaf_cm.ply','f')
    p.dump(t,'tomato_compound_assignments'); p.dump(h,'potted_numbered_identities')
    p.finish(fig,'Tomato assignment status comes from archived geometric matching and review proposals. Potted identities come from confirmed photo–mesh links. The axes in d show fractions within the match distance threshold. Insets show Big Boy Hybrid Compound_009 and rubber plant Leaf_002, independently fitted for display.')


def trait_records(p, panels):
    t=p.table('Tomato_Leaf_Trait_Pairs',panels,['scene','cultivar','official_point_instance','candidate_id','trait','status',
        'model','reference','error_model_minus_reference'])
    t['group']=t.cultivar.map(CODES)
    h=p.table('Potted_57_Trait_Measurements',panels,['plant_id','handwritten_number','leaf_id','automatic_leaf_length_cm',
        'automatic_leaf_width_cm','automatic_leaf_area_cm2','manual_leaf_length_cm','manual_leaf_width_cm','measurement_status'])
    h['group']='Potted'
    return t,h


def measurements(p):
    fig,a = p.start('A08_Leaf_measurement_coverage','Leaf trait availability and observed area',
        '70 official compound records in metric tomato scans and 57 manually identified potted leaves.',
        ['Tomato matched reference pairs','Potted extracted dimensions','Observed area availability','Observed area distributions',
         'Tomato area vs model width','Potted area vs model width'],
        'Separates dimensions extracted from mesh geometry from trait records with comparable references.')
    t,h=trait_records(p,'a–f')
    sizes=t.groupby('trait').size(); valid=t[t.status.eq('included_reference_guided_landmarks')]
    proportions(a[0],[(name,int(valid.trait.eq(key).sum()),sizes[key],TOMATO) for key,name in
        [(CHORD,'Chord'),(WIDTH,'Width'),(AREA,'Area')]],'Fraction with eligible pair')
    options=[('automatic_leaf_length_cm','Path'),('automatic_leaf_width_cm','Width'),('automatic_leaf_area_cm2','Area')]
    proportions(a[1],[(label,int(h[key].count()),len(h),POTTED_COLOR) for key,label in options],'Fraction with mesh value')
    field='model_observed_mesh_area_cm2'
    area=p.table('Tomato_70_Metric_Compounds','c–e',['scene','cultivar','official_point_instance','candidate_id',field])
    area['group']=area.cultivar.map(CODES)
    area=area.merge(valid[valid.trait.eq(WIDTH)][['scene','official_point_instance','model']],
        on=['scene','official_point_instance'],how='left',validate='one_to_one').rename(columns={'model':'width_cm',field:'area_cm2'})
    h=h.assign(area_cm2=h.automatic_leaf_area_cm2,width_cm=h.automatic_leaf_width_cm)
    values=pd.concat([area[['group','area_cm2']],h[['group','area_cm2']]],ignore_index=True)
    values['has_area']=values.area_cm2.notna()
    group_ratio(a[2],values,'has_area','Fraction with observed area')
    dots(a[3],values,'area_cm2','Observed one-sided area (cm²)'); a[3].set_yscale('log')
    scatter(a[4],area,'width_cm','area_cm2','Mesh transverse width (cm)','Observed area (cm²)',groups=GROUPS[:3],legend=True)
    scatter(a[5],h,'width_cm','area_cm2','Mesh width candidate (cm)','Observed area (cm²)',groups=['Potted'])
    for ax in a[4:]: ax.set_yscale('log')
    p.dump(t,'tomato_trait_reference_records'); p.dump(area,'tomato_model_areas'); p.dump(h,'potted_mesh_traits')
    p.finish(fig,'Tomato references are eligible for 14 chord pairs and 24 transverse-width pairs. Potted lengths are surface-path candidates. Areas describe visible one-sided surfaces of compound leaves or numbered leaves. Area has no matched reference in either cohort.')


def trait_agreement(p):
    fig,a = p.start('A09_Leaf_trait_agreement','Tomato leaf reference agreement and potted trait extraction',
        'Eligible reference-guided tomato pairs and descriptive values for the 57 potted leaves.',
        ['Tomato chord agreement','Tomato width agreement','Tomato chord residuals','Tomato width residuals',
         'Potted mesh path and width','Potted paired reference availability'],
        'Uses the tomato reference to assess landmark agreement and shows the measured range of the generalization leaves.')
    t,h=trait_records(p,'a–f')
    valid=t[t.status.eq('included_reference_guided_landmarks')]
    errors=p.table('Tomato_Leaf_Trait_Errors','a–d',['trait','n_pairs','n_scans','mae','rmse','mean_bias','r2_against_identity']).set_index('trait')
    for i,key in enumerate([CHORD,WIDTH]):
        z=valid[valid.trait.eq(key)]
        e=errors.loc[key]
        assert len(z)==e.n_pairs and np.isclose(z.error_model_minus_reference.abs().mean(),e.mae)
        identity(a[i],z[['model','reference']]); scatter(a[i],z,'reference','model','Reference (cm)','Mesh (cm)',groups=GROUPS[:3])
        a[i].grid(False)
        a[i].text(.05,.96,f'{len(z)} pairs · {int(e.n_scans)} scans\nMAE {e.mae:.2f} cm   RMSE {e.rmse:.2f} cm\nR² {e.r2_against_identity:.2f}',
            transform=a[i].transAxes,va='top',fontsize=5.5,bbox=dict(fc='white',ec='none',alpha=.88,pad=.9))
        scatter(a[i+2],z,'reference','error_model_minus_reference','Reference (cm)','Mesh − reference (cm)',groups=GROUPS[:3],legend=i==0)
        a[i+2].axhline(0,color=MUTED,ls='--',lw=.7); a[i+2].axhline(e.mean_bias,color=ACCENT,ls=':',lw=.8)
    scatter(a[4],h,'automatic_leaf_length_cm','automatic_leaf_width_cm','Mesh surface path (cm)','Mesh width candidate (cm)',groups=['Potted'])
    lengths=h[['manual_leaf_length_cm','automatic_leaf_length_cm']].notna().all(axis=1).sum()
    widths=h[['manual_leaf_width_cm','automatic_leaf_width_cm']].notna().all(axis=1).sum()
    proportions(a[5],[('Path',lengths,len(h),POTTED_COLOR),('Width',widths,len(h),POTTED_COLOR)],'Fraction with paired reference')
    p.dump(valid,'eligible_tomato_pairs'); p.dump(errors.reset_index(),'tomato_trait_errors'); p.dump(h,'potted_trait_reference_availability')
    p.finish(fig,'Tomato pairs use corresponding official and mesh landmarks. The dashed line marks identity or zero residual. The dotted residual line marks signed bias. Potted surface-path candidates remain descriptive. R² is evaluated against the identity line.')


def whole_plant(p):
    fig,a = p.start('A10_Whole_plant_traits','Visible height and canopy extent',
        '19 tomato scans with recovered scale and 19 potted plants. All lengths use centimeters.',
        ['Visible vertical span','Maximum horizontal span','Height vs canopy span',
         'Canopy span on fixed axes','Canopy-to-height ratio','Tomato official height comparison'],
        'Shows the visible dimensions available in both cohorts and the exploratory tomato height comparison.')
    samples=metric_cohort(p,'a–e')
    names=['plant_height_cm','canopy_span_a_cm','canopy_span_b_cm','canopy_width_cm','measurement_status']
    rows=[]
    for row in samples.itertuples():
        base=POTTED if row.group=='Potted' else TOMATO_PHENO
        file=base/row.sample_id/'plant_measurements.csv'
        rows.append(p.csv(file,'a–e',names).assign(sample_id=row.sample_id,group=row.group))
    d=pd.concat(rows,ignore_index=True)
    assert len(d)==38 and d[names[:4]].notna().all().all()
    d['canopy_height_ratio']=d.canopy_width_cm/d.plant_height_cm
    dots(a[0],d,'plant_height_cm','Visible height (cm)')
    dots(a[1],d,'canopy_width_cm','Maximum canopy span (cm)')
    scatter(a[2],d,'plant_height_cm','canopy_width_cm','Visible height (cm)','Maximum canopy span (cm)',legend=True)
    identity(a[3],d[['canopy_span_a_cm','canopy_span_b_cm']])
    scatter(a[3],d,'canopy_span_a_cm','canopy_span_b_cm','Axis A span (cm)','Axis B span (cm)')
    dots(a[4],d,'canopy_height_ratio','Canopy span / height')
    pairs=p.csv(Path('07_PHENOTYPING_EVALUATION/Tomato_Metric_19/Initial_visible_height_comparison.csv'),'f')
    pairs['group']=pairs.cultivar.map(CODES)
    identity(a[5],pairs[['official_cm','prediction_cm']]); scatter(a[5],pairs,'official_cm','prediction_cm',
        'Official true_height (cm)','Visible mesh span (cm)',groups=GROUPS[:3])
    a[5].text(.05,.96,f'{len(pairs)} scans · distinct height definitions',transform=a[5].transAxes,va='top',fontsize=5.3)
    p.dump(d,'whole_plant_metric_dimensions'); p.dump(pairs,'exploratory_tomato_height_pairs')
    p.finish(fig,'Height uses visible extent along the estimated vertical. Potted bases and verticals are provisional. The official tomato height panel is exploratory because the archived reference uses a different height definition.')


def mechanism_front(p):
    panels=['Tomato input preparation','Potted input preparation','Archived tomato and potted examples']
    fig=p.start('M01_RAP_FSAM3','Two routes from RGB to plant masks',
        'Tomato uses the controlled-background capture recipe. Potted plants use RAP-FSAM3.', panels,
        'Shows the front-end routes and real input masks before geometry reconstruction.',shape=None,size=(183,140))
    ax=figure_axes(fig)
    xs=[.005,.253,.501,.749]
    routes=[('a   TomatoPGT',.96,.75,TOMATO,[('Controlled RGB','Multi-view images'),('Foreground prior','Color distance from\nbackground + morphology'),
        ('Masked SfM','Estimate cameras from\nthe scan’s own images'),('SAM3 semantics','Plant, pot and organ\nmasks for 3D association')]),
        ('b   Potted generalization',.68,.47,POTTED_COLOR,[('Multi-view RGB','Varying backgrounds'),('View and prompt bank','Blur screening and\nplant prompt variants'),
        ('RAP mask selection','SG-MPS and CDR\nrank mask candidates'),('RAP refinement','SB-FBPR and guarded\ngeometry correction')])]
    for label,y,t,color,steps in routes:
        section(ax,.006,y,label)
        for i,(name,body) in enumerate(steps):
            tile(ax,xs[i],t,.229,.166,name,body,color)
            if i<3: arrow(ax,xs[i]+.235,t+.080,xs[i+1]-.006,t+.080,color=color)
    section(ax,.006,.375,'c   Archived RGB and input masks')
    for i,(folder,name,label) in enumerate([(CASE,'Original_RGB.png','Tomato RGB'),(CASE,'Input_Mask.png','Tomato input mask'),
        (POTTED_ASSETS/'KongQueZhuYu','Original_RGB.png','Potted RGB'),(POTTED_ASSETS/'KongQueZhuYu','Input_Mask.png','Potted input mask')]):
        method_img(p,fig,ax,(xs[i],.012,.228,.304),folder/name,label,'c')
    p.source(CODE/'ForeSplat_AutoGaussianCleaner/phenotyping/tomatopgt_pilot.py','a','implementation')
    p.source(FRONT/'Method/Recovered_RAP_Chain/01_PAPER_CODE_MAPPING.md','b','implementation')
    p.finish(fig,'The tomato masked-SfM foreground prior and later SAM3 plant masks are distinct. Panel c shows the archived input masks for the selected preview cameras.')


def mechanism_geometry(p):
    panels=['Shared geometry path','Tomato board and potted cube','Tomato and potted geometry intermediates']
    fig=p.start('M02_Metric_GGGS_TSDF','Shared GGGS–TSDF geometry with two metric anchors',
        'The tomato metric subset comprises 19 scans. The generalization cohort comprises 19 potted plants.',
        panels,'Shows image-to-mesh steps, metric anchors and the corresponding archived stages in both cohorts.',shape=None,size=(183,155))
    ax=figure_axes(fig)
    section(ax,.006,.965,'a   Shared geometry path')
    stages=[('COLMAP SfM','Recover cameras\nand sparse points'),('Native GGGS','30K iterations\nSH3, SG0'),
        ('Metric scaling','Apply scale once\nto model + cameras'),('Filtered depth','Mask + alpha filter\nMedian camera depth'),('Bounded TSDF','Fuse depth views\nExtract mesh')]
    for i,(label,body) in enumerate(stages):
        x=.006+i*.199
        tile(ax,x,.748,.180,.170,label,body,TOMATO)
        if i<4: arrow(ax,x+.185,.828,x+.193,.828)
    section(ax,.006,.679,'b   Physical scale recovery')
    tile(ax,.006,.468,.305,.174,'TomatoPGT board','Triangulate known board baselines.\nMetric subset: 19/19 scans.',TOMATO)
    tile(ax,.349,.468,.302,.174,'Centimeters','Convert lengths and camera\ntranslations with one scale.',ACCENT)
    tile(ax,.690,.468,.302,.174,'Potted 2 cm cube','Fit cube in multiple views.\nPotted cohort: 19/19 plants.',POTTED_COLOR)
    arrow(ax,.319,.552,.341,.552); arrow(ax,.680,.552,.659,.552,color=POTTED_COLOR); arrow(ax,.500,.649,.500,.739)
    section(ax,.006,.408,'c   Archived geometry at the selected preview cameras')
    stages=['GGGS_Render.png','Rendered_Depth.png','TSDF_Mesh_Render.png','Plant_Only_Mesh_Render.png']
    for i,label in enumerate(['GGGS render','Cached depth','TSDF mesh','Plant only']):
        ax.text(.213+i*.225,.355,label,ha='center',va='center',size=6.6,color=INK)
    for y,folder,label,color in [(.190,CASE,'Tomato',TOMATO),(.018,PANEL/'TSDF_Mesh/KongQueZhuYu','Potted',POTTED_COLOR)]:
        ax.text(.005,y+.067,label,va='center',weight='bold',size=7,color=color)
        for i,name in enumerate(stages):
            method_img(p,fig,ax,(.112+i*.225,y,.200,.137),folder/name,None,'c')
        p.source(folder/'Depth_Display_Protocol.json','c','depth display provenance')
    for filename in ['batch_mesh.py','batch_calibrate.py','tomatopgt_targets.py','auto_metric_scale.py']:
        p.source(CODE/'ForeSplat_AutoGaussianCleaner/phenotyping'/filename,'a,b','implementation')
    p.table('Tomato_Metric_19_Reconstruction','b',['sample_id','cm_per_unit'])
    p.finish(fig,'Within each row the panels share the archived preview camera. Depth colors follow the separate display protocols. Metric counts use the respective measurement cohorts.')


def independent(ax,p,path,panel):
    xyz,faces,rgb=read_triangle_ply(p.source(path,panel,'archived organ mesh'))
    xyz=xyz-xyz.mean(axis=0)
    _,basis=np.linalg.eigh(np.cov(xyz,rowvar=False))
    basis=basis[:,::-1]
    if np.linalg.det(basis)<0:
        basis[:,2]*=-1
    xyz=xyz@basis
    # Rasterize only the surface to avoid hairline seams between PDF triangles.
    collection=Poly3DCollection(xyz[faces],facecolors=rgb[faces].mean(axis=1),
                               linewidth=0,antialiaseds=False,rasterized=True)
    if not hasattr(ax,'add_collection3d'):
        ax.set_axis_off()
        ax=ax.inset_axes([.03,.03,.94,.94],projection='3d')
    ax.add_collection3d(collection)
    ax.auto_scale_xyz(*xyz.T)
    ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),.01),zoom=1.22)
    ax.view_init(elev=88,azim=-82)
    ax.set_proj_type('ortho')
    ax.set_axis_off()


def mechanism_traits(p):
    panels=['Shared multiview association','Tomato compound leaves','Potted numbered leaves']
    fig=p.start('M03_Organ_and_traits','Shared organ extraction and cohort-specific traits',
        'Surface measurements follow the available organ references in each cohort.',panels,
        'Shows the shared 3D association, paired with examples and measurement definitions.',shape=None,size=(183,138))
    ax=figure_axes(fig)
    section(ax,.006,.965,'a   From image evidence to an independent organ mesh')
    steps=[('SAM3 per view','Plant, pot and\norgan proposals'),('Camera projection','Visible pixels point\nto mesh faces'),
           ('Fuse across views','Separate semantics\nand group organ support'),('Independent meshes','Export each visible\norgan as a PLY')]
    for i,(label,body) in enumerate(steps):
        x=.006+i*.252
        tile(ax,x,.748,.233,.170,label,body,TOMATO)
        if i<3: arrow(ax,x+.238,.828,x+.246,.828)
    route_rows=[('b   Tomato compound leaves',.655,.432,.397,CASES[2],[('Chord','Euclidean base to tip'),('Width','Same transverse section'),('Reference','Official paired landmarks')],TOMATO),
                ('c   Potted numbered leaves',.295,.072,.037,CASES[-1],[('Identity','Photo number to mesh'),('Length candidate','Path along the mesh'),('Area','Observed single side')],POTTED_COLOR)]
    for key,y,t,img_y,case,data,color in route_rows:
        section(ax,.006,y,key)
        left,bottom,width,height=ax.get_position().bounds
        ref=fig.add_axes([left+.006*width,bottom+img_y*height,.192*width,.205*height])
        organ_reference(ref,p,case,key[0])
        obj=fig.add_axes([left+.204*width,bottom+img_y*height,.200*width,.205*height],projection='3d')
        independent(obj,p,case.asset/'leaf_cm.ply',key[0])
        for i,(label,body) in enumerate(data):
            tile(ax,.418+i*.195,t,.182,.151,label,body,color)
    p.source(CODE/'Entrypoints/pair_tomatopgt_compounds.py','b','matching and reference implementation')
    p.source(CODE/'ForeSplat_AutoGaussianCleaner/phenotyping/measure_phenotypes.py','c','measurement implementation')
    p.finish(fig,'The example pairs show selected organ overlays and the matching centimeter meshes. The tomato example is Celebrity Hybrid Compound_010; the potted example is the peacock plant Leaf_002. Meshes are independently fitted for display.')


def case_plate(p):
    labels=['Input RGB','GGGS render','Metric TSDF','Plant only','Selected organ','Independent organ']
    fig=p.start('C01_Tomato_and_generalization','Tomato cultivars and potted generalization cases',
        'Three tomato cultivars and two representative potted plants, with matching pipeline stages.',labels,
        'Shows the same sequence from capture to independent organ mesh in both cohorts.',shape=None,size=(183,155))
    files=['rgb.png','gaussian.png','mesh.png','plant_only.png']
    x0,step,column_w=.164,.134,.127
    for i,label in enumerate(labels):
        fig.text(x0+i*step+column_w/2,.963,label,ha='center',va='center',size=7,weight='medium')
    for row,case in enumerate(CASES):
        letter=PANEL_IDS[row]
        color=TOMATO if row<3 else POTTED_COLOR
        bottom=.778-row*.170
        preview=json.loads(p.source(case.scene/'preview_view.json',letter,'common scene camera provenance').read_text())
        fig.text(.041,bottom+.118,letter,size=7,color=color,weight='bold',va='top')
        label=case.name.replace(' Hybrid','\nHybrid')
        fig.text(.058,bottom+.118,label,size=6.7,color=color,weight='bold',va='top',linespacing=1.1)
        fig.text(.058,bottom+.058,case.sample,size=5.4,color=MUTED)
        fig.text(.058,bottom+.034,str(preview['view']),size=5.4,color=MUTED)
        for col,file in enumerate(files):
            img(p,fig,[x0+col*step,bottom,column_w,.154],case.scene/file,None,letter)
        ref=fig.add_axes([x0+4*step,bottom,column_w,.154])
        organ_reference(ref,p,case,letter)
        ax=fig.add_axes([x0+5*step,bottom+.034,column_w,.113],projection='3d')
        independent(ax,p,case.asset/'leaf_cm.ply',letter)
        fig.text(x0+5*step+column_w/2,bottom+.004,case.organ,ha='center',va='bottom',size=5.5,color=MUTED)
        if row==2:
            fig.add_artist(Line2D([.045,.965],[bottom-.010]*2,transform=fig.transFigure,lw=.9,color=RULE))
    selected=[]
    for row,case in enumerate(CASES):
        meta=organ_meta(p,case,PANEL_IDS[row]); mesh=case.asset/'leaf_cm.ply'
        selected.append(dict(plant=case.sample,organ=case.organ,handwritten=case.handwritten,
                             organ_view=meta['reference_view'],faces=meta['faces'],
                             largest_component_fraction=meta['largest_component_fraction'],
                             mesh_path=str(mesh),organ_overlay=str(case.asset/'reference.png')))
    p.dump(pd.DataFrame(selected),'selected_organs')
    p.finish(fig,'The first four columns share one archived camera per row. The selected organ overlay uses that organ’s own reference camera and a crop around its mask. The last column shows the unedited centimeter mesh at its own magnification. Examples were selected for clear visible morphology; whole-leaf completeness was not inferred.')


def sample_atlas(p):
    p.key='Sample_Atlas_61'
    d = cohort(p,'all sample rows')
    d['order'] = d.group.map(dict(zip(GROUPS,range(len(GROUPS)))))
    d = d.sort_values(['order','sample_id'])
    assert len(d)==d.sample_id.nunique()==61
    files = ['rgb.png','gaussian.png','mesh.png','plant_only.png','instances.png']
    headers = ['Input RGB','GGGS render','TSDF mesh','Plant only','Organ overlay']
    rows_per_page = 8
    p.out.joinpath('Source_Data',p.key).mkdir(parents=True,exist_ok=True)
    d = d.assign(page=np.arange(len(d))//rows_per_page+1)
    p.dump(d[['page','sample_id','group','metric_mesh','preview_rgb_archive']],'atlas_sample_index')
    with PdfPages(p.out/'Sample_Atlas_61.pdf') as pdf:
        for _, rows in d.groupby('page',sort=True):
            fig = plt.figure(figsize=(183/25.4,210/25.4))
            left, step, width = .157,.163,.151
            for i,label in enumerate(headers):
                fig.text(left+i*step+width/2,.967,label,ha='center',size=7,weight='medium')
            for row_index,row in enumerate(rows.itertuples()):
                bottom = .836-row_index*.113
                color = POTTED_COLOR if row.group=='Potted' else TOMATO
                fig.text(.045,bottom+.070,row.group,color=color,weight='bold',size=6.7)
                fig.text(.045,bottom+.045,row.sample_id,size=5.6)
                fig.text(.045,bottom+.023,'cm' if row.metric_mesh else 'SfM units',color=MUTED,size=5.4)
                directory = Path(row.preview_rgb_archive).parent
                for i,file in enumerate(files):
                    path=p.source(directory/file,row.sample_id,f'common camera: {headers[i]}')
                    ax=fig.add_axes([left+i*step,bottom,width,.100])
                    with Image.open(path) as image:
                        image.thumbnail((440,440))
                        ax.imshow(image.convert('RGB'),interpolation='antialiased')
                    ax.set_axis_off()
            pdf.savefig(fig,dpi=300)
            plt.close(fig)
    print(p.key,flush=True)


FIGURES=(mechanism_front,mechanism_geometry,mechanism_traits,case_plate,capture,front,reconstruction,
         semantics,calibration,organs,matching,measurements,trait_agreement,whole_plant)
NAMES=['番茄与盆栽的前端分割路径','公制 GGGS–TSDF 与两种尺度参照','器官分割、身份与性状量测',
       '三种番茄与两种盆栽同流程案例','图像输入与配准覆盖','前景分割证据','重建完成度与网格规模',
       '植物与花盆分离','物理尺度恢复与质控','叶片实例提取','器官身份对应','叶片性状覆盖与观测面积',
       '番茄叶长叶宽参考一致性及盆栽量测','可见株高与冠幅']


def write_index(p):
    pd.DataFrame(p.sources).to_csv(p.out/'Figure_Source_Manifest.csv',index=False)
    pd.DataFrame(p.index).assign(panels=lambda df:df.panels.map(' | '.join)).to_csv(p.out/'Figure_Index.csv',index=False)
    lines=['# ForeSplat 论文图件 · 番茄主线与跨物种泛化', '',
           '新版含 3 张机制图、1 张案例图和 10 张六面板分析图。图题、范围和图注集中在本文件，图内保留面板字母、数据标签和坐标。', '',
           '番茄为主研究队列。重建统计纳入全部 42 个番茄扫描和 19 株盆栽。番茄物理尺度恢复按公制表型子集计为 19/19，盆栽计为 19/19。扫描是采集观察单位。', '',
           '共享指标按番茄的 Beefsteak（BFS）、Big Boy Hybrid（BBH）、Celebrity Hybrid（CH）和盆栽（Potted）排列。参考协议不同的指标使用各自面板。孔雀竹芋的五种 Gaussian 和旧网格选型保存在上一版图包。', '',
           '[图件总览](Contact_Sheet.jpg) · [61 个逐样本重建图册](Sample_Atlas_61.pdf) · [图件索引](Figure_Index.csv) · [来源与面板对应](Figure_Source_Manifest.csv)', '',
           '逐样本图册按番茄三品种、盆栽排列，每个样本同一行展示 RGB、GGGS、TSDF、植物网格和器官网格投影。厘米和 SfM 单位按每行标注。', '',
           '案例器官来自连续性较好的既有实例。见 [案例器官与原始厘米 PLY 对照表](Source_Data/C01_Tomato_and_generalization/selected_organs.csv)。局部高亮图按原始实例掩膜裁切，三维图只调整视角和显示大小；未平滑顶点、补洞或删除碎片。全样本分析不因示例替换而筛除数据。', '',
           '每张分析图在 `Source_Data/` 下附逐样本或逐记录的数据。散点的小幅水平抖动仅用于区分重复位置，深色横线表示中位数。', '']
    for item,label in zip(p.index,NAMES,strict=True):
        key=item['id']
        lines+= [f'## {key.split("_")[0]} · {label}', '',
                 f'[{key}.pdf]({key}.pdf) · [SVG]({key}.svg) · [PNG]({key}.png)', '',
                 item['scope'], '',
                 ' · '.join(f'**{letter}** {name}' for letter,name in zip(PANEL_IDS,item['panels'])), '',
                 item['note'], '']
    lines += ['## 重绘', '', '在证据包的绘图环境中，从当前图件目录执行。', '',
              '```bash','python render_foresplat_tomato_first.py --evidence .. --output ../12_PAPER_FIGURES_TOMATO_FIRST_rebuild','```','']
    (p.out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ('render_foresplat_tomato_first.py','render_foresplat_paper_figures.py'):
        script=Path(__file__).with_name(name).resolve()
        if script != p.out/name:
            shutil.copy2(script,p.out/name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    p=Plates(args.evidence.resolve(),args.output.resolve(),args.dpi)
    for figure in FIGURES:
        figure(p)
    assert len(p.index)==14 and all(len(row['panels'])==6 for row in p.index if row['id'].startswith('A'))
    sample_atlas(p)
    write_index(p)
    review_sheet(p)
    bundle(p)


if __name__=='__main__':
    main()
