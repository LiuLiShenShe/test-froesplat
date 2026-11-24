#!/usr/bin/env python3
"""Assemble the print figures from archived ForeSplat evidence and metric PLYs."""

import argparse
import json
import shutil
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
from PIL import Image

from render_foresplat_paper_figures import CASE, CHORD, WIDTH, PANEL, POTTED, REP, read_triangle_ply
from render_foresplat_tomato_first import (
    CASES, CODES, GROUPS, SCAN_REPORTS, TOMATO_PHENO, Plates, metric_cohort, organ_meta,
)
import matplotlib.pyplot as plt


INK, MUTED, RULE, PALE = '#1A262C', '#45515B', '#D8DFE2', '#F3F6F7'
COLORS = ['#0072B2', '#E69F00', '#009E73', '#805476']
MARKS = ['o', '^', 's', 'D']
NAMES = ['Beefsteak', 'Big Boy Hybrid', 'Celebrity Hybrid', 'Potted plants']
ROOT = Path(__file__).resolve().parents[1]/'ForeSplat_Paper_Evidence_Package'
EXPORT_DPI = 600
DOUBLE_COLUMN_MM = 190
KQ = PANEL/'TSDF_Mesh/KongQueZhuYu'
PAPER = Path('01_MANUSCRIPT/Current_Paper')
GOOD = 'included_reference_guided_landmarks'


def figure(p, key, size, panels, claim):
    return p.start(key, key, '', panels, claim, shape=None, size=size)


def finish(p, fig, caption, *, axes=None, rows=None, cols=None, exemptions=()):
    from audit_panel_alignment import require_matplotlib_panel_alignment

    qa = p.out/'QA'
    qa.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(
        fig, axes=axes, row_groups=rows, column_groups=cols, exemptions=exemptions,
        json_out=qa/f'{p.key}_alignment.json',
    )
    p.index[-1]['note']=caption
    fig.savefig(p.out/f'{p.key}.pdf',dpi=p.dpi)
    fig.savefig(p.out/f'{p.key}.svg',dpi=p.dpi)
    fig.savefig(p.out/f'{p.key}.png',dpi=p.dpi)
    plt.close(fig)
    print(p.key,flush=True)


def band(fig, label, title, y, x=.04):
    fig.text(x, y, f'({label})', weight='bold', fontsize=9, ha='left', va='top')
    fig.text(x+.045*190/(25.4*fig.get_figwidth()), y, title, weight='bold', fontsize=8.4,
             ha='left', va='top')


def panel(ax, label, title, *, pad=6):
    ax.annotate(f'({label})',(0,1),xycoords='axes fraction',xytext=(0,pad),textcoords='offset points',
                fontweight='bold',fontsize=9,va='baseline')
    ax.annotate(title,(0,1),xycoords='axes fraction',xytext=(17,pad),textcoords='offset points',
                fontsize=8.2,va='baseline')


def symbol(i, label=None):
    return Line2D([], [], color=COLORS[i], marker=MARKS[i], linestyle='none', markersize=5,
                  label=label or NAMES[i])


def legend(fig, indices=range(4)):
    fig.legend(handles=[symbol(i) for i in indices], loc='upper right', bbox_to_anchor=(.97, .992),
               ncol=len(indices), frameon=False, handletextpad=.35, columnspacing=1.)


def photo(ax, p, path, panel_id, crop=None, *, pad=True):
    source = p.source(path, panel_id, f'RGB or mask; crop={crop}' if crop else 'RGB or mask')
    with Image.open(source) as image:
        pixels = np.asarray((image.crop(crop) if crop else image).convert('RGB'))
    bitmap(ax, pixels, pad=pad)


def bitmap(ax, pixels, *, pad=True):
    if pad:
        h, w = pixels.shape[:2]
        ax.imshow(pixels, extent=(-w/2, w/2, -h/2, h/2), aspect='auto', interpolation='antialiased')
        box = ax.get_position()
        ratio = (box.width * ax.figure.get_figwidth())/(box.height * ax.figure.get_figheight())
        x = max(w, h*ratio)
        ax.set(xlim=(-x/2, x/2), ylim=(-x/(2*ratio), x/(2*ratio)))
    else:
        ax.imshow(pixels, aspect='auto', interpolation='antialiased')
    ax.set_axis_off()


def highlight(ax, p, case, panel_id, color):
    meta = organ_meta(p, case, panel_id)
    name = meta['reference_view']
    rgb = p.source(case.asset/'reference_rgb.png', panel_id, 'organ reference RGB')
    mask = p.source(case.asset/'masks'/f'{name}.png', panel_id, 'binary mask for 38% tint')
    with Image.open(rgb) as im, Image.open(mask) as ma:
        loc = ma.convert('L').getbbox()
        assert loc and im.size == ma.size
        x, y, x2, y2 = loc
        marg = .19*max(x2-x, y2-y)
        box = (max(0, int(x-marg)), max(0, int(y-marg)), min(im.width, round(x2+marg)),
               min(im.height, round(y2+marg)))
        rgb = np.asarray(im.crop(box).convert('RGB'))/255.
        inside = np.asarray(ma.crop(box).convert('L')) > 0
    p.sources.append(dict(figure=p.key, panel=panel_id, role=f'RGB and mask crop xyxy={box}',
                          path=str(case.asset/'reference_rgb.png')))
    rgb[inside] = .62*rgb[inside]+.38*np.array(mpl.colors.to_rgb(color))
    bitmap(ax, rgb)


@lru_cache(maxsize=10)
def mesh(path):
    return read_triangle_ply(path)


def camera(axes=np.eye(3), azim=32, elev=24):
    a, e = np.deg2rad([azim, elev])
    return np.array([[-np.sin(a), np.cos(a), 0],
                     [-np.sin(e)*np.cos(a), -np.sin(e)*np.sin(a), np.cos(e)],
                     [np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]]) @ axes


def frame(p, case, panel_id):
    directory = ('Potted_Plants_2cm_Cube' if case.handwritten else 'Tomato_Target_Board')
    path = Path('04_METRIC_CALIBRATION')/directory/case.sample/'reference_frame.json'
    ref = json.loads(p.source(path, panel_id, 'estimated vertical and horizontal axes').read_text())
    base = np.array([ref[k] for k in ('horizontal_axis_a', 'horizontal_axis_b', 'vertical_unit_vector')])
    np.testing.assert_allclose(base @ base.T, np.eye(3), atol=.0001)
    return camera(base)


def scale_bar(ax, sx, sy, xlim, ylim):
    target = min(sx, sy)*.24
    assert target>0
    exponent = np.floor(np.log10(target))
    sizes = np.array([.5, 1, 2, 5, 10])*10**exponent
    bar = sizes[sizes <= target][-1]
    x, y = xlim[0]+.06*sx, ylim[0]+.05*sy
    ax.plot((x, x+bar), (y, y), color=INK, lw=1.4, solid_capstyle='butt')
    ax.annotate(f'{bar:g} cm', (x, y), xytext=(0, 1.5), textcoords='offset points',
                ha='left', va='bottom')


def draw_mesh(ax, p, rel, panel_id, view=None, *, window=None):
    data = p.source(rel, panel_id, 'unchanged centimeter mesh; orthographic rendering')
    xyz, faces, colors = mesh(data)
    centered = xyz-xyz.mean(axis=0) if view is None else xyz
    if view is None:
        eigen = np.linalg.eigh(centered.T@centered)[1]
        view = camera(eigen[:, [0, 2, 1]].T)
    xyz = centered @ view.T
    tri = xyz[faces]
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)
    light = np.array([-.28, .38, .88])
    shade = .69+.31*np.abs(normal@light)
    rgb = colors[faces].mean(axis=1)*shade[:, None]
    idx = np.argsort(tri[:, :, 2].mean(axis=1), kind='stable')
    surface=PolyCollection(tri[idx, :, :2],facecolors=rgb[idx],edgecolors='none',
                           antialiased=False,rasterized=True)
    ax.add_collection(surface)
    min_xy, max_xy = (xyz[:, :2].min(0), xyz[:, :2].max(0)) if window is None else window
    mid = (min_xy+max_xy)/2
    w, h = (max_xy-min_xy)*1.14
    box = ax.get_position()
    ratio = box.width*ax.figure.get_figwidth()/(box.height*ax.figure.get_figheight())
    sx = max(w, h*ratio/.74)
    sy = sx/ratio
    xlim = mid[0]+sx*np.array([-.5,.5])
    ylim = mid[1]+sy*np.array([-.62,.38])
    ax.set(xlim=xlim, ylim=ylim, aspect='equal')
    surface.set_clip_path(Rectangle((xlim[0],ylim[0]+.22*sy),sx,.78*sy,transform=ax.transData))
    ax.set_axis_off()
    scale_bar(ax, sx, sy, xlim, ylim)
    return min_xy, max_xy


def mesh_paths(p, case):
    name = p.root/'06_PLANT_AND_ORGAN_SEGMENTATION/Plant_Only'
    semantic = name/('Potted' if case.handwritten else 'TomatoPGT')/case.sample
    if not semantic.exists():
        semantic = name/('Potted' if case.handwritten else 'Tomato')/case.sample
    scene = Path('05_TSDF_MESH_RECONSTRUCTION')/('Potted_Meshes' if case.handwritten else 'Tomato_Meshes')
    return (scene/case.sample/'mesh_cm.ply', (semantic/'plant_only_cm.ply').relative_to(p.root),
            (semantic/'pot_cm.ply').relative_to(p.root))


def scene_window(p, plant, pot, view, panel_id):
    points=[]
    for part in (plant,pot):
        if (p.root/part).is_file():
            path=p.source(part,panel_id,'semantic-mesh bounds for common display viewport')
            xyz,faces,_=mesh(path)
            if len(faces):
                points.append(xyz @ view[:2].T)
    return np.quantile(np.concatenate(points),[.002,.998],axis=0)


def depth(ax, cax, p, folder, panel_id):
    x = p.source(folder/'Cached_Depth_K_c2w_RGB.npz', panel_id, 'cached centimeter depth')
    meta = json.loads(p.source(folder/'Depth_Display_Protocol.json', panel_id, 'display range').read_text())
    with np.load(x) as archive:
        d = archive['depth']
    low, high = (meta[f'display_quantile_{i}'] for i in (2, 98))
    arr = np.ma.masked_where(~np.isfinite(d) | (d <= 0), d)
    im = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(low,high), cmap='viridis')
    bitmap(ax,im.to_rgba(arr)[:,:,:3]+np.ma.getmaskarray(arr)[:,:,None])
    ticks = [int(np.ceil(low)), int(np.floor(high))]
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
    cb.set_label('Depth (cm)', labelpad=0)
    cb.ax.tick_params(length=1.5, pad=2)


def link(ax, start, end, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8, lw=.8,
                                facecolor=color, edgecolor=color, shrinkA=3, shrinkB=3,
                                transform=ax.transAxes, clip_on=False))


def step(ax, x, y, w, h, words, fill=PALE, edge=RULE):
    ax.add_patch(Rectangle((x,y),w,h,transform=ax.transAxes,facecolor=fill,edgecolor=edge,lw=.65))
    ax.text(x+w/2,y+h/2,words,transform=ax.transAxes,ha='center',va='center',linespacing=1.25)


def abox(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    return ax


def overview(p):
    fig = figure(p,'Fig1_Overall_framework',(190,101),
                 ['Foreground priors','Metric reconstruction','Organ association and traits'],
                 'Distinct foreground routes feed shared metric geometry and organ measurement.')
    xs = [.04,.353,.666]
    heads = ['Foreground priors','Metric reconstruction','Organ association and traits']
    for i,x in enumerate(xs):
        band(fig,'abc'[i],heads[i],.933,x)
    panels = [abox(fig,[x,.13,.287,.72]) for x in xs]
    for ax in panels[:2]:
        link(ax,(1.005,.57),(1.080,.57))
    for r,(folder,label) in enumerate([(CASE,'TomatoPGT colour-distance prior'),(KQ,'Potted RAP-FSAM3 prior')]):
        y=.537-r*.370
        fig.text(xs[0],y+.248,label,fontsize=7.5,weight='medium')
        for c,(f,name) in enumerate([('Original_RGB.png','RGB'),('Input_Mask.png','Foreground mask')]):
            x=xs[0]+.139*c
            photo(abox(fig,[x,y+.042,.129,.176]),p,folder/f,'a')
            fig.text(x+.0645,y,name,ha='center')
    b=panels[1]
    for i,name in enumerate(['COLMAP','Native GGGS','Scale once','Bounded TSDF']):
        y=.88-.25*i
        step(b,.00,y,.39,.112,name)
        if i<3:
            link(b,(.195,y),(.195,y-.138))
    photo(abox(fig,[xs[1]+.129,.591,.154,.207]),p,CASE/'GGGS_Render.png','b')
    photo(abox(fig,[xs[1]+.129,.415,.154,.138]),p,CASE/'Original_RGB.png','b',crop=(200,480,1290,948))
    fig.text(xs[1]+.206,.386,'Calibration board',ha='center')
    case=CASES[0]
    view=frame(p,case,'b')
    scene,plant,pot=mesh_paths(p,case)
    draw_mesh(abox(fig,[xs[1]+.131,.105,.149,.245]),p,scene,'b',view,window=scene_window(p,plant,pot,view,'b'))
    c=panels[2]
    step(c,.00,.88,.99,.112,'SAM3 masks + visible triangle votes')
    link(c,(.495,.88),(.495,.756))
    highlight(abox(fig,[xs[2],.318,.133,.333]),p,case,'c',COLORS[0])
    draw_mesh(abox(fig,[xs[2]+.147,.318,.139,.333]),p,case.asset/'leaf_cm.ply','c')
    fig.text(xs[2]+.0665,.292,'Matched view',ha='center')
    fig.text(xs[2]+.2165,.292,'Extracted organ',ha='center')
    step(c,.00,.035,.99,.121,'Canopy spans | Leaf chord, width, area')
    link(c,(.495,.19),(.495,.156))
    finish(p,fig,'(a) Controlled-background tomato foreground masks support independent masked SfM; RAP-FSAM3 supplies foreground priors for the potted cohort. (b) Independent COLMAP cameras initialise GGGS. Board baselines (tomato) or a 2 cm cube (potted) recover scale before screened GGGS depth fusion. (c) Separate SAM3 semantic and instance masks vote on visible triangles; exported organ surfaces link back to the source images. One tomato exemplar is shown in (b,c).',
           axes=panels, rows=[['a','b','c']])


def foreground(p):
    fig=figure(p,'Fig2_Foreground_comparisons',(190,157),
               ['VFM comparison','Prompt sensitivity'],
               'Matched potted views show alternative foreground predictions and prompt sensitivity.')
    cases=[('KongQueZhuYu_0075','Peacock plant'),('ChangShouHua2_0000','Kalanchoe'),
           ('CaoMei1_0000','Strawberry')]
    sets=[('Ground truth',None),('CLIPSeg','CLIPSeg_potted'),('Grounded SAM','GroundedSAM1_potted'),
          ('Single-prompt SAM3','SAM3_P2'),('RAP-FSAM3','RAP-FSAM3-v2')]
    band(fig,'a','Foreground methods',.975)
    grid=[]
    for i,(stem,row) in enumerate(cases):
        y=.783-.139*i
        fig.text(.042,y+.068,row,va='center',ha='left')
        cells=[]
        for j,(title,method) in enumerate(sets):
            x=.144+.165*j
            if i==0:
                fig.text(x+.077,.912,title,ha='center')
            file=(REP/'overlays'/method/f'{stem}_{method}_overlay.png' if method else
                  REP/'gt_overlays'/f'{stem}_gt_overlay.png')
            with Image.open(p.root/file) as original:
                w,h=original.size
            ax=abox(fig,[x,y,.154,.122]); photo(ax,p,file,'a',(0,72,w,h)); cells.append(ax)
        grid.append(cells)
    band(fig,'b','Single-prompt sensitivity',.469)
    # Reuse the original matched panel slices (no title or baked-in annotation text).
    columns=[(319,535,'Green region'),(536,752,'Plant instance'),(753,968,'Organ level'),
             (970,1186,'Seedling'),(1187,1402,'Exclude background')]
    for i,(y0,y1,row) in enumerate([(96,314,'Peperomia'),(316,534,'Kalanchoe'),(536,754,'Strawberry')]):
        y=.287-i*.123
        fig.text(.042,y+.051,row,va='center')
        cells=[]
        for j,(x0,x1,title) in enumerate(columns):
            x=.144+.165*j
            if i==0:
                fig.text(x+.077,.408,title,ha='center')
            crop=(x0+2,y0+2,x1-2,y1-2)
            ax=abox(fig,[x,y,.154,.102]); photo(ax,p,PAPER/'Fig4.png','b',crop); cells.append(ax)
        grid.append(cells)
    finish(p,fig,'(a) Representative4 annotated overlay comparisons on identical RGB frames within each row. Masks use the original archived overlay colours; the reference includes visible potted-plant support under that annotation convention. (b) Cropped, unaltered mask tiles from the archived prompt-sensitivity figure. The direct SAM3 prompts were “green plant”, “entire potted plant”, “leaves and stems”, “crop seedling” and “plant body without background”. This is qualitative potted-cohort evidence, not a RAP-FSAM3 score on TomatoPGT. Frame counts and numerical benchmarks remain in their protocol-specific source tables.',
           axes=sum(grid,[]), rows=rows_for(grid),
           cols=[group[i:i+3] for group in cols_for(grid) for i in (0,3)])


def rows_for(grid):
    ids=[chr(97+i) if i<26 else 'a'+chr(97+i-26) for i in range(sum(map(len,grid)))]
    return [ids[i:i+len(grid[0])] for i in range(0,len(ids),len(grid[0]))]


def cols_for(grid):
    return [list(row) for row in zip(*rows_for(grid))]


def flow(p):
    fig=figure(p,'Fig3_Sample_flow',(190,158),['Scan workflow','Compound identity','Trait pair QC'],
               'The measurement subsets narrow at distinct stages and use distinct denominators.')
    rec,cal,metric,pairs,plants,leaves=(p.table(name,'a–c') for name in (
        'Tomato_Metric_19_Reconstruction','Tomato_42_Calibration','Tomato_70_Metric_Compounds',
        'Tomato_Leaf_Trait_Pairs','Potted_19_Reconstruction',
        'Potted_57_Leaf_Matching'))
    all_mesh=rec.tsdf_mesh_archive.map(lambda path:(p.root/path).is_file()).sum()
    plant_mesh=plants.tsdf_mesh_archive.map(lambda path:(p.root/path).is_file()).sum()
    assert len(rec)==all_mesh==len(plants)==plant_mesh==19
    assert rec.metric_mesh.all() and plants.metric_mesh.all()
    cal=cal[cal.sample_id.isin(rec.sample_id)]
    assert set(rec.scene)==set(cal[cal.status.str.startswith('CALIBRATED')].sample_id)
    assert set(metric.scene)<=set(rec.scene) and metric.metric_scan.all() and len(metric)==70
    assert set(pairs.scene)<=set(rec.scene)
    assert int(leaves.human_confirmed.sum())==57
    m=metric.status.value_counts()
    candidates=int(metric.candidate_id.notna().sum())
    assert candidates==m.strong_suggestion+m.review_suggestion+m.ambiguous_assignment
    board_published=int(cal.method.str.startswith('Published').sum())
    cube_auto=int(plants.calibration_mode.str.startswith('AUTOMATIC').sum())
    panels=[]
    for i,(y,h,heading) in enumerate([(.702,.248,'Reconstruction and scale | scans and plants'),
                                      (.393,.271,'Tomato reference correspondence | compound records'),
                                      (.032,.323,'Tomato trait QC | 70 metric records per trait')]):
        band(fig,'abc'[i],heading,y+h+.032)
        panels.append(abox(fig,[.04,y,.92,h]))
    a,bc,c=panels
    step(a,.0,.56,.20,.265,f'TomatoPGT\n{len(rec)} scale-qualified scans')
    step(a,.26,.56,.25,.265,f'COLMAP + GGGS + TSDF\n{all_mesh}/{len(rec)} meshes')
    step(a,.61,.56,.39,.265,f'Target board: {len(rec)}/{len(rec)} metric meshes\n{len(cal)-board_published} transferred + {board_published} published')
    link(a,(.20,.692),(.26,.692)); link(a,(.51,.692),(.61,.692))
    step(a,.0,.14,.20,.265,f'Potted plants\n{len(plants)} plants')
    step(a,.26,.14,.25,.265,f'COLMAP + GGGS + TSDF\n{plant_mesh}/{len(plants)} meshes')
    step(a,.61,.14,.39,.265,f'2 cm cube: {plant_mesh}/{len(plants)} metric meshes\n{cube_auto} automatic + {len(plants)-cube_auto} assisted')
    link(a,(.20,.272),(.26,.272)); link(a,(.51,.272),(.61,.272))
    step(bc,.0,.50,.22,.32,f'{len(metric)} official\ncompound records')
    step(bc,.315,.56,.25,.26,f'{candidates} with candidate')
    step(bc,.315,.19,.25,.26,f'{m.no_reliable_match} no reliable match',fill='white')
    step(bc,.67,.64,.33,.24,f'{m.strong_suggestion} strong suggestions')
    step(bc,.67,.32,.33,.24,f'{m.review_suggestion} review suggestions\n{m.ambiguous_assignment} ambiguous assignments',fill='white')
    link(bc,(.22,.66),(.315,.69)); link(bc,(.22,.57),(.315,.32))
    link(bc,(.565,.72),(.67,.76)); link(bc,(.565,.64),(.67,.44))
    bc.text(.0,.065,f'Potted: {leaves.human_confirmed.sum()}/{len(leaves)} numbered leaf identities confirmed',transform=bc.transAxes)
    labels={GOOD:'Eligible reference-guided pairs',
            'no_model_landmark_pair_passing_existing_qc':'Model landmark QC',
            'missing_reference_landmarks':'Missing reference landmarks',
            'scale_review':'Pair scale check', 'ambiguous_assignment':'Ambiguous identity',
            'no_matching_mesh':'No matching mesh',
            'no_comparable_one_sided_area_reference':'No comparable area reference'}
    records=pairs.groupby(['trait','status','upstream_match_status']).size().rename('records').reset_index()
    counts=pairs.groupby(['status','trait']).size().unstack(fill_value=0)
    eligible=pairs[pairs.status.eq(GOOD)]
    assert eligible.upstream_match_status.eq('strong_suggestion').all()
    assert counts.sum().eq(len(metric)).all()
    c.text(.0,.88,'Pair outcome',weight='bold',transform=c.transAxes)
    c.text(.53,.88,'Chord',ha='center',weight='bold',transform=c.transAxes)
    c.text(.72,.88,'Width',ha='center',weight='bold',transform=c.transAxes)
    c.text(.91,.88,'Observed area',ha='center',weight='bold',transform=c.transAxes)
    for i,(k,label) in enumerate(labels.items()):
        y=.757-.104*i
        if i==0:
            c.add_patch(Rectangle((0,y-.035),1,.13,transform=c.transAxes,fc=PALE,ec='none'))
        c.text(.0,y,label,transform=c.transAxes,weight='bold' if i==0 else 'normal')
        for x,trait in zip((.53,.72,.91),(CHORD,WIDTH,'observed_one_sided_leaf_area')):
            n=counts.loc[k,trait]
            text=f'{n}/{len(metric)}' if i==0 else (str(n) if n else '–')
            c.text(x,y,text,ha='center',transform=c.transAxes,weight='bold' if i==0 else 'normal')
    p.dump(records,'trait_exclusions_by_upstream_match')
    p.dump(m.rename('records').rename_axis('status').reset_index(),'reference_match_counts')
    p.dump(pd.DataFrame([['TomatoPGT',len(rec),all_mesh,int(rec.metric_mesh.sum())],
                         ['Potted',len(plants),plant_mesh,int(plants.metric_mesh.sum())]],
                        columns=['cohort','inputs','meshes','metric_meshes']), 'scan_counts')
    p.dump(rec[['sample_id','cultivar','cm_per_unit','tsdf_mesh_archive']],'included_tomato_scans')
    p.dump(metric[['scene','official_point_instance','candidate_id','status','registration','overlap_score',
                   'reference_coverage','prediction_coverage','runner_up_gap','competing_reference_gap',
                   'alignment_only_scale','metric_pair_qc']],'compound_match_dispositions')
    p.dump(pairs[['scene','official_point_instance','candidate_id','trait','status']],'trait_pair_dispositions')
    finish(p,fig,'(a) The analysis includes the 19 TomatoPGT scans that passed physical scale QC and all 19 potted plants. Each included sample has a metric mesh. (b) The tomato scans contribute 70 official compound records. Of these, 69 have a candidate: 40 met the strong match criteria for overlap, coverage and separation from competing identities, 27 met the lower review threshold, and two were ambiguous. One record had no reliable match. The 57 numbered potted leaf identities were human-confirmed and form a separate cohort. (c) Each tomato trait starts from the same 70 records. The pair scale check flags a reference alignment scale more than 20% from unity. The 14 chord and 24 transverse-width pairs are subsets of the 40 strong suggestions. One organ can contribute to both traits. No comparable area reference was available. Fig. 6 uses manual and virtual measurements paired within the historical workbook for the potted trait comparison.',
           axes=panels,cols=[['a','b','c']])


def semantics(p):
    fig=figure(p,'Fig4_Semantics_and_organs',(190,155),
               ['Beefsteak','Big Boy Hybrid','Celebrity Hybrid','Rubber plant','Peacock plant'],
               'The shared semantic and organ workflow exports interpretable surfaces across cohorts.')
    headings=['Input RGB','GGGS','Scene mesh','Plant only','Pot class','Organ mask','Organ mesh']
    xs=.139+np.arange(len(headings))*.118
    w=.113
    legend(fig)
    for x,name in zip(xs,headings):
        fig.text(x+w/2,.914,name,ha='center',weight='medium')
    grid=[]; selection=[]
    metric_ids=set(p.table('Tomato_Metric_19_Reconstruction','a–c',['sample_id']).sample_id)
    for r,case in enumerate(CASES):
        assert case.handwritten or case.sample in metric_ids
        pid='abcde'[r]
        y=.742-r*.172
        color=COLORS[min(r,3)]
        fig.text(.027,y+.127,f'({pid})',ha='left',weight='bold',fontsize=9)
        fig.text(.059,y+.128,case.name.replace(' Hybrid','\nHybrid').replace(' plant','\nplant'),
                 ha='left',va='top',weight='medium')
        fig.add_artist(Line2D([.060,.102],[y+.067]*2,transform=fig.transFigure,color=color,lw=2.5))
        cells=[abox(fig,[x,y,w,.144]) for x in xs]; grid.append(cells)
        for ax,file in zip(cells,('rgb.png','gaussian.png')):
            photo(ax,p,case.scene/file,pid)
        p.source(case.scene/'preview_view.json',pid,'matched RGB and GGGS camera')
        view=frame(p,case,pid)
        scene,plant,pot=mesh_paths(p,case)
        # Render the scene and semantic subsets with identical scale and framing within a row.
        window=scene_window(p,plant,pot,view,pid)
        draw_mesh(cells[2],p,scene,pid,view,window=window)
        draw_mesh(cells[3],p,plant,pid,view,window=window)
        if (p.root/pot).is_file() and len(mesh(p.root/pot)[1]):
            draw_mesh(cells[4],p,pot,pid,view,window=window)
        else:
            cells[4].text(.5,.5,'No pot\nmesh',transform=cells[4].transAxes,ha='center',va='center')
        highlight(cells[5],p,case,pid,color)
        draw_mesh(cells[6],p,case.asset/'leaf_cm.ply',pid)
        meta=organ_meta(p,case,pid)
        selection.append(dict(sample=case.sample,organ=case.organ,handwritten=case.handwritten,
                              reference_view=meta['reference_view'],faces=meta['faces'],
                              largest_component_fraction=meta['largest_component_fraction'],
                              scene=str(scene),plant=str(plant),pot=str(pot) if (p.root/pot).is_file() else '',
                              organ_mesh=str(case.asset/'leaf_cm.ply')))
    p.dump(pd.DataFrame(selection),'selected_case_manifest')
    finish(p,fig,'Representative metric reconstructions: (a) BFS_06262025, Compound_002; (b) BBH_06122025, Compound_009; (c) CH_06102025, Compound_010; (d) XiangPiShu2, numbered leaf 1 (Leaf_002); (e) KongQueZhuYu, numbered leaf 3 (Leaf_002). RGB and GGGS use a matched archived camera. Scene, plant and pot-class meshes use the same orthographic camera and centimetre viewport within each row. Faces assigned to the pot class can include the target board in tomato views. The viewport can crop scene fragments. Organ masks are tinted at 38% opacity in their supporting views. Independent organ meshes have their own view and physical scale bar. Geometry is unmodified.',
           axes=sum(grid,[]),rows=rows_for(grid),cols=cols_for(grid))


def plot_grid(fig, nrows=2, ncols=3, *, left=.105, bottom=.11, right=.965, top=.865,
              wspace=.48, hspace=.55):
    g=fig.add_gridspec(nrows,ncols,left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)
    return [fig.add_subplot(cell) for cell in g]


def marks(ax, data, x, y, *, indices=range(4)):
    for i in indices:
        g=data[data.group.eq(GROUPS[i])]
        ax.scatter(g[x],g[y],color=COLORS[i],marker=MARKS[i],s=21,alpha=.85,edgecolors='white',linewidths=.35)


def strip(ax, data, y, *, indices=range(4), show_n=False):
    for i in indices:
        g=data[data.group.eq(GROUPS[i])][y].dropna()
        off=np.random.default_rng(i).uniform(-.21,.21,len(g))
        ax.scatter(i+off,g,color=COLORS[i],marker=MARKS[i],s=20,alpha=.83,edgecolors='white',linewidths=.32)
        if len(g):
            ax.hlines(g.median(),i-.26,i+.26,color=INK,lw=1.3)
    labs=[f'{GROUPS[i]}\nn={data.loc[data.group.eq(GROUPS[i]),y].count()}' if show_n else GROUPS[i]
          for i in indices]
    ax.set(xticks=list(indices),xticklabels=labs,xlim=(min(indices)-.53,max(indices)+.53))


def nonzero_log(ax, values):
    values=np.asarray(values)
    values=values[np.isfinite(values)]
    assert np.all(values>0)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'))


def quality(p):
    fig=figure(p,'Fig5_Geometry_and_metric_traits',(190,137),
               ['Held-out silhouettes','Scale factors','Calibration residuals','Visible height','Canopy span','Observed area'],
               'Reconstruction diagnostics support the context for descriptive metric traits in both cohorts.')
    a=plot_grid(fig)
    legend(fig)
    m=metric_cohort(p,'a,b,d–e')
    scan=m[m.group.ne('Potted')]
    views=pd.concat([p.csv(SCAN_REPORTS/r.sample_id/'Figures/heldout_silhouettes.csv','a',['view','iou'])
                     .assign(sample_id=r.sample_id,group=r.group)
                     for r in scan.itertuples()],ignore_index=True)
    scores=views.groupby(['sample_id','group'],as_index=False).iou.median()
    assert scores.iou.notna().sum()==len(scan)==19
    panel(a[0],'a','Held-out silhouettes')
    strip(a[0],scores,'iou',indices=range(3),show_n=True)
    a[0].set(ylabel='Median mesh–SAM3 IoU',ylim=(.5,1))
    panel(a[1],'b','Physical scale factors')
    strip(a[1],m,'cm_per_unit',show_n=True)
    a[1].set_ylabel('cm per SfM unit')
    cube=p.table('Potted_19_Calibration','c',['sample_id','cube_reprojection_error_px','boundary_rms_px'])
    cube['group']='Potted'
    qc=cube.melt(id_vars=['sample_id','group'],var_name='metric',value_name='error')
    panel(a[2],'c','Cube fit residuals')
    for j,(k,d) in enumerate(qc.groupby('metric',sort=False)):
        off=np.random.default_rng(j).uniform(-.18,.18,len(d))
        a[2].scatter(j+off,d.error,marker=MARKS[3],s=20,color=COLORS[3],alpha=.78,
                     edgecolors='white',linewidths=.33)
        a[2].hlines(d.error.median(),j-.25,j+.25,color=INK,lw=1.3)
    a[2].set(xticks=[0,1],xticklabels=['Reprojection','Boundary RMS'],xlim=(-.55,1.55),ylabel='Residual (px)')
    fields=['plant_height_cm','canopy_width_cm','canopy_span_a_cm','canopy_span_b_cm']
    plant=pd.concat([p.csv((POTTED if r.group=='Potted' else TOMATO_PHENO)/r.sample_id/'plant_measurements.csv',
                           'd,e',fields).assign(sample_id=r.sample_id,group=r.group)
                     for r in m.itertuples()],ignore_index=True)
    assert len(plant)==38 and plant[fields].notna().all().all()
    for idx,(key,name,unit) in enumerate(zip(fields[:2],['Visible height','Max. canopy span'],['Visible height (cm)','Horizontal span (cm)']),3):
        panel(a[idx],'de'[idx-3],name)
        strip(a[idx],plant,key,show_n=True)
        a[idx].set_ylabel(unit)
    area_key='model_observed_mesh_area_cm2'
    t=p.table('Tomato_70_Metric_Compounds','f',['scene','cultivar','official_point_instance',area_key])
    t=t.rename(columns={area_key:'area_cm2'}).assign(group=t.cultivar.map(CODES))
    h=p.table('Potted_57_Trait_Measurements','f',['plant_id','leaf_id','automatic_leaf_area_cm2'])
    h=h.rename(columns={'automatic_leaf_area_cm2':'area_cm2'}).assign(group='Potted')
    d=pd.concat([t,h],ignore_index=True)
    panel(a[5],'f','Observed organ area')
    strip(a[5],d,'area_cm2',show_n=True)
    nonzero_log(a[5],d.area_cm2)
    a[5].yaxis.set_major_locator(FixedLocator([1,10,100]))
    a[5].set_ylabel('One-sided area (cm²)')
    board=p.table('Tomato_42_Calibration','b',['sample_id','max_relative_baseline_scale_deviation',
                                             'board_reprojection_raw_px','status','method'])
    p.dump(views,'tomato_heldout_silhouette_views'); p.dump(scores,'tomato_heldout_silhouette_scan_medians')
    p.dump(cube,'potted_cube_fit_qc'); p.dump(m.merge(board,on='sample_id',how='left'),'calibrated_models_and_board_qc')
    p.dump(plant,'whole_plant_dimensions'); p.dump(d,'observed_area_records')
    finish(p,fig,'(a) One dot per scale-qualified tomato scan (n = 19): median mesh–SAM3 silhouette IoU on held-out views, a 2D consistency score rather than independently measured 3D surface error. (b) Calibrated samples only; starting SfM units are arbitrary per reconstruction. (c) Two residual definitions for the 19 cube fits. Tomato board QC and scale factors are in the source data, with residuals reported under their own definition. (d,e) Each point is one of 19 metric tomato scans or 19 potted plants. Height is the visible span along the estimated vertical. Canopy span is maximum horizontal extent. (f) Observed area in 69/70 tomato records and 57/57 potted numbered leaves, plotted on a log axis. No matched area reference was available. Black bars are medians throughout. Cohort sample sizes are descriptive and not biological replicate counts.',axes=a)


POTTED_TYPES = {
    'CaoMei': ('草莓', 'Strawberry', '*'),
    'ChangShouHua': ('长寿花', 'Kalanchoe', 'D'),
    'DouBanLv': ('豆苗绿', 'Peperomia', 'P'),
    'HongZhang': ('红掌', 'Anthurium', 'X'),
    'KongQueZhuYu': ('孔雀竹芋', 'Peacock plant', 'v'),
    'WanNianQing': ('万年青', 'Aglaonema', '<'),
    'WangWenCao': ('网纹草', 'Fittonia', '>'),
    'XianKeLai': ('仙客来', 'Cyclamen', 'p'),
    'XiangPiShu': ('橡皮树', 'Rubber plant', 'h'),
}
FIG6_TRAITS = [('height', 'Plant height', '株高'), ('canopy', 'Canopy width', '冠幅'),
               (CHORD, 'Leaf length', '叶长'), (WIDTH, 'Leaf width', '叶宽')]


def historical_potted_pairs(p, cohort):
    path = p.root.parent/'00-论文优化重构/植株数据.xlsx'
    book = pd.read_excel(p.source(path,'a–h','author-selected potted manual/virtual workbook'), sheet_name='Sheet1')
    book['excel_row'] = np.arange(len(book))+2
    expected = cohort[cohort.group.eq('Potted')].plant_type.value_counts()
    candidates = book[book['品种'].isin([POTTED_TYPES[k][0] for k in expected.index])].copy()
    candidates['plant_type'] = candidates['品种'].map({x[0]: typ for typ,x in POTTED_TYPES.items()})
    assert len(book) == 21 and len(candidates) == 20
    choices = [8, 9]
    crosswalk = book[['excel_row','品种']].copy()
    crosswalk['used_in_figure'] = crosswalk.excel_row.isin(candidates.loc[~candidates.excel_row.eq(9),'excel_row'])
    crosswalk['reason'] = 'Workbook pair retained by manifest species quota'
    crosswalk.loc[crosswalk['品种'].eq('白掌'),'reason'] = 'Species absent from the current manifest'
    crosswalk.loc[crosswalk.excel_row.eq(8),'reason'] = 'First peacock plant row for a quota of one; specimen ID unavailable'
    crosswalk.loc[crosswalk.excel_row.eq(9),'reason'] = 'Alternative peacock plant row; specimen ID unavailable'
    p.dump(crosswalk,'potted_workbook_selection')
    records = []
    for peacock_row in choices:
        other = next(i for i in choices if i != peacock_row)
        use = candidates[~candidates.excel_row.eq(other)]
        assert use.plant_type.value_counts().to_dict() == expected.to_dict()
        rows = []
        for _, r in use.iterrows():
            for trait, _, label in FIG6_TRAITS:
                for number in ([1,2,3] if trait in {CHORD,WIDTH} else ['']):
                    rows.append(dict(trait=trait, model=r[f'{label}虚拟植{number}'], reference=r[f'{label}真值{number}'],
                                     species=r.plant_type, sample_id=f'Sheet1!A{r.excel_row}', organ_id=number,
                                     cohort='Potted', source=str(path)))
        records.append(pd.DataFrame(rows))
    return records


def agreement_summary(data):
    result = []
    for trait, _, _ in FIG6_TRAITS:
        d = data[data.trait.eq(trait)]
        err = d.model-d.reference
        centered = d.reference-d.reference.mean()
        rmse = np.sqrt(np.mean(err**2))
        bias, sd = err.mean(), err.std(ddof=1)
        result.append(dict(trait=trait, n=len(d), n_samples=d[['cohort','sample_id']].drop_duplicates().shape[0],
                           mae_cm=err.abs().mean(), rmse_cm=rmse,
                           r2_identity=1-(err**2).sum()/(centered**2).sum(), mean_bias_cm=bias,
                           sd_cm=sd, lower_loa_cm=bias-1.96*sd, upper_loa_cm=bias+1.96*sd))
    return pd.DataFrame(result)


def agreement(p):
    fig=figure(p,'Fig6_Leaf_trait_agreement',(190,237),
               ['Height parity','Height Bland–Altman','Canopy parity','Canopy Bland–Altman',
                'Leaf length parity','Leaf length Bland–Altman','Leaf width parity','Leaf width Bland–Altman'],
               'Pooled agreement across tomato and potted paired observations for each of four traits.')
    cohort = metric_cohort(p,'a–h')
    tomato = cohort[cohort.group.ne('Potted')]
    official = p.csv(Path('07_PHENOTYPING_EVALUATION/Tomato_Metric_19/Whole_plant_pointcloud_reference.csv'),'a–d')
    assert set(official.sample_id) == set(tomato.sample_id)
    fields = ['plant_height_cm','canopy_width_cm']
    plant = pd.concat([p.csv(TOMATO_PHENO/r.sample_id/'plant_measurements.csv','a–d',fields)
                       .assign(sample_id=r.sample_id, species=r.group) for r in tomato.itertuples()])
    plant = plant.merge(official, on='sample_id', validate='one_to_one')
    p.dump(plant,'tomato_whole_plant_reference_coverage')
    plant = plant[plant.status.eq('derived_from_official_labeled_cloud')]
    assert len(plant) == 18
    whole = [plant[['sample_id','species',model,ref]].rename(columns={model:'model',ref:'reference'})
             .assign(trait=trait, organ_id='', cohort='Tomato', source='Official annotated cloud; metric mesh')
             for model, ref, trait in zip(fields,['reference_height_cm','reference_canopy_cm'],['height','canopy'])]
    pairs = p.table('Tomato_Leaf_Trait_Pairs','e–h')
    pairs = pairs[pairs.status.eq(GOOD) & pairs.scene.isin(tomato.sample_id)].copy()
    pairs['species'] = pairs.cultivar.map(CODES)
    leaves = pairs[['scene','official_point_instance','candidate_id','species','trait','model','reference','endpoints_json']].rename(
        columns={'scene':'sample_id','official_point_instance':'organ_id'}).assign(cohort='Tomato',source='Official annotated cloud; reference-guided mesh endpoints')
    potted, alternate = historical_potted_pairs(p,cohort)
    data = pd.concat([*whole,leaves,potted],ignore_index=True)
    assert data[['model','reference']].gt(0).all().all() and data.species.notna().all()
    data['difference_cm'] = data.model-data.reference
    data['absolute_error_cm'] = data.difference_cm.abs()
    summary = agreement_summary(data)
    assert dict(zip(summary.trait,summary.n)) == {'height':37,'canopy':37,CHORD:71,WIDTH:81}
    alternative_summary = agreement_summary(pd.concat([*whole,leaves,alternate],ignore_index=True))
    sensitivity = pd.concat([summary.assign(peacock_excel_row=8),alternative_summary.assign(peacock_excel_row=9)],
                            ignore_index=True)
    p.dump(data.sort_values(['trait','absolute_error_cm'],ascending=[True,False]),'eligible_pairs')
    p.dump(summary,'agreement_and_limits')
    p.dump(sensitivity,'potted_peacock_row_sensitivity')

    species = {k:(NAMES[i],MARKS[i],COLORS[i]) for i,k in enumerate(GROUPS[:3])}
    species.update({k:(name,marker,COLORS[3]) for k,(_,name,marker) in POTTED_TYPES.items()})
    handles = [Line2D([],[],label=label,marker=marker,markerfacecolor=color,
                      markeredgecolor='white',markeredgewidth=.35,color='none',markersize=5.4)
               for label,marker,color in species.values()]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.985,.994),ncol=6,frameon=False,
               handletextpad=.2,columnspacing=.8,labelspacing=.6)
    grid = fig.add_gridspec(4,2,left=.12,right=.953,bottom=.064,top=.883,wspace=.27,hspace=.64)
    axes = [fig.add_subplot(cell) for cell in grid]
    for i,(trait,title,_) in enumerate(FIG6_TRAITS):
        parity,difference = axes[2*i:2*i+2]
        panel(parity,chr(97+2*i),title,pad=32)
        panel(difference,chr(98+2*i),title+' differences',pad=32)
        group = data[data.trait.eq(trait)]
        extent = group[['model','reference']].max().max()
        limit = next(n for n in [3,4,5,8,10,12,15,20,25,30,40,50] if n > extent*1.03)
        parity.set_xlim(0,limit); parity.set_ylim(0,limit)
        parity.set_box_aspect(1)
        difference.set_box_aspect(1)
        parity.plot([0,limit],[0,limit],c=MUTED,lw=.7,ls='--',zorder=0)
        for name, sub in group.groupby('species'):
            _, marker, color = species[name]
            style = dict(marker=marker,c=color,alpha=.86,s=24,edgecolors='white',linewidths=.32,zorder=3)
            parity.scatter(sub.reference,sub.model,**style)
            difference.scatter((sub.reference+sub.model)/2,sub.model-sub.reference,**style)
        stats = summary.set_index('trait').loc[trait]
        parity.annotate(f'n = {int(stats.n)}   RMSE = {stats.rmse_cm:.2f} cm   R² = {stats.r2_identity:.2f}',
                        (0,1),xycoords='axes fraction',xytext=(0,15),textcoords='offset points',
                        va='baseline',ha='left')
        difference.annotate(f'Bias = {stats.mean_bias_cm:+.2f} cm   LoA [{stats.lower_loa_cm:+.2f}, {stats.upper_loa_cm:+.2f}]',
                            (0,1),xycoords='axes fraction',xytext=(0,15),textcoords='offset points',
                            va='baseline',ha='left')
        for val,style in [(stats.mean_bias_cm,'-'),(stats.lower_loa_cm,'--'),(stats.upper_loa_cm,'--')]:
            difference.axhline(val,ls=style,color=MUTED,lw=.8,zorder=1)
        lo = min(group.difference_cm.min(),stats.lower_loa_cm)
        hi = max(group.difference_cm.max(),stats.upper_loa_cm)
        margin = .08*(hi-lo)
        difference.set(xlim=(0,limit),ylim=(lo-margin,hi+margin))
        parity.set_ylabel('Model (cm)')
        difference.set_ylabel('Model − reference (cm)')
        if i==3:
            parity.set_xlabel('Reference (cm)')
            difference.set_xlabel('Mean of reference and model (cm)')
    finish(p,fig,'Pooled agreement for (a,b) plant height, (c,d) canopy width, (e,f) leaf length and (g,h) leaf width. The tomato and potted records contribute to a single set of statistics per trait. Shapes identify species or tomato cultivars. The left panels show reference versus model with the 1:1 line. R² is evaluated against identity. The right panels show Bland–Altman differences, with observed mean bias (solid) and limits of agreement at mean bias ± 1.96 sample SD (dashed). Tomato plant dimensions are the visible vertical span and maximum horizontal span of the plant mesh and official labelled cloud in their respective reference frames. Eighteen of the 19 metric scans have the official transform. Tomato leaf length is the 3D base-to-tip chord and width is a corresponding midsection chord (14 and 24 paired observations). Potted observations are paired manual and virtual values from the supplied workbook, filtered to the current cohort’s species counts (19 plants, 57 leaf entries). Sheet1 row 8 provisionally supplies the peacock plant entry. The pooled statistics describe the plotted observations across the two measurement protocols. Leaves can share a plant and multiple scans can share a cultivar.',axes=axes)


def module(p,key,labels,desc,*, size=(90,88)):
    fig=figure(p,key,size,labels,desc)
    n=len(labels)
    axes=[]
    for i,(heading,body) in enumerate(labels):
        y=.74-i*.236
        ax=abox(fig,[.071,y,.858,.158]); axes.append(ax)
        band(fig,'abc'[i],heading,y+.225,x=.075)
        step(ax,.00,.04,1,.96,body)
        if i<n-1:
            link(ax,(.94,.02),(.94,-.473))
    return fig,axes


def supplements(p):
    fig,axes=module(p,'FigS1_Foreground_detail',[
        ('Cohort-specific inputs','Tomato: controlled-background mask\nPotted: VFM prompt bank'),
        ('Potted candidate selection','Semantic-gated multi-prompt selection\nContrast-decoupled re-ranking'),
        ('Potted refinement','Structure-balanced foreground and\nbackground prompting'),
    ],'Front-end module detail.')
    p.source(Path('02_RAP_FSAM3_FRONTEND/Method/Recovered_RAP_Chain/01_PAPER_CODE_MAPPING.md'),'b,c')
    finish(p,fig,'The archived potted reconstruction protocol uses the RAP-FSAM3 route. The controlled-background TomatoPGT route uses a colour-distance foreground prior for masked SfM; it does not share the full potted prompt chain.',axes=axes,cols=[['a','b','c']])
    fig=figure(p,'FigS2_Metric_GGGS_TSDF_detail',(190,105),['Scale anchors','Metric depth observations'],
               'A common GGGS–TSDF path uses separate physical anchors and screened depth.')
    band(fig,'a','Metric geometry',.94)
    u=abox(fig,[.045,.63,.91,.22])
    stages=['COLMAP SfM','Native GGGS','Scale once','Screen depth','Bounded TSDF']
    for i,title in enumerate(stages):
        x=.207*i
        step(u,x,.61,.172,.37,title)
        if i<4: link(u,(x+.172,.795),(x+.207,.795))
    step(u,.311,.015,.38,.36,'Tomato board | Potted 2 cm cube')
    link(u,(.50,.37),(.50,.61))
    band(fig,'b','GGGS renderings and depth',.605)
    cells=[]
    for i,(folder,label) in enumerate([(CASE,'TomatoPGT'),(KQ,'Potted peacock plant')]):
        x=.045+i*.465
        group=abox(fig,[x,.16,.445,.34]); cells.append(group)
        fig.text(x,.496,label,va='bottom')
        photo(abox(fig,[x,.203,.209,.271]),p,folder/'GGGS_Render.png','b')
        ax=abox(fig,[x+.237,.203,.207,.271]); cb=fig.add_axes([x+.246,.156,.189,.018])
        depth(ax,cb,p,folder,'b')
        fig.text(x+.105,.167,'GGGS rendering',ha='center')
    finish(p,fig,'(a) COLMAP initialises GGGS. The tomato physical anchor uses independently triangulated board baselines; the potted cohort uses a 2 cm cube. The scale is applied once. (b) Archived GGGS renderings and cached metric depth, with invalid values white and the 2nd and 98th percentiles of valid depth as the per-scene colour limits. Colorbars report centimetres. Fusion uses screened depth and the fixed bounded-TSDF route.',axes=cells,rows=[['a','b']])
    fig,axes=module(p,'FigS3_Organ_association_detail',[
        ('Multiview SAM3','Plant and pot masks\nCandidate organ masks'),
        ('Visible-triangle votes','Semantic mesh labelling\nPersistent organ association'),
        ('Traceable trait exports','Source RGB and instance mesh\nMetric spans and observed area'),
    ],'Semantic/organ module detail.')
    p.source(Path('10_REPRODUCIBILITY/Source_Code_References/ForeSplat_AutoGaussianCleaner/phenotyping/measure_phenotypes.py'),'c')
    finish(p,fig,'Semantic integration separates plant and pot. Multiview instance association groups organ support over visible triangles. Independent organ exports preserve identity and a supporting reference view. Tomato compound organs and manually numbered potted leaves remain separate measurement units.',axes=axes,cols=[['a','b','c']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'12_PAPER_FIGURES_REORGANIZED_DRAFT')
    parser.add_argument('--dpi',type=int,default=EXPORT_DPI)
    args=parser.parse_args()
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                         'font.size':7.4,'xtick.labelsize':7.2,
                         'ytick.labelsize':7.2,'axes.labelsize':7.4,'legend.fontsize':7.25,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.right':False,
                         'axes.spines.top':False,'axes.linewidth':.6,'axes.edgecolor':MUTED,
                         'axes.grid':False,'text.color':INK,'axes.labelcolor':INK,
                         'xtick.color':INK,'ytick.color':INK,'figure.facecolor':'white'})
    args.output.mkdir(parents=True,exist_ok=True)
    p=Plates(args.evidence.resolve(),args.output.resolve(),args.dpi)
    for f in (overview,foreground,flow,semantics,quality,agreement,supplements):
        f(p)
    pd.DataFrame(p.sources).to_csv(p.out/'Figure_Source_Manifest.csv',index=False)
    pd.DataFrame(p.index).to_csv(p.out/'Figure_Index.csv',index=False)
    (p.out/'Figure_Captions.md').write_text('# Figure captions\n\n'+'\n\n'.join(
        f"**{r['id'].split('_')[0]}**. {r['note']}" for r in p.index)+'\n',encoding='utf-8')
    for name in ('render_foresplat_reorganized_figures.py','render_foresplat_tomato_first.py','render_foresplat_paper_figures.py'):
        src=Path(__file__).with_name(name).resolve()
        if src!=p.out/name:
            shutil.copy2(src,p.out/name)


if __name__=='__main__':
    main()
