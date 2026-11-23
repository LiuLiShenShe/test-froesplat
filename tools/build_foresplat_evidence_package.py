#!/usr/bin/env python3
"""Archive existing ForeSplat evidence; never run reconstruction or edit a source."""

import argparse
import csv
import errno
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from plot_foresplat_evidence import RGB_RESIDUAL_MAX, depth_preview, plot_pairs, rgb_residual_preview


ROOT = Path(__file__).resolve().parents[1]
POT = ROOT / 'experiments/ForeSplat_MultiSpecies_Phenotyping'
TOM = ROOT / 'experiments/ForeSplat_TomatoPGT_Independent'
COMPARE = ROOT / 'experiments/COLMAP_Gaussian_Mesh_Selection'
RAW = Path('/fj/end-paper/tomato_3d_benchmark')
RAW_TOM = RAW / '02_raw/TomatoPGT'
STD = RAW / '03_standardized/TomatoPGT'
MANUSCRIPT = ROOT / '2026-11-18修改后论文'
SOURCES = ROOT / '00-论文优化重构/计算机与电子农业特刊实验工作区'
DATA = ROOT / '00-论文优化重构/数据管理'
CLEANER = SOURCES / 'ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner'
REP4 = SOURCES / '06-实验一四样本代表集/E1_Representative4_PottedPlant_VFM'
RECOVERY = SOURCES / 'ForeSplat_Original_Method_Recovery'
REVIEW = TOM / 'summary/compound_auto_review'

INDEX, PAPER, FRONT, GAUSS, SCALE, MESH, ORGANS, TRAITS, FIGS, SUP, REPRO, PORTABLE = (
    Path(n) for n in (
        '00_README_AND_INDEX', '01_MANUSCRIPT', '02_RAP_FSAM3_FRONTEND', '03_COLMAP_GGGS_RECONSTRUCTION',
        '04_METRIC_CALIBRATION', '05_TSDF_MESH_RECONSTRUCTION', '06_PLANT_AND_ORGAN_SEGMENTATION',
        '07_PHENOTYPING_EVALUATION', '08_PAPER_FIGURE_MATERIALS', '09_SUPPLEMENTARY_MATERIALS',
        '10_REPRODUCIBILITY', '11_PORTABLE_RESEARCH_PACK',
    )
)
POT_VER, TOM_VER, SEL_VER, FRONT_VER, LEGACY = (
    'potted_gggs_tsdf_all19', 'tomatopgt_independent_gggs_tsdf_42', 'kongquezhu_five_gaussian_selection_30k',
    'rap_fsam3_frontend', 'legacy_2dgs_m2m3'
)
POT_REPS = ('KongQueZhuYu', 'CaoMei1', 'ChangShouHua2', 'XianKeLai1')
TOM_REPS = ('BFS_06262025', 'BBH_05272025', 'CH_06172025')
IMAGES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.svg', '.webp'}
SMALL = {'.csv', '.json', '.jsonl', '.md', '.txt', '.yaml', '.yml', '.html', '.npz'}
CODE = {'.py', '.sh', '.js', '.json', '.toml', '.yaml', '.yml', '.md', '.html', '.csv', '.txt'}
MIB = 1024 * 1024


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def json_data(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def csv_save(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def filesystem_files(path):
    if path.is_dir():
        base = path.resolve()
        return sorted(p for p in base.rglob('*') if p.is_file()
                      and not any(x.startswith('.') or x == '__pycache__' for x in p.relative_to(base).parts))
    return []


def image_for(parent, view):
    return next((p for p in sorted(parent.glob(f'{view}.*')) if p.suffix.lower() in IMAGES), None)


def rate_transfer(source, target, rate=20*MIB):
    start = time.monotonic()
    done = 0
    while chunk := source.read(MIB):
        target.write(chunk)
        done += len(chunk)
        if delay := done / rate - (time.monotonic() - start):
            if delay > 0:
                time.sleep(delay)


def copy_material(src, dst, isolated=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not isolated:
        try:
            os.link(src.resolve(strict=True), dst)
            return 'hardlink'
        except OSError as e:
            if e.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK}:
                raise
    if src.stat().st_size >= 16 * MIB:
        with src.open('rb') as f, dst.open('xb') as out:
            rate_transfer(f, out)
        shutil.copystat(src, dst)
    else:
        shutil.copy2(src, dst)
    return 'copy'


def classify(path, role):
    if path.is_dir():
        return 'directory'
    if path.suffix.lower() in IMAGES:
        return 'image'
    if path.suffix == '.ply':
        if 'Gaussian' in role or path.name in {'original_gaussians.ply', 'metric_gaussians.ply'}:
            return 'gaussian_ply'
        if 'mesh' in role.lower():
            return 'mesh_ply'
        return 'point_cloud_ply'
    return path.suffix.lstrip('.') or 'file'


class Archive:
    def __init__(self, out):
        self.out = out
        self.assets = []
        self.destinations = {}
        self.source_locations = {}
        self.tables = {}
        self.figures = {}
        self.gaps = []

    def add(self, src, dst=None, *, dataset='', version='', sample='', organ='', view='', role='', unit='',
            required=False, isolated=False):
        if src is None or not src.exists():
            if required:
                raise FileNotFoundError(src)
            return ''
        location = dst.as_posix() if dst else ''
        real = src.resolve(strict=True)
        size = real.stat().st_size if real.is_file() else 0
        if dst:
            if not real.is_file():
                raise ValueError(f'Archive entry must be a file: {src}')
            if location in self.destinations:
                if real != self.destinations[location]:
                    raise ValueError(f'Two sources for {location}')
                return location
            delivery = copy_material(real, self.out/dst, isolated)
            self.destinations[location] = real
            self.source_locations.setdefault(real, location)
        else:
            delivery = 'server_index'
        self.assets.append(dict(dataset=dataset, experiment_version=version, sample_id=sample, organ_id=organ,
                                view_id=view, role=role, kind=classify(real, role), unit=unit, delivery=delivery,
                                bytes=size, archive_path=location, source_path=str(src), source_real_path=str(real)))
        return location

    def tree(self, src, dst=None, *, extensions=None, limit=None, **kwargs):
        base = src.resolve()
        for p in filesystem_files(src):
            if (extensions is None or p.suffix.lower() in extensions) and (limit is None or p.stat().st_size <= limit):
                rel = p.relative_to(base)
                self.add(src/rel, dst/rel if dst is not None else None, **kwargs)

    def table(self, name, rows, folder=SUP/'Per_Sample_Tables'):
        path = folder/f'{name}.csv'
        csv_save(self.out/path, rows)
        self.tables[name] = path
        self.generated(path, 'derived_master_table')
        return path

    def generated(self, path, role, *, source=None, **meta):
        self.assets.append(dict(dataset=meta.get('dataset', ''), experiment_version=meta.get('version', ''),
                                sample_id=meta.get('sample', ''), organ_id=meta.get('organ', ''), view_id=meta.get('view', ''),
                                role=role, kind=classify(self.out/path, role), unit=meta.get('unit', ''), delivery='generated',
                                bytes=(self.out/path).stat().st_size, archive_path=path.as_posix(),
                                source_path=str(source) if source else '', source_real_path=str(source.resolve()) if source else ''))

    def text(self, path, value):
        dest = self.out/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value.rstrip()+'\n', encoding='utf-8')

    def archived(self, src):
        return self.source_locations.get(src.resolve(), '')


def manuscript(a):
    old = SOURCES/'ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant'
    for p in sorted(MANUSCRIPT.iterdir()):
        if p.name.startswith('Fig') and p.suffix == '.png':
            target = PAPER/'Figures_Original'/p.name
        elif p.suffix in {'.bib', '.bst'} or 'References' in p.name:
            target = PAPER/'References'/p.name
        elif p.name.startswith('YBENG-'):
            target = PAPER/'Original_Paper/Submitted'/p.name
        else:
            target = PAPER/'Current_Paper'/p.name
        a.add(p, target, role='manuscript', isolated=True)
        if p.name.startswith('Fig') and p.suffix == '.png' or p.suffix in {'.bib', '.bst'}:
            a.add(p, PAPER/'Current_Paper'/p.name, role='manuscript_build_dependency', isolated=True)
    a.tree(old, PAPER/'Original_Paper/Earlier_Draft', role='earlier_manuscript', isolated=True)
    a.tree(ROOT/'00参考文章', PAPER/'References/Local_Papers', extensions={'.pdf'}, role='local_reference')
    a.tree(SOURCES/'ForeSplat_Manuscript_Final_Verification', PAPER/'Revision_Notes_Archive', extensions=SMALL,
           role='historical_revision_recommendations', version=LEGACY)


def frontend(a):
    meta = dict(dataset='RAP-FSAM3', version=FRONT_VER)
    a.add(ROOT/'02-FFT/blur_filter.py', FRONT/'Method/FFT/blur_filter.py', role='implementation', **meta)
    directory = next((DATA/'07-运行脚本与超参').glob('S20-RAP*'))
    a.tree(directory, FRONT/'Method/S20_Implementation', extensions=CODE, limit=4*MIB, role='implementation', **meta)
    a.tree(ROOT/'00-论文优化重构/compag特刊增修模块', FRONT/'Method/Module_Notes', extensions={'.md'}, role='method', **meta)
    for section in ('configs', 'scripts', 'figures', 'metrics'):
        target = 'Figures/Recovered_RAP_Chain' if section == 'figures' else f'Ablations/Recovered_RAP_Chain/{section}'
        a.tree(RECOVERY/section, FRONT/target, extensions=CODE | IMAGES | {'.npz'}, limit=8*MIB,
               role=section, **meta)
    for p in RECOVERY.glob('*.md'):
        a.add(p, FRONT/'Method/Recovered_RAP_Chain'/p.name, role='run_report', **meta)
    for pattern in ('阶段二十C_R1*', '阶段二十D_固定*', '阶段二十D_独立*', '阶段二十一*'):
        for project in SOURCES.glob(pattern):
            for p in sorted(project.iterdir()):
                if p.is_file() and p.suffix in SMALL and not p.name.startswith('.'):
                    a.add(p, FRONT/'Ablations'/project.name/p.name, role='evaluation_protocol', **meta)
            for sub in ('metrics', 'reports', 'figures', 'scripts', 'configs', 'gt_protocol_v2'):
                a.tree(project/sub, FRONT/'Ablations'/project.name/sub, extensions=CODE | IMAGES | {'.pdf'},
                       limit=8*MIB, role=sub, **meta)

    for cohort, source in [('Representative4', REP4), ('S23_Smaller_Benchmark', DATA/'05-评测结果/S23_Experiment1_VFM_Benchmark')]:
        for sub in ('gt_masks', 'selected_frames', 'method_masks', 'metrics', 'visual_assets'):
            target = 'Masks_and_RGB' if sub in ('gt_masks', 'selected_frames', 'method_masks') else 'Evaluation_Data'
            a.tree(source/sub, FRONT/target/cohort/sub, extensions=CODE | IMAGES | {'.pdf'}, role=sub, **meta)
        a.add(source/'manifest.csv', FRONT/'Evaluation_Data'/cohort/'manifest.csv', role='sample_manifest', **meta)
        a.add(source/'rap_runs', role='all_historical_prompt_and_stage_assets', **meta)
    a.add(DATA/'03-分割Mask/05-RAP-FSAM3掩膜', role='all_historical_rap_masks', **meta)
    cg = COMPARE/'common_input/RAP_FSAM3'
    a.add(cg, role='selection_rap_legacy_stage_diagnostics', **meta)
    for sub in ('叠加图', '语义门控框', '正负提示细化掩膜', '几何修正提示图', '选择后掩膜'):
        for view in ('0000', '0048', '0096', '0144'):
            for p in sorted((cg/sub).glob(f'*{view}.*')):
                if p.suffix.lower() in IMAGES:
                    a.add(p, FRONT/'Masks_and_RGB/Selection_Cached_Stages'/sub/p.name,
                          sample='KongQueZhuYu', view=view, role='stage_diagnostic_check_run_provenance', **meta)
    for p in (cg/'ABORTED_NOT_USED.md', cg/'参数.json'):
        a.add(p, FRONT/'Masks_and_RGB/Selection_Cached_Stages'/p.name, role='stage_provenance', **meta)


def selection(a):
    meta = dict(dataset='KongQueZhuYu selection', version=SEL_VER, sample='KongQueZhuYu')
    src = COMPARE/'summary'
    a.tree(src, GAUSS/'Five_Method_Comparison/summary', extensions=SMALL | IMAGES, role='comparison_evidence', **meta)
    for p in src.glob('*resources.*'):
        a.add(p, GAUSS/'Performance'/p.name, role='runtime_resource_source', **meta)
    a.tree(COMPARE/'scripts', REPRO/'Source_Code_References/Selection', extensions=CODE, role='implementation', **meta)
    for n in ('colmap_source.json', 'test_views.txt', 'train_views.txt', 'image_manifest.json', 'rap_mask_manifest.json'):
        a.add(COMPARE/'common_input'/n, GAUSS/'Method_and_Configurations'/n, role='frozen_protocol', **meta)
    a.add(COMPARE/'common_input/source_sparse', role='selection_native_sfm', **meta)
    for method in sorted((COMPARE/'gaussian').iterdir()):
        for p in method.iterdir():
            if p.is_file() and (p.suffix in SMALL or p.name == 'cfg_args'):
                a.add(p, GAUSS/'Method_and_Configurations'/method.name/p.name, role='method_configuration', **meta)
            elif p.suffix in {'.pt', '.pth'} or p.name in {'ckpts', 'tb'}:
                a.add(p, role='training_output', **meta)
        a.tree(method/'evaluation', GAUSS/'Render_Quality'/method.name, extensions=SMALL | IMAGES, role='evaluation', **meta)
        for p in method.glob('**/*.ply'):
            role = 'native_Gaussian_PLY' if p.name != 'input.ply' else 'SfM_seed_PLY'
            a.add(p, GAUSS/'Representative_Gaussian_Models'/method.name/p.name if p.name != 'input.ply' else None,
                  role=role, unit='sfm_units', **meta)
    for p in filesystem_files(COMPARE/'mesh'):
        rel = p.relative_to((COMPARE/'mesh').resolve())
        src_p = COMPARE/'mesh'/rel
        role = 'selection_mesh' if p.suffix == '.ply' else 'geometry_evaluation'
        keep = p.suffix != '.npz' or p.stat().st_size <= 8*MIB
        a.add(src_p, MESH/'Mesh_Comparison'/rel if keep else None, role=role, unit='sfm_units' if p.suffix == '.ply' else '', **meta)
    for n in ('mesh_comparison.csv', 'mesh_overview.png', 'final_selection.md', 'depth_display_protocol.json'):
        a.add(src/n, GAUSS/'Geometry_Comparison'/n, role='geometry_selection', **meta)
    protocol = dict(dataset='KongQueZhuYu', experiment_version=SEL_VER, scope='one_plant_only',
                    train_views=183, test_views=27, geometry_unit='sfm_units', reference='heldout_RGB_and_RAP_proxy')
    g = [dict(protocol, **r) for r in csv_rows(src/'gaussian_comparison.csv')]
    m = [dict(protocol, **r) for r in csv_rows(src/'mesh_comparison.csv')]
    a.table('Gaussian_Method_Comparison', g)
    a.table('Mesh_Method_Comparison', m)
    perf = [dict(protocol, stage='Gaussian_training', method=r['method'], **{k: v for k,v in r.items() if
                  k.startswith(('training_', 'peak_', 'observed_', 'io_'))}) for r in g]
    perf.extend(dict(protocol, stage=f"Mesh_{r['stage']}", method=f"{r['model']} / {r['method']}",
                     **{k:v for k,v in r.items() if k.startswith(('extraction_', 'peak_', 'observed_', 'io_'))}) for r in m)
    a.table('Runtime_and_Resources', perf)


def legacy_and_code(a):
    for tag in ('阶段二十E0*', '阶段二十E1*', '阶段二十E2*', '阶段二十E3*', '阶段二十F*'):
        for p in SOURCES.glob(tag):
            for f in p.glob('*.md'):
                a.add(f, SUP/'Legacy_2DGS_and_M2M3'/p.name/f.name, version=LEGACY, role='legacy_protocol')
            for sub in ('configs','metrics','reports','figures','scripts'):
                a.tree(p/sub, SUP/'Legacy_2DGS_and_M2M3'/p.name/sub, extensions=CODE | IMAGES, limit=8*MIB,
                       version=LEGACY, role='legacy_2dgs_result_or_implementation')
    a.tree(ROOT/'ForeSplat_real_release/examples/paper_tables', SUP/'Legacy_2DGS_and_M2M3/Release_Paper_Tables',
           version=LEGACY, role='legacy_quantitative')
    for p in (ROOT/'00-论文优化重构/论文2026-05-28').glob('fig*'):
        if p.is_dir():
            a.tree(p, PAPER/'Historical_Figure_Assets'/p.name, extensions=CODE | IMAGES | {'.pdf'}, limit=16*MIB,
                   version=LEGACY, role='earlier_figure_components_check_mapping')
    for sub in ('phenotyping', 'colmap_selection', 'configs'):
        a.tree(CLEANER/sub, REPRO/'Source_Code_References/ForeSplat_AutoGaussianCleaner'/sub, extensions=CODE,
               limit=8*MIB, role='pipeline_implementation')
    tools = ROOT/'tools'
    for p in (ROOT/'run_colmap_phenotyping_all19.py', ROOT/'run_tomatopgt_pilot.py',
              tools/'pair_tomatopgt_compounds.py', tools/'tomatopgt_compound_review.js',
              tools/'summarize_tomatopgt_leaf_errors.py', tools/'build_foresplat_measurement_workbook.py',
              tools/'build_foresplat_evidence_package.py', tools/'plot_foresplat_evidence.py'):
        a.add(p, REPRO/'Plotting_Scripts'/p.name if p.name.startswith(('plot_', 'summarize_'))
              else REPRO/'Source_Code_References/Entrypoints'/p.name, role='pipeline_or_plot_entrypoint')


def views_from(calibration, tomato):
    if not tomato:
        views = set()
        for item in (calibration.get('reference_views') or []) + (calibration.get('rejected_views') or []):
            view = item.get('view') if isinstance(item, dict) else item
            if view:
                views.add(str(view))
        return sorted(views)
    return sorted({str(v) for t in (calibration.get('targets') or {}).values() for v in (t.get('views') or [])})


def collect_calibration(a, root, sample, tomato, row, meta):
    c = root/'calibration/scale_calibration.json'
    j = json_data(c)
    dest = SCALE/('Tomato_Target_Board' if tomato else 'Potted_Plants_2cm_Cube')/sample
    a.tree(c.parent, dest, extensions=SMALL | IMAGES, role='scale_and_reference_frame', **meta)
    input_dir = root/('raw_rgb' if tomato else 'common_input/images')
    for view in views_from(j, tomato):
        if image := image_for(input_dir, view):
            a.add(image, dest/'Reference_RGB'/image.name, role='calibration_RGB', view=view, **meta)
        if not tomato:
            mask = root/'semantic_masks'/view/'blue_cube.png'
            a.add(mask, dest/'Detected_Cube_Masks'/f'{view}.png', role='calibration_mask', view=view, **meta)
    if j.get('coordinate_csv'):
        target = Path(j['coordinate_csv'])
        a.add(target, dest/'Published_Target_Coordinates'/target.name, role='published_board_geometry', **meta)
    metric = row['metric_mesh'] == 'True'
    return dict(sample_id=sample, dataset=meta['dataset'], cultivar_or_type=row.get('cultivar', row.get('plant_type', '')),
                experiment_version=meta['version'], metric_mesh=row['metric_mesh'], unit='cm' if metric else 'sfm_units',
                reference_kind='published_target_board' if tomato else '2 cm blue cube',
                status=j.get('status', row.get('calibration', row.get('calibration_status', ''))),
                method=j.get('method', j.get('calibration_mode', '')), scale_cm_per_unit=j.get('scale_cm_per_unit', ''),
                exploratory_scale_not_for_measurement=j.get('exploratory_scale_cm_per_unit', ''),
                fitted_cube_edge_units=j.get('reconstructed_edge_length_units', ''),
                retained_view_count=j.get('matched_views', len(j.get('reference_views') or [])),
                calibration_view_ids=' '.join(views_from(j, tomato)), rejected_view_ids=json.dumps(j.get('rejected_views', [])),
                cube_reprojection_error_px=j.get('reprojection_error_px', ''),
                boundary_rms_px=j.get('boundary_support_rms_px', ''),
                board_reprojection_raw_px=j.get('max_target_median_reprojection_raw_px', ''),
                max_relative_baseline_scale_deviation=j.get('max_relative_baseline_scale_deviation', ''),
                qc_details=json.dumps(j.get('qc', {}), ensure_ascii=False),
                scale_json=a.archived(c), reference_frame_json=a.archived(c.parent/'reference_frame.json'), source_json=str(c))


def collect_instance(a, src, dst, organ, unit, meta, archive=False, original_masks=False):
    m = src/f'leaf_{unit}.ply'
    role = 'compound_leaf_mesh' if organ.startswith('Compound_') else 'leaf_instance_mesh'
    kw = dict(organ=organ, **meta)
    path = a.add(m, dst/m.name if archive else None, role=role, unit='cm' if unit == 'cm' else 'sfm_units',
                 required=True, **kw)
    for n in ('reference.png', 'reference_rgb.png', 'instance.json', 'landmarks.json', 'viewer.html', 'source_indices.npz'):
        p = src/n
        kind = {'reference.png':'instance_overlay', 'reference_rgb.png':'instance_RGB',
                'landmarks.json':'model_landmark_record'}.get(n, 'instance_metadata')
        a.add(p, dst/n if archive else None, role=kind, **kw)
    if original_masks:
        a.tree(src/'masks', dst/'masks' if archive else None, role='original_instance_mask', **kw)
    else:
        a.add(src/'masks', role='original_instance_masks_directory', **kw)
    return path or a.archived(m)


def datasets(a):
    all_pot = csv_rows(POT/'summary/all19_results.csv')
    all_tom = csv_rows(TOM/'summary/batch_results.csv')
    pot_matches = csv_rows(POT/'summary/three_leaf_model_locations.csv')
    compounds = csv_rows(REVIEW/'compound_match_suggestions.csv')
    pot_keys = {(r['plant_id'], r['leaf_id']): r for r in pot_matches if r['human_confirmed'] == 'True'}
    tom_keys = {(r['scene'], r['candidate_id']) for r in compounds
                if r['candidate_id'] and r['metric_scan'] == 'True' and r['reference_scope'] == 'compound_leaf'}
    assert len(all_pot) == 19 and len(all_tom) == 42 and len(pot_keys) == 57
    compound_counts = Counter(r['scene'] for r in compounds if r['reference_scope'] == 'compound_leaf')
    for run, rows, tomato in ((POT, all_pot, False), (TOM, all_tom, True)):
        tag, version = ('TomatoPGT', TOM_VER) if tomato else ('Potted', POT_VER)
        meta = dict(dataset=tag, version=version)
        batch_dir = TRAITS/('TomatoPGT_42' if tomato else 'Potted_19')
        for p in sorted((run/'summary').iterdir()):
            if p.is_file() and p.suffix.lower() in (SMALL | IMAGES):
                a.add(p, batch_dir/'Batch_Reports'/p.name, role='original_batch_source', **meta)
        a.add(run/'README.md', batch_dir/'RUN_README.md', role='batch_description', **meta)
        if tomato:
            a.tree(REVIEW, batch_dir/'Compound_Review', extensions=SMALL | IMAGES | {'.js'}, role='compound_review', **meta)
        prepared, calibrations = [], []
        for row in rows:
            sample = row['scene' if tomato else 'plant_id']
            x = run/sample
            ctx = dict(meta, sample=sample)
            is_metric = row['metric_mesh'] == 'True'
            unit = 'cm' if is_metric else 'units'
            real_unit = 'cm' if is_metric else 'sfm_units'
            representative = sample in (TOM_REPS if tomato else POT_REPS)
            sample_dir = SUP/('All_42_Tomato_Scans' if tomato else 'All_19_Potted_Plants')/sample
            mdir = MESH/('Tomato_Meshes' if tomato else 'Potted_Meshes')/sample
            data_dir = batch_dir/'Per_Sample'/sample
            a.tree(x/'summary', sample_dir/'Figures', extensions=IMAGES | SMALL, role='per_sample_preview_and_qc', **ctx)
            a.tree(x/'phenotype', data_dir, extensions=SMALL, role='per_sample_traits', **ctx)
            for n in ('train_views.txt', 'test_views.txt', 'cameras.json', 'complete.json', 'image_manifest.json'):
                a.add(x/'common_input'/n, sample_dir/'Protocol'/n, role='actual_common_input', **ctx)
            for p in sorted((x/'logs').glob('*_process.json')):
                a.add(p, sample_dir/'Protocol/Runtime'/p.name, role='per_sample_stage_execution', **ctx)
            for folder, role in (('common_input/images', 'undistorted_RGB'), ('common_input/rap_masks', 'RAP_input_masks'),
                                 ('raw_rgb', 'raw_RGB'), ('semantic_masks', 'multi_view_semantic_mask_directory'),
                                 ('plant_only/projection_cache', 'projection_cache'), ('gaussian/original', 'training_output_directory'),
                                 ('logs', 'runtime_and_execution_logs'), ('common_input/sparse', 'native_COLMAP_input')):
                a.add(x/folder, role=role, **ctx)
            a.tree(x/'common_input/sparse', sample_dir/'Protocol/Used_COLMAP_Sparse', extensions={'.bin', '.txt'},
                   role='actual_COLMAP_camera_and_sparse_points', **ctx)
            for n in ('independent_sfm.json', 'mapper_complete.json', 'run-colmap-geometric.sh', 'run-colmap-photometric.sh'):
                a.add(x/'colmap'/n, sample_dir/'Protocol'/n, role='actual_COLMAP_runtime', **ctx)
            a.add(x/'colmap', role='independent_COLMAP_output', **ctx)
            gauss = x/'gaussian/original_gaussians.ply'
            cmgauss = x/'gaussian/metric_gaussians.ply'
            gauss_dir = GAUSS/'Representative_Gaussian_Models'/tag/sample
            original_archive = a.add(gauss, gauss_dir/gauss.name if representative else None, role='native_Gaussian_PLY',
                                     unit='sfm_units', required=True, **ctx)
            metric_archive = a.add(cmgauss, gauss_dir/cmgauss.name if representative else None, role='metric_Gaussian_PLY',
                                   unit='cm', required=is_metric, **ctx) if is_metric else ''
            for n in ('metric_cameras.json', 'metric_conversion.json', 'original/cameras.json', 'original/cfg_args',
                      'original/training_configuration.json', 'original/complete.json', 'original/verified_input.json'):
                a.add(x/'gaussian'/n, sample_dir/'Protocol/gaussian'/n, role='Gaussian_training_or_scale_metadata', **ctx)
            for p in (x/'gaussian/original').glob('**/*'):
                if p.suffix in {'.pth', '.pt'}:
                    a.add(p, role='Gaussian_training_checkpoint', **ctx)
            mesh = a.add(x/f'mesh/mesh_{unit}.ply', mdir/f'mesh_{unit}.ply', role='TSDF_mesh',
                         unit=real_unit, required=True, **ctx)
            plant = a.add(x/f'plant_only/plant_only_{unit}.ply', ORGANS/'Plant_Only'/tag/sample/f'plant_only_{unit}.ply',
                          role='plant_only_mesh', unit=real_unit, required=True, **ctx)
            a.add(x/f'plant_only/pot_{unit}.ply', ORGANS/'Plant_Only'/tag/sample/f'pot_{unit}.ply',
                  role='pot_only_mesh', unit=real_unit, **ctx)
            a.add(x/'mesh/raw_mesh.ply', role='raw_prefilter_mesh', unit=real_unit, **ctx)
            for n in ('input_index.json', 'complete.json', 'render_complete.json', 'renderer_units.json',
                      'surface/spatial_scale.json', 'source_indices.npz'):
                a.add(x/'mesh'/n, mdir/n, role='bounded_TSDF_parameters', **ctx)
            a.tree(x/'mesh/renderer_metadata', mdir/'renderer_metadata', extensions=SMALL, role='renderer_metadata', **ctx)
            view = json_data(x/'summary/preview_view.json').get('view', '')
            frames = x/'mesh/surface/frames'
            a.add(frames, role='all_depth_and_camera_frames', **ctx)
            if view:
                a.add(frames/f'{view}.npz', mdir/'Representative_Depth'/f'{view}.npz', unit=real_unit,
                      role='cached_masked_depth_K_c2w_RGB', view=view, **ctx)
                raw = image_for(x/('raw_rgb' if tomato else 'common_input/images'), view)
                if raw:
                    a.add(raw, sample_dir/'Figures'/f'camera_{view}_source{raw.suffix.lower()}',
                          role='representative_uncropped_input_RGB', view=view, **ctx)
            for rel in ('plant_only/triangle_semantics.npy', 'plant_only/semantic_votes.npz',
                        'plant_only/plant_only_source_indices.npz', 'plant_only/pot_source_indices.npz',
                        'leaf_instances/triangle_leaf_ids.npy', 'leaf_instances/leaf_instances.json', 'leaf_instances/candidates.json',
                        'compound_leaf_instances/compound_leaf_instances.json', 'compound_leaf_instances/triangle_leaf_ids.npy',
                        'compound_leaf_instances/leaf_instances.json', 'compound_leaf_instances/candidates.json'):
                a.add(x/rel, ORGANS/'Semantic_and_Instance_Labels'/tag/sample/rel, role='mesh_labels_and_indices', **ctx)
            cal = collect_calibration(a, x, sample, tomato, row, ctx)
            calibrations.append(cal)
            official = {}
            if tomato:
                ref_base = STD/sample/'annotations'
                for p in filesystem_files(ref_base):
                    rel = p.relative_to(ref_base.resolve())
                    reference_category = rel.parts[0]
                    # Preserve complete annotated point clouds and point labels for the public cohort.
                    location = a.add(ref_base/rel, TRAITS/'Reference_Data/TomatoPGT'/sample/rel, role='official_'+reference_category,
                                     unit='official_raw_coordinates_check_metadata' if reference_category in {'reference_cloud', 'semantic'} else '', **ctx)
                    official[reference_category+'_archive'] = location
                off_label = RAW/'04_annotations/instances/TomatoPGT'/f'{sample}_labels.csv'
                official['standardized_point_labels_archive'] = a.add(off_label, TRAITS/'Reference_Data/TomatoPGT'/sample/off_label.name,
                                                                       role='official_label_derivative', **ctx)
                official['registration_archive'] = a.add(REVIEW/sample/'registration.json',
                                                          data_dir/'compound_alignment/registration.json', role='alignment_only', **ctx)
                for n in ('registration.png', 'suggestions.csv', 'unused_predictions.csv'):
                    a.add(REVIEW/sample/n, data_dir/'compound_alignment'/n, role='reference_association', **ctx)
                official['capture_date_iso'] = datetime.strptime(sample.split('_')[-1], '%m%d%Y').date().isoformat()
                official['official_compounds_from_point_labels'] = compound_counts[sample]
            for folder, prefix in (('leaf_instances', 'Leaf_'), ('compound_leaf_instances', 'Compound_')):
                for p in sorted((x/folder).glob(f'{prefix}*')):
                    if not p.is_dir() or not (p/f'leaf_{unit}.ply').is_file():
                        continue
                    match = pot_keys.get((sample, p.name)) if not tomato else None
                    keep = bool(match) or (sample, p.name) in tom_keys
                    dest = (ORGANS/'Potted_Leaf_Instances'/sample/f"Handwritten_{match['handwritten_number']}_{p.name}" if match else
                            ORGANS/'Tomato_Compound_Leaves'/sample/p.name)
                    collect_instance(a, p, dest, p.name, unit, ctx, archive=keep, original_masks=keep)
                    if match:
                        original_rgb = Path(match['original_rgb_local_path'])
                        a.add(original_rgb, dest/f'Confirmed_View_Original_RGB{original_rgb.suffix.lower()}',
                              role='numbered_leaf_original_RGB', organ=p.name, view=match['reference_view'], required=True, **ctx)
            prep = dict(row, dataset=tag, experiment_version=version, unit=real_unit, sample_id=sample,
                        original_gaussian_source=str(gauss), metric_gaussian_source=str(cmgauss) if is_metric else '',
                        original_gaussian_archive=original_archive, metric_gaussian_archive=metric_archive,
                        tsdf_mesh_archive=mesh, plant_only_archive=plant, preview_view=view,
                        train_views_archive=str(sample_dir/'Protocol/train_views.txt'),
                        test_views_archive=str(sample_dir/'Protocol/test_views.txt'), calibration_archive=cal['scale_json'],
                        preview_rgb_archive=a.archived(x/'summary/rgb.png'), source_sample_directory=str(x), **official)
            prepared.append(prep)
        name = 'Tomato_42' if tomato else 'Potted_19'
        a.table(name+'_Reconstruction', prepared)
        a.table(name+'_Calibration', calibrations, SCALE/'Scale_Results')
        if tomato:
            metric_rows = [r for r in prepared if r['metric_mesh']=='True']
            a.table('Tomato_Metric_19_Reconstruction', metric_rows, TRAITS/'Tomato_Metric_19')
    return pot_matches, compounds


def phenotyping_tables(a, pot_matches, compounds):
    ps, ts = POT/'summary', TOM/'summary'
    a.tree(POT/'mesh_2dgs_official_voxel', MESH/'Mesh_Comparison/Potted_Finer_2DGS_Voxel_Separate',
           extensions=SMALL | IMAGES | {'.ply'}, version='potted_finer_voxel_separate', role='alternate_grid_mesh')
    a.add(ROOT/'植株数据.xlsx', TRAITS/'Reference_Data/Potted/historical_raw_plant_measurements.xlsx',
          role='historical_manual_reference', isolated=True)
    a.add(MANUSCRIPT/'植株数据.xlsx', TRAITS/'Reference_Data/Potted/manuscript_historical_measurements.xlsx',
          role='historical_manual_reference_edited_virtual_column', isolated=True)
    for p in (DATA/'00-规范与索引').glob('ForeSplat_*'):
        a.add(p, TRAITS/'Potted_19/Measurement_Protocol'/p.name, role='measurement_template_and_protocol', isolated=True)
    a.add(MANUSCRIPT/'A1_逐株标定记录.csv', SCALE/'Scale_Results/Potted_calibration_A1_original.csv', role='original_A1_calibration')
    a.add(RAW_TOM/'tomatoPGT-tools/README.pdf', TRAITS/'Reference_Data/TomatoPGT/TomatoPGT_Tools_Manual.pdf', role='reference_manual')
    a.add(RAW_TOM, role='TomatoPGT_raw_public_dataset', dataset='TomatoPGT', version=TOM_VER)
    a.add(ROOT/'experiments/ForeSplat_TomatoPGT_Pilot', role='historical_pilot_not_independent', version='tomato_reference_camera_pilot')
    ml = {(r['plant_id'], r['handwritten_number']): r for r in pot_matches}
    for orig, name in [('three_leaf_matches.csv', 'Potted_57_Leaf_Matching'),
                       ('three_leaf_measurements.csv', 'Potted_57_Trait_Measurements')]:
        records = []
        for r in csv_rows(ps/orig):
            m = ml[r['plant_id'], r['handwritten_number']]
            model = Path(m['model_local_path'])
            archive = a.archived(model)
            records.append(dict(r, dataset='Potted', experiment_version=POT_VER, unit='cm; area cm2',
                                model_source=str(model), model_archive=archive,
                                asset_folder=str(Path(archive).parent),
                                original_rgb_source=m['original_rgb_local_path'],
                                original_rgb_archive=a.archived(Path(m['original_rgb_local_path'])),
                                reference_rgb_archive=a.archived(model.parent/'reference_rgb.png'),
                                reference_overlay_archive=a.archived(model.parent/'reference.png'),
                                mask_folder_archive=str(Path(archive).parent/'masks'),
                                landmark_archive=a.archived(model.parent/'landmarks.json'),
                                reference_workbook_archive=str(TRAITS/'Reference_Data/Potted/historical_raw_plant_measurements.xlsx')))
        a.table(name, records)
    compounds_rows = []
    for r in compounds:
        if r['reference_scope'] != 'compound_leaf':
            continue
        scene = r['scene']
        crop = Path(r['model_local_path']) if r['model_local_path'] else None
        compounds_rows.append(dict(r, dataset='TomatoPGT', experiment_version=TOM_VER,
                                   unit='cm' if r['metric_scan']=='True' else 'sfm_units',
                                   model_archive=a.archived(crop) if crop else '',
                                   official_cloud_archive=a.archived(STD/scene/'annotations/reference_cloud/official_point_cloud.ply'),
                                   registration_archive=a.archived(REVIEW/scene/'registration.json'),
                                   overlay_archive=a.archived(REVIEW/scene/r['overlay']) if r['overlay'] else '',
                                   label_csv_archive=a.archived(RAW/'04_annotations/instances/TomatoPGT'/f'{scene}_labels.csv')))
    a.table('Tomato_245_Compound_Instances', compounds_rows)
    a.table('Tomato_70_Metric_Compounds', [r for r in compounds_rows if r['metric_scan']=='True'])
    pairs = csv_rows(ts/'final_leaf_trait_pairs.csv')
    for r in pairs:
        r['model_archive'] = a.archived(Path(r['model_ply'])) if r['model_ply'] else ''
        r['experiment_version'] = TOM_VER
    a.table('Tomato_Leaf_Trait_Pairs', pairs, TRAITS/'Trait_Error_Tables')
    a.table('Tomato_Leaf_Trait_Errors', [dict(r, dataset='TomatoPGT', experiment_version=TOM_VER,
                                           reference_source='TomatoPGT annotated point cloud and CloudGraph landmarks')
                                      for r in csv_rows(ts/'final_leaf_trait_errors.csv')], TRAITS/'Trait_Error_Tables')
    a.table('Tomato_Leaf_Plot_Data', [r for r in pairs if r['status']=='included_reference_guided_landmarks'], TRAITS/'Plot_Data')
    a.add(ts/'batch_height_pairs.csv', TRAITS/'Tomato_Metric_19/Initial_visible_height_comparison.csv', role='preliminary_height_only', version=TOM_VER)


def figure_plans(a, pot_matches, compounds):
    plans = (
        ('Workflow', '候选 1', '保留原 RAP-FSAM3 方框。新后端箭头以实际源码和配置重绘；此处不生成示意图代替实验。'),
        ('Frontend', '候选 2–4', '原稿 Fig. 2–4 原样保存；拆分素材按实际前端验证批次组织，勿把后续恢复链或四株表自动标成原稿 21 序列。'),
        ('Scale_Calibration', '候选 5', '盆栽用 2 cm 蓝块；番茄用有已知几何的平面标定靶。原生与厘米 PLY 仅比例不同。'),
        ('TSDF_Mesh', '候选 6A', '孔雀竹芋同一个 preview_view 的 RGB、GGGS、完整 TSDF、去盆表面、缓存深度和实际输入 mask。'),
        ('Organ_Segmentation', '候选 6B', '实例 mask、实际引用 RGB 和独立叶网格。实例参考视角可独立于整株预览视角，按 view_id 查找。'),
        ('Gaussian_Comparison', '候选 7', '孔雀竹芋单株的 183/27 视角 30K 对比，含同视角原图、渲染、深度与固定色标的像素差图；Mesh 比较与 Gaussian 比较分开统计。'),
        ('Potted_MultiSpecies', '候选 8', '不同叶形代表株；逐样本使用同相机的四阶段面板。原稿 Fig. 8 保留。'),
        ('Phenotyping', '候选 9', '参考引导端点的番茄复叶一致性；实线数据为现有有效配对，不包含叶面积误差；历史盆栽 Fig. 9 另存。'),
        ('Tomato_Application', '候选 10', '跨三个品种及三次采集日期，整株同行视角一致；复叶参考和端点图是独立视角或配准空间。未正式编号。'),
    )
    for key, provisional, caption in plans:
        a.figures[key] = dict(figure_group=key, provisional_number=provisional, caption_note=caption,
                              source_table=str(FIGS/key/'figure_sources.csv'))
        a.text(FIGS/key/'figure_caption_draft.md', f'# {key}（{provisional}）\n\n{caption}')
    def add(group, source, local=None, **meta):
        return a.add(source, FIGS/group/(local or source.name), **meta)

    for i in range(1, 10):
        group = {1:'Workflow', 2:'Frontend', 3:'Frontend', 4:'Frontend', 5:'Workflow', 6:'Workflow',
                 7:'Gaussian_Comparison', 8:'Potted_MultiSpecies', 9:'Phenotyping'}[i]
        add(group, MANUSCRIPT/f'Fig{i}.png', f'Original_Submission/Fig{i}.png', role='existing_manuscript_figure')
    add('Workflow', POT/'summary/run_configuration.json', role='potted_actual_workflow_configuration', version=POT_VER)
    add('Workflow', TOM/'README.md', 'Tomato_Independent_Workflow.md', role='tomato_actual_workflow_description', version=TOM_VER)
    for p in sorted((RECOVERY/'figures').glob('fig*.png')):
        add('Frontend', p, f'Recovered_Chain/{p.name}', role='frontend_recovery_figure', version=FRONT_VER)
    for section in ('gt_masks', 'selected_frames', 'method_masks', 'visual_assets/source_data'):
        a.tree(REP4/section, FIGS/'Frontend/Representative4'/section, extensions=IMAGES | SMALL,
               role='representative4_source_panel_or_data', version=FRONT_VER)
    for section in ('overlays', 'error_maps', 'gt_overlays'):
        for p in filesystem_files(REP4/'visual_assets'/section):
            if p.suffix.lower() in IMAGES and ('KongQueZhuYu' in p.name or 'CaoMei1' in p.name):
                rel = p.relative_to((REP4/'visual_assets').resolve())
                add('Frontend', REP4/'visual_assets'/rel, f'Representative4/visual_assets/{rel}',
                    role='frontend_overlay_or_error_panel', version=FRONT_VER)
    for p in ('gaussian_overview.png','gaussian_Depth.png','mesh_overview.png','final_selection.md','comparison_protocol.json'):
        add('Gaussian_Comparison', COMPARE/'summary'/p, role='selection_evidence', dataset='KongQueZhuYu selection', version=SEL_VER)
    rgb_pairs, reference_hashes = [], {}
    gaussian_fig = FIGS/'Gaussian_Comparison'
    comp_meta = dict(dataset='KongQueZhuYu selection', sample='KongQueZhuYu', version=SEL_VER)
    for name in ('2DGS', '3DGS', 'GGGS', 'MCMC_3DGS', 'gsplat'):
        evaluation = COMPARE/'gaussian'/name/'evaluation'
        dest = gaussian_fig/'Panel_Sources'/name
        for p in sorted(evaluation.glob('*.png')):
            a.add(p, dest/p.name, view=p.stem.split('_', 1)[0], role='same_view_comparison_panel', **comp_meta)
        reference_views = set()
        for original in sorted(evaluation.glob('*_Original.png')):
            view = original.stem.removesuffix('_Original')
            reference_views.add(view)
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            if reference_hashes.setdefault(view, digest) != digest:
                raise ValueError(f'Reference RGB differs across Gaussian methods for view {view}')
            prediction = evaluation/f'{view}_RGB.png'
            output = dest/f'{view}_RGB_Absolute_Error.png'
            rgb_residual_preview(original, prediction, a.out/output)
            a.generated(output, 'RGB_residual_from_saved_panels', source=prediction, view=view, **comp_meta)
            rgb_pairs.append(dict(method=name, sample_id=comp_meta['sample'], experiment_version=SEL_VER,
                                  view_id=view, reference_archive=str(dest/original.name),
                                  prediction_archive=str(dest/prediction.name), error_archive=str(output),
                                  color_map='magma', display_min=0, display_max=RGB_RESIDUAL_MAX,
                                  unit='8-bit channel levels', reference_source=str(original), prediction_source=str(prediction)))
        if not reference_views or reference_views != reference_hashes.keys():
            raise ValueError(f'Inconsistent reference view set for {name}')
    a.table('RGB_Residual_Sources', rgb_pairs, gaussian_fig)
    note = gaussian_fig/'RGB_Residual_Display.md'
    a.text(note, '# 五方法同视角像素差图\n\n'
           '逐像素计算已保存的预测图与原图之间三个 8 位 RGB 通道的绝对差均值。'
           f'所有方法共用 magma 色标，范围 0–{RGB_RESIDUAL_MAX} 灰阶（超过上限显示为最亮色）。'
           '显示整幅画面，不改变原来 PSNR、SSIM、LPIPS 的评价口径，不用于三维几何定量。'
           '\n\n原图、预测图和差值图的逐视角路径见 [RGB_Residual_Sources.csv](RGB_Residual_Sources.csv)。')
    a.generated(note, 'RGB_residual_display_protocol', **comp_meta)
    a.tree(COMPARE/'mesh/TSDF', FIGS/'Gaussian_Comparison/Mesh_TSDF_Panels', extensions=IMAGES,
           sample='KongQueZhuYu', version=SEL_VER, role='existing_TSDF_comparison_panel')

    def sample_panels(group, run, s, tomato, depth=False, models=True):
        x = run/s
        view = json_data(x/'summary/preview_view.json')['view']
        meta = dict(dataset='TomatoPGT' if tomato else 'Potted', version=TOM_VER if tomato else POT_VER, sample=s, view=view)
        for name, label in [('rgb','Original_RGB'),('gaussian','GGGS_Render'),('mesh','TSDF_Mesh_Render'),
                            ('plant_only','Plant_Only_Mesh_Render'),('instances','Instance_Overlay'),('measurement','Measurement_Preview')]:
            add(group, x/f'summary/{name}.png', f'{s}/{label}.png', role='same_camera_'+name, required=True, **meta)
        for name in ('preview_view.json', 'validation.json'):
            add(group, x/'summary'/name, f'{s}/{name}', role='camera_and_panel_provenance', **meta)
        raw = image_for(x/('raw_rgb' if tomato else 'common_input/images'), view)
        if raw:
            add(group, raw, f'{s}/Uncropped_Source_RGB{raw.suffix.lower()}', role='uncropped_input', **meta)
        rap = image_for(x/'common_input/rap_masks', view)
        if rap:
            add(group, rap, f'{s}/Input_Mask{rap.suffix.lower()}', role='RAP_input_mask', **meta)
        if models:
            for name, alias, role in (('gaussian/metric_gaussians.ply','Gaussian_Model_cm.ply','metric_Gaussian_PLY'),
                                       ('mesh/mesh_cm.ply','TSDF_Mesh_cm.ply','TSDF_mesh'),
                                       ('plant_only/plant_only_cm.ply','Plant_Only_Mesh_cm.ply','plant_only_mesh'),
                                       ('plant_only/pot_cm.ply','Pot_Only_Mesh_cm.ply','pot_mesh')):
                add(group, x/name, f'{s}/{alias}', role=role, unit='cm', **meta)
        if depth:
            src = x/'mesh/surface/frames'/f'{view}.npz'
            add(group, src, f'{s}/Cached_Depth_K_c2w_RGB.npz', role='existing_masked_render_depth', required=True, unit='cm', **meta)
            folder = FIGS/group/s
            depth_preview(src, a.out/folder)
            for name in ('Rendered_Depth.png', 'Valid_Depth_Mask.png', 'Depth_Display_Protocol.json'):
                a.generated(folder/name, 'derived_from_cached_depth', source=src, **meta, unit='cm')
        return meta

    add('Potted_MultiSpecies', POT/'summary/all19_overview.png', role='cohort_overview', version=POT_VER)
    for s in POT_REPS:
        sample_panels('Potted_MultiSpecies', POT, s, False)
    sample_panels('TSDF_Mesh', POT, 'KongQueZhuYu', False, depth=True)
    for s in ('CaoMei1', 'KongQueZhuYu') + TOM_REPS:
        tomato = s in TOM_REPS
        x = (TOM if tomato else POT)/s
        ctx = dict(dataset='TomatoPGT' if tomato else 'Potted', version=TOM_VER if tomato else POT_VER, sample=s)
        for name, role, unit in (('gaussian/original_gaussians.ply', 'native_Gaussian_PLY', 'sfm_units'),
                                 ('gaussian/metric_gaussians.ply', 'metric_Gaussian_PLY', 'cm'),
                                 ('calibration/scale_calibration.json', 'reference_geometry_and_qc', ''),
                                 ('calibration/board_matches.json', 'reference_target_pixels', ''),
                                 ('calibration/cube_corners.json', 'reference_cube_corners', ''),
                                 ('summary/gaussian.png', 'GGGS_preview', '')):
            add('Scale_Calibration', x/name, f'{s}/{Path(name).name}', role=role, unit=unit, **ctx)
        sc = json_data(x/'calibration/scale_calibration.json')
        for view in views_from(sc, tomato)[:3]:
            rgb = image_for(x/('raw_rgb' if tomato else 'common_input/images'), view)
            if rgb:
                add('Scale_Calibration', rgb, f'{s}/calib_{view}{rgb.suffix.lower()}', role='calibration_RGB', view=view, **ctx)
            if not tomato:
                add('Scale_Calibration', x/'semantic_masks'/view/'blue_cube.png', f'{s}/cube_mask_{view}.png',
                    role='cube_mask', view=view, **ctx)
    for r in pot_matches:
        if r['plant_id'] in POT_REPS and r['handwritten_number'] == '1':
            loc = Path(r['model_local_path'])
            a.tree(loc.parent, FIGS/'Organ_Segmentation'/r['plant_id']/loc.parent.name,
                   extensions=IMAGES | {'.ply','.json'}, role='independent_leaf_mesh_and_source_panels',
                   sample=r['plant_id'], organ=r['leaf_id'], version=POT_VER)
    for s in TOM_REPS:
        meta = sample_panels('Tomato_Application', TOM, s, True, depth=(s == TOM_REPS[0]))
        ref = STD/s/'annotations/reference_cloud/official_point_cloud.ply'
        add('Tomato_Application', ref, f'{s}/Official_Reference.ply', role='official_reference', **meta)
        add('Tomato_Application', REVIEW/s/'registration.png', f'{s}/Official_Predicted_Registration.png', role='alignment_preview', **meta)
        prepared = [r for r in compounds if r['scene']==s and r['reference_scope']=='compound_leaf' and r['candidate_id']]
        prepared.sort(key=lambda r: (r['status']!='strong_suggestion', not (r['model_chord_cm'] or r['model_transverse_width_cm']),
                                     -float(r['overlap_score'] or 0)))
        pick = next((r for r in prepared if s==TOM_REPS[0] and r['candidate_id']=='Compound_006'), None) or prepared[0]
        model = Path(pick['model_local_path'])
        group = f"{s}/Leaf_Inset_{pick['candidate_id']}"
        ctx = dict(dataset=meta['dataset'], version=meta['version'], sample=s, organ=pick['candidate_id'])
        for name, label, role in (('leaf_cm.ply','Leaf_Instance_Mesh_cm.ply','compound_leaf_mesh'),
                                  ('reference.png','Instance_Overlay.png','organ_overlay'),
                                  ('reference_rgb.png','Reference_RGB.png','organ_RGB'),
                                  ('instance.json','instance.json','instance_provenance')):
            add('Tomato_Application', model.parent/name, f'{group}/{label}', role=role, **ctx)
        if pick['overlay']:
            add('Tomato_Application', REVIEW/s/pick['overlay'], f'{group}/Measurement_or_Registration_Landmarks.png',
                role='existing_alignment_or_landmark_image', **ctx)
        a.table(f'{s}_Selected_Compound', [pick], FIGS/'Tomato_Application'/group)

    chart_dir = FIGS/'Phenotyping/Plot_Previews'
    (a.out/chart_dir).mkdir(parents=True)
    pairs = a.tables['Tomato_Leaf_Plot_Data']
    plot_pairs(a.out/pairs, a.out/chart_dir)
    for p in (a.out/chart_dir).iterdir():
        a.generated(chart_dir/p.name, 'rendered_plot', source=TOM/'summary/final_leaf_trait_pairs.csv',
                    dataset='TomatoPGT', version=TOM_VER)
    for key in ('Tomato_Leaf_Plot_Data', 'Tomato_Leaf_Trait_Errors', 'Potted_57_Trait_Measurements'):
        dest = FIGS/'Phenotyping/Source_Data'/f'{key}.csv'
        copy_material(a.out/a.tables[key], a.out/dest)
        a.generated(dest, 'plot_source_csv')

    for group, info in a.figures.items():
        target = FIGS/group
        entries = [dict(r, local_file=str(Path(r['archive_path']).relative_to(target))) for r in a.assets
                   if r['archive_path'].startswith(target.as_posix()+'/')]
        csv_save(a.out/info['source_table'], entries)
    a.text(FIGS/'Figure_Plan.md', '# 图件素材计划（编号为候选，未覆盖原稿）\n\n' + '\n'.join(
        f"- **{v['provisional_number']}** [{k}]({k}/figure_sources.csv)：{v['caption_note']}"
        for k,v in a.figures.items()))


def claim_matrix(a):
    def tab(key):
        return str(a.tables[key])
    value = [
        ('前端 RAP-FSAM3 对照', '原稿 21 条序列陈述；已定位同值源表为 4 株 / 20 帧',
         'F1 0.9706; mIoU 0.9641; BF1 0.4643',
         str(FRONT/'Evaluation_Data/Representative4/visual_assets/source_data/paper_summary_metrics.csv'),
         'Frontend', 'protocol_denominator_requires_reconciliation'),
        ('单株五 Gaussian 与七网格选型', '孔雀竹芋；183 训练视角 / 27 测试视角',
         '5 Gaussian；7 提取组；GGGS bounded TSDF Post 轮廓 IoU 0.89965（RAP 代理）',
         tab('Mesh_Method_Comparison'), 'Gaussian_Comparison', 'usable_as_single_scene_selection'),
        ('盆栽厘米重建', '19 株／9 类；开发 4 + 留出 15', '重建 19/19；公制 19/19；SAM3 蓝块轮廓 18/19，历史辅助 1/19',
         tab('Potted_19_Calibration'), 'Potted_MultiSpecies', 'batch_completion_and_calibration'),
        ('番茄独立重建和尺度', 'TomatoPGT 三个品种，42 扫描', '原 RGB 独立 COLMAP + GGGS + TSDF 42/42；公制 19/42',
         tab('Tomato_42_Reconstruction'), 'Tomato_Application', 'batch_completion_and_calibration'),
        ('盆栽独立叶实例及对应', '19 株', '798 自动候选；57/57 标号叶身份确认',
         tab('Potted_57_Leaf_Matching'), 'Organ_Segmentation', 'identity_not_segmentation_accuracy'),
        ('番茄复叶官方点标签与候选', '42 扫描；公制子集 19 扫描', '245 片复叶官方参考；184 有分配网格（含歧义）；公制复叶 70',
         tab('Tomato_245_Compound_Instances'), 'Tomato_Application', 'matching_status_not_standard_3D_IoU'),
        ('番茄复叶直线叶长配对一致性', '14 叶 / 8 扫描', 'MAE 0.202 cm；RMSE 0.301 cm；Bias -0.186 cm；R² 0.900',
         tab('Tomato_Leaf_Trait_Errors'), 'Phenotyping', 'reference_guided_not_automatic_accuracy'),
        ('番茄复叶同部位叶宽配对一致性', '24 叶 / 10 扫描', 'MAE 0.069 cm；RMSE 0.103 cm；Bias -0.045 cm；R² 0.968',
         tab('Tomato_Leaf_Trait_Errors'), 'Phenotyping', 'reference_guided_not_max_width_or_automatic_accuracy'),
        ('番茄整株株高候选', '18 扫描初步配对', '模型可见竖直跨度 vs 官方 true_height；定义和方向待统一',
         str(TRAITS/'Tomato_Metric_19/Initial_visible_height_comparison.csv'), 'Tomato_Application', 'preliminary_not_formal_error'),
        ('独立分割 IoU / 全队列画质评分', '盆栽和番茄分别评估', '待独立标注或统一方法对照',
         tab('Tomato_245_Compound_Instances'), 'Organ_Segmentation', 'not_evaluated'),
        ('新盆栽人工表型误差 / 统一冠幅 / 单侧面积误差', '各性状使用自己的有效配对', '尚无正式同口径独立配对或参考面积',
         tab('Potted_57_Trait_Measurements'), 'Phenotyping', 'not_evaluated'),
    ]
    rows = [dict(claim=c, dataset_and_denominator=d, quantitative_evidence=v, archive_source=s,
                 figure_sources=str(FIGS/f/'figure_sources.csv'), status=t) for c,d,v,s,f,t in value]
    a.table('Paper_Claim_Evidence', rows, INDEX)
    lines = ['# 论文结论与证据', '', '| 结论 | 队列与分母 | 数值或状态 | 表／图 | 证据状态 |', '| --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['claim']} | {r['dataset_and_denominator']} | {r['quantitative_evidence']} | "
                     f"[表](../{r['archive_source']}) · [图源](../{r['figure_sources']}) | {r['status']} |")
    a.text(INDEX/'PAPER_CLAIM_EVIDENCE_MATRIX.md', '\n'.join(lines))


def write_docs(a, head):
    a.text(INDEX/'START_HERE.md', f'''# ForeSplat 论文证据包

冻结来源：`{head}`。现有历史稿、正在修改的 Markdown、原始图、实际阶段产物都在包内；本次只归档和整理，未重新运行或修改模型。

1. 查样本、器官、素材类型、状态和可用表格：[EVIDENCE_MASTER_INDEX.xlsx](EVIDENCE_MASTER_INDEX.xlsx)。其 `Files` 页可筛数据集、版本、样本、器官、视角和素材角色。
2. 全部可定位文件与输入目录：[ALL_ASSETS_MANIFEST.csv](ALL_ASSETS_MANIFEST.csv)。有 `archive_path` 可离线打开；只有 `source_path` 的是服务器上的原数据或大型缓存。
3. 准备论文：[结论—证据](PAPER_CLAIM_EVIDENCE_MATRIX.md) · [实验版本](EXPERIMENT_VERSION_MAP.md) · [稿件修改映射](../01_MANUSCRIPT/Paper_Revision_Map.md) · [图片候选与分图原料](../08_PAPER_FIGURE_MATERIALS/Figure_Plan.md)。
4. 重做图表：[Master CSV](../09_SUPPLEMENTARY_MATERIALS/Per_Sample_Tables/) · [性状误差及配对](../07_PHENOTYPING_EVALUATION/Trait_Error_Tables/) · [绘图数据](../07_PHENOTYPING_EVALUATION/Plot_Data/) · [绘图脚本](../10_REPRODUCIBILITY/Plotting_Scripts/)。
5. 去掉服务器依赖给 AI：[便携阅读入口](../11_PORTABLE_RESEARCH_PACK/README_FOR_AI.md)，ZIP 在证据包根目录；读取索引时以解压目录为根。

| 位置 | 内容 |
| --- | --- |
| `01_MANUSCRIPT` | 已投稿 PDF、原稿 TeX、现改 Markdown、两版附图、参考文献 |
| `02_RAP_FSAM3_FRONTEND` | 前端代码、FFT、R1/SG-MPS/SB-FBPR、独立 GT、代表集预测及误差图 |
| `03_COLMAP_GGGS_RECONSTRUCTION` | 同协议的代表株五方法选型；完整训练日志位置及可看模型 |
| `04_METRIC_CALIBRATION` | 逐样本标定 JSON、标定视角、原有掩膜和质控 |
| `05_TSDF_MESH_RECONSTRUCTION` | 整株网格、深度缓存、实际体素与截断距离；历史 Mesh 分开 |
| `06_PLANT_AND_ORGAN_SEGMENTATION` | 去盆与花盆、57 片编号叶的自包含模型及参考图、番茄复叶、语义标号 |
| `07_PHENOTYPING_EVALUATION` | 原批次表、同口径番茄误差、全官方点云和逐点标签、历史人工参考 |
| `08_PAPER_FIGURE_MATERIALS` | 可按视角重新排图的组件，不覆盖原稿图号 |
| `09_SUPPLEMENTARY_MATERIALS` | 19+42 个样本预览、训练测试划分、旧 2DGS/M2M3-GO、Master Tables |
| `10_REPRODUCIBILITY` | 代码快照、实际配置和重绘脚本 |
| `11_PORTABLE_RESEARCH_PACK` | 文稿、原图、裁剪后的索引、配对统计和代表性三维资产 |

主要定量结果和暂缓项见 [整理报告](DELIVERY_REPORT.md)。硬链接是实际文件，可只读打开；编辑时请另存新文件。只有选定素材需要上传便携 ZIP，其余在本机可以逐样本浏览。
''')
    a.text(INDEX/'EXPERIMENT_VERSION_MAP.md', '''# 实验版本与适用范围

| 版本 | 样本 | 使用范围 |
| --- | --- | --- |
| `rap_fsam3_frontend` | 正文写 21 序列；同值源表 `Representative4` 明列 4 株 / 20 帧，另有 S23、固定轮廓、独立 GT 和 recovered chain | 分开保留设置、分母、人工 GT、R1/CDR、SG-MPS、SB-FBPR 等实际运行；正式引用前核对正文分母 |
| `legacy_2dgs_m2m3` | 旧稿重建与表型 21 条序列，旧版表型 21 整株及 63 叶；其它消融另有分母 | 旧 2DGS 优化、前景注入、M2M3-GO 和旧画质/速度/表型表均只对应旧版 |
| `kongquezhu_five_gaussian_selection_30k` | 单株孔雀竹芋，历史原生 COLMAP、183/27 分割 | 5 Gaussian 的画质与深度几何代理、7 组 Mesh 的 Raw/Post；不外推到其它样本 |
| `potted_gggs_tsdf_all19` | 9 类、19 株（开发 4、留出 15） | 原生 GGGS + 官方 bounded TSDF；2 cm 蓝块标定；表型候选、57 片已核对身份 |
| `potted_finer_voxel_separate` | 同 19 株的独立细体素版本 | 可作网格查看或另设比较，不能替换原 57 片的模型来源 |
| `tomatopgt_independent_gggs_tsdf_42` | TomatoPGT 三品种、42 扫描，19 个公制 | 使用自己的 RGB/COLMAP 位姿恢复，GGGS 与 bounded TSDF；官方点云、标签和骨架只用作参考/匹配，已知平面靶几何定尺度 |
| `tomato_reference_camera_pilot` | 历史试验 | 曾使用第三方相机，不与独立 42 扫描合并 |

`Compound_NNN` 是番茄复叶候选；`Leaf_NNN` 在番茄中是小叶候选，在盆栽中是独立叶候选。物理尺度与身份关联所用相似配准尺度是不同的字段，后者不用于再缩放公制模型。
''')
    a.text(INDEX/'DATA_DICTIONARY.md', '''# 数据字典

- **资产清单**：`source_path` 是原目录入口，`source_real_path` 是解析符号链接后的文件；`archive_path` 相对本包根目录。`server_index` 只有原路径；`hardlink` 和 `copy` 是包内实际文件；`generated` 是现有数据导出的索引或制图。`bytes` 是该文件逻辑大小，目录为 0。`view_id` 及手写编号保留原字符串。
- **重建表**：按 `sample_id` 查实验版本、物种/品种、相机注册、深度和网格来源。`capture_date_iso` 来自扫描 ID，不是独立株编号。`*_archive` 在证据包内，`*_source` 指服务器。
- **尺度**：`metric_mesh=True` 才能按 cm 测量；`sfm_units` 为未定标单位。`exploratory_scale_not_for_measurement` 只作排查。标定靶和方块的误差来自不同检验，不混算。标定 JSON 和 `reference_frame` 完整保留，竖直为候选方向的记录不能当作人工真值。
- **网格**：TSDF 的 `mesh/input_index.json`、`complete.json` 保存实际体素、截断距离、训练/测试视角、深度缩放和提取器。缓存 NPZ 原样保存 `depth`,`K`,`c2w`,`rgb`；制图的深度 PNG 按单视图截尾，数值取 NPZ，`Valid_Depth_Mask.png` 是缓存中的正深度支持。
- **编号叶**：`plant_id + handwritten_number` 指向实际 `leaf_id`；人工身份确认与历史参考行确认是不同的字段。历史表的旧虚拟值不作为新模型自动测量。
- **番茄复叶**：`scene + official_point_instance` 是官方逐点 ID，`official_id` 是可关联的 CloudGraph 骨架边，可能缺失。`overlap_score` 是距离阈值下双向覆盖率的调和均值，不是标准三维 IoU。复叶和子叶分类分别可从源复核表追踪，小叶另有独立索引。
- **误差**：`included_reference_guided_landmarks` 表示纳入复叶测量。`model` 与 `reference` 保留源精度，分别为原公制网格及官方参考端点的距离，`error_model_minus_reference` 保留符号。`r2_against_identity` 为 1−SSE/SST，不是回归拟合相关系数。单位长度 cm，面积 cm²；无同口径独立面积参考时误差留空。
''')
    a.text(INDEX/'SOURCE_PATHS.md', f'''# 来源地址

- Git 仓库：`https://github.com/LiuLiShenShe/test-froesplat`，原始快照 `{head}`。
- 当前稿：`{MANUSCRIPT}`；旧稿和前端恢复见 `{SOURCES}`。
- 历史五方法：`{COMPARE}`；其服务器实体路径经符号链接解析并保留于清单。
- 盆栽：`{POT}`；番茄独立批次：`{TOM}`。
- TomatoPGT 官方 DOI `10.17632/72md54c7n7.1`；本地原始 `02_raw/TomatoPGT`、标准化 `03_standardized/TomatoPGT` 与实例索引 `04_annotations/instances/TomatoPGT` 均以 `{RAW}` 为根。参考点云的原始坐标和 CloudGraph 转换元数据均照原文件保存。
- 本包 `archive_path` 以本证据包为根，ZIP 中 `portable_path` 以便携目录为根，均独立于服务器绝对路径。
- 查点位、照片、权重或非代表性 checkpoint 时，通过 `ALL_ASSETS_MANIFEST.csv` 的 `role` 和样本定位其物理目录。原运行报告的链接是历史路径；本包的 Master Tables/figure_sources 另提供可迁移的归档路径。
''')
    a.text(PAPER/'Paper_Revision_Map.md', '''# 原稿到现版修改映射（仅工作记录）

| 原稿部分 | 后续操作 | 证据 |
| --- | --- | --- |
| Introduction、RAP-FSAM3 方法、FFT、多提示、CDR/R1、SG-MPS、SB-FBPR | 保留方法贡献；核对前端数据分母后再统一摘要和结果文字 | [前端材料](../02_RAP_FSAM3_FRONTEND/) · [原 Fig. 2–4 与组件](../08_PAPER_FIGURE_MATERIALS/Frontend/) |
| Data / acquisition | 拆为旧前端验证、单株选型、19 株盆栽、42 次番茄独立扫描；区分发展/留出和相机训练/测试 | [样本表](../09_SUPPLEMENTARY_MATERIALS/Per_Sample_Tables/) |
| Foreground-object 2DGS、soft weighting、pruning、M2M3-GO | 原稿保持不变。新主后端叙述替换为真实 COLMAP + GGGS + bounded TSDF。旧 2DGS 消融可单独保留，不能当作 GGGS 算法组成或新贡献 | [五方法](../03_COLMAP_GGGS_RECONSTRUCTION/Five_Method_Comparison/) · [历史版本](../09_SUPPLEMENTARY_MATERIALS/Legacy_2DGS_and_M2M3/) |
| Mesh、scale、plant-only、器官 | 补两个不同的标定方案、厘米单位、TSDF 参数、花盆分离、复叶与小叶区分 | [尺度](../04_METRIC_CALIBRATION/) · [网格](../05_TSDF_MESH_RECONSTRUCTION/) · [分割](../06_PLANT_AND_ORGAN_SEGMENTATION/) |
| Results、Discussion | 盆栽跨形态展示与番茄公开数据应用分开；先报完成率和标定分母，再报条件尺寸一致性，独立三维 IoU 等空缺保持开放 | [结论证据](../00_README_AND_INDEX/PAPER_CLAIM_EVIDENCE_MATRIX.md) · [定量数据](../07_PHENOTYPING_EVALUATION/Trait_Error_Tables/) |
| Figures | 原 Fig. 1–9 不覆盖；候选编号与正式稿编号分离。番茄图仅提供组件，后续定稿时再编号 | [原图](Figures_Original/) · [新图组](../08_PAPER_FIGURE_MATERIALS/Figure_Plan.md) |

`Current_Paper/ForeSplat.md` 是重构工作稿；`Current_Paper/ForeSplat_中英文对照.md` 有近期表 7 文案；`Current_Paper/ForeSplat.tex` 是原有 2DGS 口径的 TeX 核对锚点，三者按原文件复制并均未改写。
''')


def cell_value(value):
    if not isinstance(value, str) or not value:
        return value
    if value.startswith('0') and value.isdigit() and len(value) > 1:
        return value
    return value[:32767]


def workbook_sheet(book, title, rows, link_prefix='..'):
    ws = book.create_sheet(title[:31])
    fields = list(dict.fromkeys(k for row in rows for k in row))
    ws.append(fields)
    for line, row in enumerate(rows, 2):
        ws.append([cell_value(row.get(k, '')) for k in fields])
        for column, k in enumerate(fields, 1):
            cell = ws.cell(line, column)
            if cell.data_type == 'f':
                cell.data_type = 's'
            if cell.value and (k.endswith('_archive') or k in {'archive_path', 'archive_source', 'figure_sources', 'source_table'}):
                cell.hyperlink = f'{link_prefix}/{cell.value}'
                cell.font = Font(color='0563C1', underline='single')
    for i, k in enumerate(fields, 1):
        h = ws.cell(1,i)
        h.font = Font(bold=True, color='FFFFFF')
        h.fill = PatternFill('solid', fgColor='155545')
        ws.column_dimensions[get_column_letter(i)].width = min(max(18, len(k)+2), 44)
    ws.freeze_panes = 'C2'
    ws.auto_filter.ref = ws.dimensions
    return ws


def workbook(a):
    w = Workbook()
    w.remove(w.active)
    workbook_sheet(w, 'Read first', [dict(table=name, archive_path=str(path)) for name,path in a.tables.items()])
    for name,path in a.tables.items():
        workbook_sheet(w, name, csv_rows(a.out/path))
    workbook_sheet(w, 'Paper figures', list(a.figures.values()))
    workbook_sheet(w, 'Files', a.assets)
    w.save(a.out/INDEX/'EVIDENCE_MASTER_INDEX.xlsx')


def portable(a):
    base = a.out/PORTABLE
    refs = []
    copied = set()
    def put(relative, local):
        if local in copied:
            return
        source = a.out/relative
        copy_material(source, base/local)
        copied.add(local)
        refs.append(dict(portable_path=local.as_posix(), package_path=str(relative),
                         original_source=str(a.destinations.get(str(relative), '')), bytes=source.stat().st_size))
    for rel in (PAPER/'Current_Paper', PAPER/'Figures_Original'):
        for p in (a.out/rel).iterdir():
            if p.is_file() and (p.suffix in {'.md','.tex','.bib','.png','.sty','.cls','.bst'}):
                put(rel/p.name, Path('Manuscript')/rel.name/p.name)
    for name,path in a.tables.items():
        put(path, Path('Research_Tables')/f'{name}.csv')
    for rel in (INDEX/'EXPERIMENT_VERSION_MAP.md', INDEX/'DATA_DICTIONARY.md', INDEX/'PAPER_CLAIM_EVIDENCE_MATRIX.md',
                INDEX/'SOURCE_PATHS.md', PAPER/'Paper_Revision_Map.md', FIGS/'Figure_Plan.md',
                REPRO/'Plotting_Scripts/plot_foresplat_evidence.py'):
        put(rel, Path('Source_References')/rel.name)
    # Mathematical conditions and exact evaluation provenance travel with the data.
    for src in (RECOVERY/'02_RAP_FSAM3_FULL_CHAIN_AUDIT.md', TOM/'README.md', TOM/'summary/final_leaf_trait_notes.md',
                COMPARE/'summary/final_selection.md', POT/'summary/run_configuration.json'):
        loc = a.archived(src)
        if loc:
            put(Path(loc), Path('Source_References')/('Tomato_Independent_README.md' if src.name == 'README.md' else src.name))
    for item in a.assets:
        p = Path(item['archive_path'])
        if not item['archive_path'] or not p.is_relative_to(FIGS):
            continue
        rel = p.relative_to(FIGS)
        name = rel.parts[0]
        if name in {'Potted_MultiSpecies','Tomato_Application','TSDF_Mesh','Scale_Calibration','Phenotyping'}:
            if item['kind'] == 'gaussian_ply' and not (name=='Tomato_Application' and item['sample_id']==TOM_REPS[0]):
                continue
            if item['kind'].endswith('_ply'):
                if name in {'Potted_MultiSpecies','Scale_Calibration','TSDF_Mesh'} and item['sample_id'] not in {POT_REPS[0], ''}:
                    continue
                if item['kind']=='point_cloud_ply' and item['sample_id']!=TOM_REPS[0]:
                    continue
                put(p, Path('Selected_Models')/rel)
            elif p.suffix in IMAGES | {'.pdf','.json','.npz','.csv'}:
                put(p, Path('Selected_Figures')/rel)
        elif name in {'Workflow','Organ_Segmentation'}:
            if p.suffix in IMAGES | {'.md','.json','.ply'}:
                put(p, Path('Selected_Models' if p.suffix=='.ply' else 'Selected_Figures')/rel)
        elif name in {'Frontend','Gaussian_Comparison'}:
            if len(rel.parts) <= 3 or (p.suffix in IMAGES and ('0000' in p.name or '0050' in p.name)):
                put(p, Path('Selected_Figures')/rel)
    # Include the actual source point labels and graph for the featured matched compound leaf.
    for p in filesystem_files(a.out/TRAITS/'Reference_Data/TomatoPGT'/TOM_REPS[0]):
        if p.parent.name in {'semantic','skeletons','phenotypes'}:
            put(p.relative_to(a.out), Path('Selected_Models/Official_BFS_06262025')/p.parent.name/p.name)
    for i in ('Manifest','Source'):
        name = 'ALL_ASSETS_MANIFEST.csv' if i=='Manifest' else 'PAPER_CLAIM_EVIDENCE_MATRIX.md'
        put(INDEX/name, Path('Source_References')/name)
    csv_save(base/'Source_References/PORTABLE_ASSET_MAP.csv', refs)
    a.text(PORTABLE/'README_FOR_AI.md', '''# ForeSplat 便携研究包：先读这里

ForeSplat 研究如何用多视角 RGB 和视觉基础模型提取可追溯的厘米植物三维器官。保留原有 RAP-FSAM3 前端；新批次是原生 COLMAP → GGGS → 两种公制恢复 → bounded TSDF → 去盆与多视角叶实例。旧 2DGS/M2M3-GO 是另一版本。

1. 先读 `Manuscript/Current_Paper/ForeSplat.md` 和中英文稿；TeX 是旧后端口径。各阶段范围在 [EXPERIMENT_VERSION_MAP.md](Source_References/EXPERIMENT_VERSION_MAP.md)，本包原图 Fig. 1–9 未覆盖。
2. `Research_Tables` 的 `Potted_19_Reconstruction.csv` 和 `Tomato_42_Reconstruction.csv` 分开使用。前者验证跨形态，后者是独立作物应用。盆栽 19/19 公制，番茄 19/42 公制。非公制扫描的 `units` 不能用于厘米比较。五方法/七 Mesh 提取比较仅孔雀竹芋，不是跨队列对照。
3. `Tomato_Leaf_Trait_Errors.csv` 和 `Tomato_Leaf_Plot_Data.csv` 支持参考引导端点一致性，直线长 14 对 / 8 扫描、叶宽 24 对 / 10 扫描。分割标准 IoU、盆栽新配对误差、面积误差及定义统一后的整株误差待补，不从代理量推算。
4. `Selected_Figures` 按主题和样本含可独立读取的 PNG/原始图；`Selected_Models` 含可直接打开的 Gaussian PLY、对应网格、器官和官方参考。整株同行依赖同一 `preview_view.json`；器官 inset 的视角可能不同。有关图片出处和离线可用性，使用 [PORTABLE_ASSET_MAP.csv](Source_References/PORTABLE_ASSET_MAP.csv) 的 `portable_path`；完整主档另有每幅图的 `figure_sources.csv`。
5. 先查看 `Research_Tables/Paper_Claim_Evidence.csv` 和 `Source_References/Paper_Revision_Map.md`。RAP 前端同值源表记录 4 株、20 个标注帧；原稿写 21 序列，待核对对应原始汇总。旧 2DGS 数值不能用于 GGGS。复叶和小叶分别处理。

本包不含全部大模型。Master Table 中 `*_archive` 路径以完整证据目录为根，需要在便携包内打开时按 `PORTABLE_ASSET_MAP.csv` 对应 `package_path` 和 `portable_path`；`*_source` 保留服务器路径以便补充。便携素材均为普通文件，无符号链接。绘图脚本 `Source_References/plot_foresplat_evidence.py` 可读取已有 CSV 或深度 NPZ，无需运行重建。
''')
    return base


def pack_zip(base, out):
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(base.rglob('*')):
            if not p.is_file():
                continue
            arcname = p.relative_to(base.parent).as_posix()
            if p.stat().st_size >= 8*MIB:
                info = zipfile.ZipInfo.from_file(p, arcname)
                info.compress_type = zipfile.ZIP_DEFLATED
                with p.open('rb') as f, z.open(info, 'w', force_zip64=True) as dest:
                    rate_transfer(f, dest)
            else:
                z.write(p, arcname)


def validate(a):
    names = Counter(r['kind'] for r in a.assets if r['archive_path'])
    originals = defaultdict(set)
    for r in a.assets:
        p = a.out/r['archive_path'] if r['archive_path'] else Path(r['source_path'])
        if not p.exists() or (r['archive_path'] and (p.is_symlink() or p.stat().st_size != r['bytes'])):
            raise ValueError(f'Missing or mismatched evidence: {p}')
        if r['source_real_path'] and Path(r['source_real_path']).is_file():
            stat = Path(r['source_real_path']).stat()
            originals[r['kind']].add((stat.st_dev,stat.st_ino))
    for table, expected in {'Potted_19_Reconstruction':19, 'Potted_19_Calibration':19, 'Potted_57_Leaf_Matching':57,
                            'Tomato_42_Reconstruction':42, 'Tomato_42_Calibration':42, 'Tomato_Metric_19_Reconstruction':19,
                            'Tomato_245_Compound_Instances':245, 'Tomato_70_Metric_Compounds':70,
                            'Tomato_Leaf_Trait_Pairs':210, 'Tomato_Leaf_Plot_Data':38}.items():
        rows = csv_rows(a.out/a.tables[table])
        if len(rows)!=expected:
            raise ValueError(f'{table}: {len(rows)} instead of {expected}')
    for row in csv_rows(a.out/a.tables['Potted_57_Leaf_Matching']):
        for k in ('model_archive','reference_rgb_archive','reference_overlay_archive','landmark_archive','mask_folder_archive'):
            if not row[k] or not (a.out/row[k]).exists():
                raise ValueError(f"{row['plant_id']} leaf {row['handwritten_number']} missing {k}")
    if any('mesh_cm' in row['tsdf_mesh_archive'] for row in csv_rows(a.out/a.tables['Tomato_42_Reconstruction'])
           if row['unit'] != 'cm'):
        raise ValueError('Nonmetric model labelled as metric')
    return names, originals


def report(a, names, originals, archive):
    rows = [r for r in a.assets if r['archive_path']]
    archived_unique = defaultdict(set)
    for r in rows:
        s = (a.out/r['archive_path']).stat()
        archived_unique[r['kind']].add((s.st_dev,s.st_ino))
    m = csv_rows(a.out/a.tables['Tomato_70_Metric_Compounds'])
    counts = dict(assets=len(a.assets), files_archived=len(rows), image_entries=names['image'],
                  distinct_archived_images=len(archived_unique['image']), gaussian_ply_archived=len(archived_unique['gaussian_ply']),
                  gaussian_ply_indexed=len(originals['gaussian_ply']), mesh_ply_archived=len(archived_unique['mesh_ply']),
                  mesh_ply_indexed=len(originals['mesh_ply']), official_clouds_archived=sum(r['role']=='official_reference_cloud' for r in rows),
                  potted_leaf_plies=57, metric_compound_plies=sum(bool(r['model_archive']) for r in m),
                  portable_zip_bytes=archive.stat().st_size)
    logical = sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file())
    counts['logical_bytes_with_aliases_and_zip'] = logical
    counts['directory_du_bytes_including_shared_hardlinks'] = int(subprocess.check_output(['du','-sB1',a.out], text=True).split()[0])
    source_inodes = set().union(*originals.values())
    all_entries = {(s.st_dev,s.st_ino):s for p in a.out.rglob('*') if p.is_file() for s in [p.stat()]}
    counts['new_file_disk_bytes_excluding_source_hardlinks'] = sum(s.st_blocks*512 for key,s in all_entries.items() if key not in source_inodes)
    a.text(INDEX/'DELIVERY_STATS.json', json.dumps(counts, ensure_ascii=False, indent=2))
    a.text(INDEX/'DELIVERY_REPORT.md', f'''# 整理交付记录

- 重建归档：19 株盆栽 + 42 扫描番茄；前端、单株选型及历史 2DGS 另有协议。两批共 38 样本具备厘米网格。
- 物理图像素材：{counts['distinct_archived_images']:,} 张（按实体文件去掉硬链接别名，含个体独立面板/原图、复核叠加和掩膜；完整量不等于主文最终选图），共 {names['image']:,} 条图片归档路径。
- Gaussian PLY：包内 {counts['gaussian_ply_archived']:,} 个实体，含原生、公制和五方法代表文件；全索引 {counts['gaussian_ply_indexed']:,} 个实体。Mesh PLY：包内 {counts['mesh_ply_archived']:,} 个实体，全索引 {counts['mesh_ply_indexed']:,} 个实体（含历史选型、独立器官及部分对照）。官方原始参考点云另有 {counts['official_clouds_archived']:,} 个。
- 57 片编号盆栽叶：57/57 的厘米模型、两套参考图、掩膜目录及端点 JSON 可直接定位。全 798 个盆栽候选的模型索引保留。番茄 245 个官方复叶记录中 184 个具有独立模型来源；公制 70 个复叶中 69 个有实际厘米网格归档（包括歧义行，未将歧义升级为人工确认）。所有预测复叶与小叶各自可筛。
- 可立即做图：原稿图与五方法单株对照（含同视角 RGB 像素差图）；19/19 与 19/42 公制覆盖；同视角 RGB/GGGS/TSDF/去盆样例；番茄参考引导端点一致性 14 长 / 24 宽。先核对前端 21 序列和同值 4 株 / 20 帧源表的关系。
- 待补：独立分割准确度、盆栽新版同株同叶实测对照、统一口径的整株误差、正式面积误差。番茄复叶匹配分数不是 IoU，误差结果不代表全自动端点定位。
- 完整目录 `du` 计入共享硬链接约 {counts['directory_du_bytes_including_shared_hardlinks']/MIB/1024:.2f} GiB；排除与原目录共享硬链接后，新文件块约 {counts['new_file_disk_bytes_excluding_source_hardlinks']/MIB:.1f} MiB（包括 ZIP；不计目录元数据）。ZIP `{archive.name}` 为 {archive.stat().st_size/MIB:.1f} MiB。

验证：逐项检查文件存在、真实副本字节数、全量表分母、57 片关键路径、未标定命名与 ZIP 清单。源码实验保持原状。
''')
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'ForeSplat_Paper_Evidence_Package')
    args = p.parse_args()
    out = args.output.resolve()
    temporary = out.with_name(out.name+'.building')
    if out.exists() or temporary.exists():
        p.error(f'Output already exists: {out} or {temporary}; use a new --output path.')
    temporary.mkdir(parents=True)
    a = Archive(temporary)
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    manuscript_hash = hashlib.sha256((MANUSCRIPT/'ForeSplat.tex').read_bytes()).hexdigest()
    print(f'{head}: manuscript, frontend, selection', flush=True)
    manuscript(a)
    frontend(a)
    selection(a)
    legacy_and_code(a)
    print('Archiving all 61 samples and official references', flush=True)
    pot, compounds = datasets(a)
    phenotyping_tables(a, pot, compounds)
    print('Building figure source sets, claim matrix and workbook', flush=True)
    figure_plans(a, pot, compounds)
    claim_matrix(a)
    write_docs(a, head)
    a.text(REPRO/'Data_Provenance/GIT_SNAPSHOT.json', json.dumps(dict(head=head, original_manuscript_sha256=manuscript_hash,
        origin='https://github.com/LiuLiShenShe/test-froesplat', source_project=str(ROOT)), indent=2))
    csv_save(a.out/INDEX/'ALL_ASSETS_MANIFEST.csv', a.assets)
    workbook(a)
    names, originals = validate(a)
    assert hashlib.sha256((MANUSCRIPT/'ForeSplat.tex').read_bytes()).hexdigest() == manuscript_hash
    print('Building standalone ZIP at 20 MiB/s', flush=True)
    base = portable(a)
    archive = a.out/'ForeSplat_Portable_Research_Pack.zip'
    pack_zip(base, archive)
    with zipfile.ZipFile(archive) as z:
        expected = {p.relative_to(a.out).as_posix():p.stat().st_size for p in base.rglob('*') if p.is_file()}
        assert {i.filename:i.file_size for i in z.infolist()} == expected
        assert z.read(f'{PORTABLE}/README_FOR_AI.md') == (base/'README_FOR_AI.md').read_bytes()
    counts = report(a, names, originals, archive)
    temporary.rename(out)
    print(json.dumps(counts, indent=2), flush=True)
    print(out, flush=True)


if __name__ == '__main__':
    main()
