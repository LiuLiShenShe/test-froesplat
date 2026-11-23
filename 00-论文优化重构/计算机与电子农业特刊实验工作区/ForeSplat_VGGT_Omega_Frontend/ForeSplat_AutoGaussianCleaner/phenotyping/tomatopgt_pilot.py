"""Run an independent TomatoPGT COLMAP-to-phenotyping workflow."""
import argparse
import csv
import fcntl
import html
import itertools
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
                  OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')

import numpy as np
from scipy.optimize import least_squares

import batch_common as bc

SCAN = 'BFS_06262025'
MARKER_SCAN = 'BFS_05082025'
DATA = Path('/fj/end-paper/tomato_3d_benchmark')
RAW = DATA / '02_raw/TomatoPGT'
STANDARDIZED = DATA / '03_standardized/TomatoPGT'
CULTIVARS = {'BFS': ('BS', 'BEEF_STEAK'), 'BBH': ('BH', 'BIG_BOY_HYBRID'),
             'CH': ('CH', 'Celebrity_Hybrid')}


def raw_scan_name(scene):
    prefix, date = scene.split('_')
    return f'{CULTIVARS[prefix][0]}{int(date[:2])}_{int(date[2:4])}_{date[4:]}'


SCANS = {p.name: raw_scan_name(p.name) for p in sorted(STANDARDIZED.iterdir())
         if p.is_dir() and (p / 'metadata.json').is_file()}
PROJECT = RAW / 'sample_MetashapePRJ_Beef_Steak_5_8_2025'
CHUNK = PROJECT / 'Beef_Steak_5_8_2025.files/0'
OUT = bc.ROOT / 'experiments/ForeSplat_TomatoPGT_Independent'
STAGES = ('colmap', 'calibration', 'sam3', 'gggs', 'metric', 'tsdf', 'instances',
          'phenotype', 'manual', 'report', 'validation', 'compound', 'comparison')
MARKERS = {'colmap': 'common_input/complete.json', 'calibration': 'calibration/scale_calibration.json',
           'sam3': 'semantic_masks/complete.json', 'gggs': 'gaussian/original/complete.json',
           'metric': 'gaussian/metric_conversion.json', 'tsdf': 'mesh/complete.json',
           'instances': 'leaf_instances/complete.json', 'phenotype': 'phenotype/complete.json',
           'manual': 'phenotype/manual_pairing_audit.json', 'report': 'summary/report_complete.json',
           'validation': 'summary/validation.json', 'compound': 'compound_leaf_instances/complete.json',
           'comparison': 'summary/comparison.json'}


class TomatoContext(bc.Context):
    preserve_independent_plant_mask = True
    use_visible_height = True
    leaf_prompts = ('leaf', 'tomato leaflet')

    def __init__(self, plant=SCAN):
        self.plant = plant
        self.record = {'plant': plant, 'group': 'tomatopgt_independent_colmap'}
        self.out = OUT / plant
        self.common = self.out / 'common_input'
        self.raw = self.out / 'raw_rgb'
        self.original_rgb = RAW / SCANS[plant]
        self.standardized = STANDARDIZED / plant
        self.annotation = self.out / 'no_labelme_annotations'
        self.meter = bc.Meter()
        self.out.mkdir(parents=True, exist_ok=True)


def _tomato_foreground_mask(rgb):
    from scipy import ndimage

    rgb = np.asarray(rgb.convert('RGB'), dtype=np.float32)
    border = np.concatenate((rgb[:20].reshape(-1, 3), rgb[-20:].reshape(-1, 3),
                             rgb[:, :20].reshape(-1, 3), rgb[:, -20:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    mask = np.linalg.norm(rgb - background, axis=2) > 28
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((9, 9)))
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == 1 + int(np.argmax(sizes))
    return (mask * 255).astype(np.uint8)


def run_independent_colmap(ctx):
    """Reproduce the benchmark's masked-SfM recipe from this scan's raw RGB."""
    work = ctx.out / 'colmap'
    images, masks = work / 'sfm_images', work / 'masks'
    original = work / 'input'
    logs = work / 'logs'
    for folder in (images, masks, logs):
        folder.mkdir(parents=True, exist_ok=True)
    source_images = sorted(p for p in ctx.original_rgb.iterdir()
                           if p.suffix.lower() in ('.jpg', '.jpeg'))
    if not source_images:
        raise FileNotFoundError(f'No TomatoPGT RGB images: {ctx.original_rgb}')
    for source in source_images:
        name = source.stem + '.jpg'
        bc.link(ctx.raw / name, source)
        bc.link(original / source.name, source)

    database = work / 'database.db'
    sparse = work / 'working_sparse'
    raw_model = work / 'distorted/sparse/0'
    result_path = work / 'independent_sfm.json'
    if not result_path.is_file():
        dimensions = set()
        for source in source_images:
            with Image.open(source) as image:
                dimensions.add(image.size)
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                image.save(images / source.name, quality=95)
                mask = _tomato_foreground_mask(image)
                if not mask.any():
                    raise ValueError(f'{ctx.plant}: empty COLMAP foreground mask for {source.name}')
                mask_path = masks / f'{source.stem}.png'
                Image.fromarray(mask).save(mask_path)
                bc.link(masks / f'{source.name}.png', mask_path)
        if len(dimensions) != 1:
            raise ValueError(f'{ctx.plant}: mixed image dimensions with single_camera enabled')
        width, height = dimensions.pop()
        if not (work / 'mapper_complete.json').is_file():
            database.unlink(missing_ok=True)
            if sparse.exists():
                shutil.rmtree(sparse)
            sparse.mkdir(parents=True)
        commands = [
            ['colmap', 'feature_extractor', '--database_path', str(database),
             '--image_path', str(images), '--ImageReader.mask_path', str(masks),
             '--ImageReader.camera_model', 'SIMPLE_RADIAL', '--ImageReader.single_camera', '1',
             '--SiftExtraction.use_gpu', '0', '--SiftExtraction.max_image_size', '1600',
             '--SiftExtraction.max_num_features', '16384', '--SiftExtraction.num_threads', '8'],
            ['colmap', 'exhaustive_matcher', '--database_path', str(database),
             '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', '8'],
            ['colmap', 'mapper', '--database_path', str(database), '--image_path', str(images),
             '--output_path', str(sparse), '--Mapper.num_threads', '8'],
        ]
        def run(command):
            log_path = logs / f'{command[1]}.log'
            with log_path.open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, QT_QPA_PLATFORM='offscreen',
                                        CUDA_VISIBLE_DEVICES='1'))
        if not (work / 'mapper_complete.json').is_file():
            for command in commands:
                run(command)
            bc.write(work / 'mapper_complete.json', dict(commands=commands))
        io = bc.colmap_io()
        models = sorted(sparse.glob('*/images.bin'))
        if not models:
            raise RuntimeError(f'{ctx.plant}: independent COLMAP mapper produced no model')
        model = max(models, key=lambda path: len(io.read_extrinsics_binary(path)))
        registered = io.read_extrinsics_binary(model)
        if len(registered) / len(source_images) < .8:
            raise ValueError(f'{ctx.plant}: only {len(registered)}/{len(source_images)} images registered')
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(model.parent))
        camera = next(iter(reconstruction.cameras.values()))
        pixel_scale = np.array([width / camera.width, height / camera.height])
        camera.rescale(width, height)
        for image in reconstruction.images.values():
            for observation in image.points2D:
                observation.xy = observation.xy * pixel_scale
        raw_model.mkdir(parents=True, exist_ok=True)
        reconstruction.write_binary(str(raw_model))
        command = ['colmap', 'image_undistorter', '--image_path', str(original),
                   '--input_path', str(raw_model), '--output_path', str(work),
                   '--output_type', 'COLMAP', '--blank_pixels', '1', '--min_scale', '1',
                   '--max_scale', '1', '--max_image_size', '2000']
        run(command)
        exported = work / 'sparse'
        model_output = work / 'sparse/0'
        if not model_output.is_dir():
            model_output.mkdir(parents=True)
            for filename in ('cameras.bin', 'images.bin', 'points3D.bin'):
                (exported / filename).replace(model_output / filename)
        result = dict(status='complete', source='TomatoPGT raw RGB; independently run masked COLMAP',
                      recipe='benchmark prepare_colmap.py: resize to 1600, bg-distance mask(28), SIMPLE_RADIAL, '
                             'single_camera, CPU SIFT, exhaustive_matcher, mapper',
                      input_images=len(source_images), registered_images=len(registered),
                      selected_model=str(model.parent), raw_model=str(raw_model), mask_dir=str(masks),
                      camera_model='SIMPLE_RADIAL')
        bc.write(result_path, result)
        shutil.rmtree(images)
    import batch_prepare
    return batch_prepare.prepare(ctx, source=(work, raw_model))


def prepare(ctx):
    if (ctx.common / 'complete.json').is_file():
        return bc.read(ctx.common / 'complete.json')
    return run_independent_colmap(ctx)


def read_zip_xml(path):
    with ZipFile(path) as archive:
        return ET.fromstring(archive.read('doc.xml'))


def calibrate(ctx):
    import pycolmap

    output = ctx.out / 'calibration/scale_calibration.json'
    if output.is_file():
        return
    if ctx.plant != MARKER_SCAN:
        return calibrate_transferred(ctx, output)
    chunk = read_zip_xml(CHUNK / 'chunk.zip')
    frame = read_zip_xml(CHUNK / '0/frame.zip')
    camera_names = {c.attrib['camera_id']: Path(c.find('photo').attrib['path']).stem
                    for c in frame.findall('cameras/camera')}
    marker_ids = {m.attrib['label']: m.attrib['id'] for m in chunk.findall('markers/marker')}
    with (PROJECT / 'marker_points.csv').open(newline='') as stream:
        anchors = {row['marker_label']: 100 * np.array([float(row[k]) for k in ('x_m', 'y_m', 'z_m')])
                   for row in csv.DictReader(stream)
                   if row['scan_id'] == chunk.attrib['label'] and row['source'] == 'reference'}
    if len(anchors) != 3:
        raise ValueError('Expected the three published reference targets of the high resolution scan')
    data, views = ctx.cameras()
    camera = pycolmap.Camera(**data['source_camera'])
    estimated = {}
    for label in anchors:
        points, poses, names = [], [], []
        for loc in frame.findall(f"markers/marker[@marker_id='{marker_ids[label]}']/location"):
            name = camera_names.get(loc.attrib['camera_id'])
            if name not in views or loc.attrib.get('pinned') != 'true':
                continue
            points.append([float(loc.attrib['x']), float(loc.attrib['y'])])
            poses.append(np.column_stack([views[name]['R'], views[name]['tvec']]))
            names.append(name)
        if len(points) < 4:
            raise ValueError(f'{label} has only {len(points)} matched target observations')
        pixels, poses = np.asarray(points), np.asarray(poses)
        rays = np.asarray([camera.cam_from_img(p) for p in pixels])
        equations = (rays[..., None] * poses[:, None, 2] - poses[:, :2]).reshape(-1, 4)
        homogeneous = np.linalg.svd(equations, full_matrices=False)[2][-1]

        def residual(xyz):
            xyz_cam = poses[:, :, :3] @ xyz + poses[:, :, 3]
            predicted = [camera.img_from_cam(p[:2] / p[2]) for p in xyz_cam]
            return (predicted - pixels).ravel()

        xyz = least_squares(residual, homogeneous[:3] / homogeneous[3], loss='soft_l1', f_scale=3).x
        errors = np.linalg.norm(residual(xyz).reshape(-1, 2), axis=1)
        estimated[label] = dict(xyz_colmap_units=xyz, reference_xyz_cm=anchors[label], views=names,
                                errors_raw_px=errors, median_error_raw_px=np.median(errors),
                                positive_depth=bool(np.all(poses[:, 2, :3] @ xyz + poses[:, 2, 3] > 0)))
    baselines = []
    for a, b in itertools.combinations(anchors, 2):
        cm = np.linalg.norm(anchors[a] - anchors[b])
        units = np.linalg.norm(estimated[a]['xyz_colmap_units'] - estimated[b]['xyz_colmap_units'])
        baselines.append(dict(targets=[a, b], reference_cm=cm, reconstructed_units=units, scale=cm / units))
    scale = float(np.median([r['scale'] for r in baselines]))
    spread = max(abs(r['scale'] / scale - 1) for r in baselines)
    worst_median = max(r['median_error_raw_px'] for r in estimated.values())
    good = (np.isfinite(scale) and scale > 0 and spread <= .05 and worst_median <= 12
            and all(r['positive_depth'] for r in estimated.values()))
    status = 'CALIBRATED_MULTI_VIEW_TARGETS' if good else 'UNCALIBRATED_TARGET_QC_FAILED'
    bc.write(output, dict(status=status, scale_cm_per_unit=scale if good else None,
                          exploratory_scale_cm_per_unit=scale, method='Published planar target baselines; target pixels triangulated in actual COLMAP cameras; Metashape 3D poses not used',
                          coordinate_csv=str(PROJECT / 'marker_points.csv'), archived_pixels=str(CHUNK / '0/frame.zip'),
                          baselines=baselines, targets=estimated, max_relative_baseline_scale_deviation=spread,
                          max_target_median_reprojection_raw_px=worst_median,
                          qc=dict(max_relative_baseline_scale_deviation=.05, max_target_median_reprojection_raw_px=12,
                                  minimum_pinned_views_per_target=4)))
    ctx.status('calibration', 'complete' if good else 'uncalibrated', calibration_status=status,
               scale_cm_per_unit=scale if good else None)
    print(status, scale, 'baseline deviation', spread, 'max target median raw px', worst_median, flush=True)


def calibrate_transferred(ctx, output):
    from tomatopgt_targets import fit_points, transfer_pixels

    reference = TomatoContext(MARKER_SCAN)
    calibration_path = reference.out / 'calibration/scale_calibration.json'
    estimates = None
    matches = {}
    if calibration_path.is_file():
        calibration = bc.read(calibration_path)
        if bc.has_metric_scale(calibration):
            matches = transfer_pixels(reference, ctx, calibration)
            estimates = fit_points(ctx, matches, calibration)
    bc.write(output.with_name('board_matches.json'), matches)
    baselines = []
    scale = None
    spread = None
    if estimates:
        for a, b in itertools.combinations(estimates, 2):
            cm = np.linalg.norm(np.array(estimates[a]['reference_xyz_cm']) - estimates[b]['reference_xyz_cm'])
            units = np.linalg.norm(estimates[a]['xyz_colmap_units'] - estimates[b]['xyz_colmap_units'])
            baselines.append(dict(targets=[a, b], reference_cm=cm, reconstructed_units=units, scale=cm / units))
        scale = float(np.median([r['scale'] for r in baselines]))
        spread = max(abs(r['scale'] / scale - 1) for r in baselines)
    passed = (estimates and np.isfinite(scale) and scale > 0 and spread <= .05
              and all(r['positive_depth'] and r['median_error_undistorted_px'] <= 2 for r in estimates.values()))
    status = 'CALIBRATED_MULTI_VIEW_TARGETS' if passed else 'UNCALIBRATED_BOARD_TRANSFER_QC_FAILED'
    bc.write(output, dict(status=status, scale_cm_per_unit=scale if passed else None,
                          exploratory_scale_cm_per_unit=scale, matched_views=len(matches), targets=estimates,
                          baselines=baselines, max_relative_baseline_scale_deviation=spread,
                          method='Same printed board across scans: SIFT planar homographies transfer target pixels; independently triangulated in current COLMAP. Only documented reference baselines are used.',
                          reference_calibration=str(calibration_path)))
    ctx.status('calibration', 'complete' if passed else 'uncalibrated', calibration_status=status,
               scale_cm_per_unit=scale if passed else None)
    print(status, 'matched views', len(matches), 'scale', scale, 'baseline deviation', spread, flush=True)


def report(ctx):
    from batch_report import panels, viewer
    from PIL import Image, ImageDraw, ImageOps

    assets = OUT / 'summary/assets'
    assets.mkdir(parents=True, exist_ok=True)
    bc.link(assets / 'plotly.min.js', bc.OUT / 'summary/assets/plotly.min.js')
    catalog = bc.read(ctx.out / 'leaf_instances/leaf_instances.json')
    mesh = ctx.mesh()
    metric = catalog['metric']
    suffix = 'cm' if metric else 'units'
    semantic = bc.read(ctx.out / 'plant_only/complete.json')
    items = []
    for leaf in catalog['instances']:
        viewer(ctx, leaf, mesh, catalog_href='../../summary/index.html')
        ident = html.escape(leaf['leaf_id'])
        root = f'../leaf_instances/{ident}'
        items.append(f'<article><a href="{root}/viewer.html"><img loading="lazy" src="{root}/reference.png" alt="{ident}"><br>{ident} | 3D</a> <a href="{root}/leaf_{suffix}.ply">PLY</a></article>')
    panels(ctx)
    names = [('rgb', 'RGB'), ('gaussian', 'GGGS'), ('mesh', 'TSDF'),
             ('plant_only', 'Plant only'), ('instances', 'Leaf instances'),
             ('measurement', 'Candidate landmarks')]
    folder = ctx.out / 'summary'
    view = bc.read(folder / 'preview_view.json')['view']
    with Image.open(ctx.out / 'semantic_masks' / view / 'plant.png') as image:
        bbox = image.getbbox()
        if bbox:
            x0, y0, x1, y1 = bbox
            margin = max(20, round(.07 * max(x1-x0, y1-y0)))
            bbox = (max(0, x0-margin), max(0, y0-margin), min(image.width, x1+margin), min(image.height, y1+margin))
    canvas = Image.new('RGB', (1800, 485 if bbox else 250), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, label) in enumerate(names):
        with Image.open(folder / f'{name}.png') as image:
            thumb = ImageOps.contain(image.convert('RGB'), (294, 222))
            if bbox:
                closeup = ImageOps.contain(image.convert('RGB').crop(bbox), (294, 212))
                draw.text((i * 300 + 8, 253), 'Same-camera plant crop', fill='black')
                canvas.paste(closeup, (i * 300 + (300 - closeup.width) // 2, 272))
        draw.text((i * 300 + 8, 5), f'{label} / {view}', fill='black')
        canvas.paste(thumb, (i * 300 + (300 - thumb.width) // 2, 23))
    canvas.save(folder / 'overview.png')
    heldout = heldout_checks(ctx, folder)
    reference = official_reference(ctx)
    if reference:
        bc.write(folder / 'official_reference.json', reference)
    reference_html = '' if not reference else (
        f'<p>TomatoPGT reference point cloud: {reference["compound_leaf_count"]} compound leaves '
        f'(leaflets plus rachis); plant height {reference["plant_height_cm"]:.2f} cm. '
        'The mesh candidates have not been paired to these compound leaves. '
        '<a href="official_reference.json">Reference provenance and definitions</a></p>')
    (folder / 'index.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>ForeSplat TomatoPGT {ctx.plant}</title>'
        '<style>body{font:16px system-ui;margin:32px}img{width:100%;height:auto}section{display:grid;'
        'grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px}article img{aspect-ratio:4/3;object-fit:contain}</style>'
        f'<h1>ForeSplat TomatoPGT / {ctx.plant}</h1><p>{len(items)} candidates | {suffix}</p>'
        f'<img src="overview.png" alt="Reconstruction stages"><p><a href="../mesh/mesh_{suffix}.ply">TSDF PLY</a> | '
        f'<a href="../plant_only/plant_only_{suffix}.ply">Plant-only PLY</a> | '
        + (f'<a href="../plant_only/pot_{suffix}.ply">Pot PLY</a> | ' if semantic['pot_faces'] else '') +
        '<a href="../calibration/scale_calibration.json">Calibration</a> | '
        '<a href="../phenotype/leaf_measurements.csv">Leaf candidates CSV</a> | '
        '<a href="heldout_silhouettes.csv">Held-out silhouette scores</a></p>'
        + reference_html +
        '<a href="heldout_silhouettes.png"><img src="heldout_silhouettes.png" alt="Held-out RGB and silhouette overlays"></a><section>'
        + ''.join(items) + '</section>', encoding='utf-8')
    bc.write(folder / 'report_complete.json', dict(status='complete', leaf_viewers=len(items),
             heldout_silhouette_proxy=heldout))


def official_reference(ctx):
    prefix = ctx.plant.split('_')[0]
    root = RAW / CULTIVARS[prefix][1]
    traits_path = root / f'{prefix}_Phenotypes' / f'{ctx.plant}_annotated_ext_phenotypes.csv'
    graph_path = root / f'{prefix}_Graphs' / f'{ctx.plant}_annotated_ext_graph.json'
    if not (traits_path.is_file() and graph_path.is_file()):
        return None
    with traits_path.open(newline='') as stream:
        traits = list(csv.DictReader(stream))
    global_metrics = bc.read(graph_path)['meta']['global_metrics']
    leaves = [r for r in traits if r['Organ'] == 'CL']
    return dict(source='TomatoPGT CloudSeg manual structural labeling of the official point cloud, then CloudGraph traits.',
                definitions=str(RAW / 'tomatoPGT-tools/README.pdf'), traits_file=str(traits_path), graph_file=str(graph_path),
                compound_leaf_count=len(leaves), compound_leaf_traits=leaves,
                plant_height_cm=100 * global_metrics['true_height'],
                canopy_width_x_cm=100 * global_metrics['canopy_width_x'],
                canopy_depth_y_cm=100 * global_metrics['canopy_depth_y'],
                matching='No known instance correspondences, common vertical/base convention, or paired manual dimensions.')


def heldout_checks(ctx, folder):
    from batch_measure import csvwrite
    from PIL import Image, ImageDraw, ImageOps

    names = [Path(n).stem for n in (ctx.common / 'test_views.txt').read_text().splitlines()]
    semantic = np.load(ctx.out / 'plant_only/triangle_semantics.npy')
    rows = []
    plots = []
    for name in names:
        with np.load(ctx.out / 'plant_only/projection_cache' / f'{name}.npz') as data:
            ids = data['triangle_id']
        rendered = np.zeros(ids.shape, dtype=bool)
        visible = ids >= 0
        rendered[visible] = semantic[ids[visible]] == 1
        with Image.open(ctx.out / 'semantic_masks' / name / 'plant.png') as image:
            reference = np.array(image) > 0
        both = rendered & reference
        union = rendered | reference
        rows.append(dict(view=name, sam3_pixels=int(reference.sum()), mesh_pixels=int(rendered.sum()),
                         intersection_pixels=int(both.sum()), union_pixels=int(union.sum()),
                         iou=float(both.sum() / union.sum()) if union.any() else '',
                         sam3_coverage=float(both.sum() / reference.sum()) if reference.any() else ''))
        rgb = np.array(ctx.image(name))
        overlay = rgb.copy()
        for mask, color in [(both, [44, 160, 65]), (reference & ~rendered, [238, 160, 36]),
                            (rendered & ~reference, [102, 97, 229])]:
            overlay[mask] = .42 * overlay[mask] + .58 * np.asarray(color)
        plots.append((name, ImageOps.contain(Image.fromarray(overlay), (388, 260))))
    csvwrite(folder / 'heldout_silhouettes.csv', rows)
    canvas = Image.new('RGB', (1200, 40 + 290 * ((len(plots) + 2) // 3)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), 'Green: agreement | Orange: SAM3 only | Purple: mesh only', fill='black')
    for i, (name, tile) in enumerate(plots):
        x, y = 400 * (i % 3), 40 + 290 * (i // 3)
        label = f"{name}  IoU {rows[i]['iou']:.3f}" if rows[i]['iou'] != '' else name
        draw.text((x + 6, y), label, fill='black')
        canvas.paste(tile, (x + 6, y + 18))
    canvas.save(folder / 'heldout_silhouettes.png')
    ious = [r['iou'] for r in rows if r['iou'] != '']
    return dict(views=len(rows), mean_iou=float(np.mean(ious)) if ious else None,
                reference='SAM3 masks; holdout for training and TSDF fusion only, not for semantic assignment. Not independent ground truth.')


def segment(ctx):
    import batch_sam

    semantic = ctx.out / 'semantic_masks'
    marker = semantic / 'complete.json'
    if marker.is_file() and not bc.read(marker).get('preserve_independent_plant_mask'):
        archive = ctx.out / 'segmentation_attempts' / str(time.time_ns())
        archive.mkdir(parents=True)
        for relative in ('semantic_masks', 'leaf_instances/candidates', 'leaf_instances/candidates.json'):
            path = ctx.out / relative
            if path.exists():
                path.rename(archive / path.name)
    batch_sam.main(ctx)


def stage(name, plant):
    bc.Context = TomatoContext
    ctx = TomatoContext(plant)
    if name in ('compound', 'comparison'):
        import tomatopgt_evaluate

        return getattr(tomatopgt_evaluate, name)(ctx)
    if name in ('colmap', 'calibration', 'report', 'sam3'):
        return dict(colmap=prepare, calibration=calibrate, report=report, sam3=segment)[name](ctx)
    if name in ('tsdf', 'tsdf-fuse'):
        import batch_mesh

        if name == 'tsdf-fuse':
            return batch_mesh.main(ctx, fuse_only=True)
        # Resuming an earlier pilot may have used the unadapted SAM3 envelope.
        with (ctx.out / 'logs/sam3_geometry_input.log').open('a') as log:
            subprocess.run([str(bc.UV), 'run', '--no-project', '--python', str(bc.PYTHON),
                            str(Path(__file__).resolve()), '--plant', plant, '--stage-worker', 'sam3'],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        return batch_mesh.main(ctx, fuse_worker=[str(Path(__file__).resolve()), '--plant', plant,
                                                  '--stage-worker', 'tsdf-fuse'])
    modules = {'gggs': ('batch_train', 'main'),
               'metric': ('auto_metric_scale', 'apply'),
               'instances': ('batch_instances', 'main'), 'phenotype': ('batch_measure', 'main'),
               'manual': ('batch_manual', 'main'), 'validation': ('batch_validate', 'validate')}
    from importlib import import_module
    module, func = modules[name]
    getattr(import_module(module), func)(plant if name == 'gggs' else ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--plant', choices=SCANS, default=SCAN)
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--from-stage', choices=STAGES, default=STAGES[0])
    parser.add_argument('--until-stage', choices=STAGES)
    parser.add_argument('--stage-worker', choices=(*STAGES, 'tsdf-fuse'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage_worker:
        stage(args.stage_worker, args.plant)
        return
    summary = OUT / 'summary'
    summary.mkdir(parents=True, exist_ok=True)
    stages = STAGES[STAGES.index(args.from_stage):STAGES.index(args.until_stage or ('comparison' if args.all else 'validation')) + 1]
    if not stages:
        parser.error('--until-stage must not precede --from-stage')
    with (summary / 'pilot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.all:
            from tomatopgt_batch import run_all

            return run_all(stages)
        run_scene(TomatoContext(args.plant), stages, reuse=False)


def run_scene(ctx, stages, reuse):
    logs = ctx.out / 'logs'
    logs.mkdir(exist_ok=True)
    progress = OUT / 'summary/progress.json'
    for name in stages:
        marker = ctx.out / MARKERS[name]
        if reuse and marker.is_file() and (name != 'validation' or bc.read(marker).get('passed')):
            continue
        if reuse and name == 'metric' and not bc.has_metric_scale(bc.read(ctx.out / MARKERS['calibration'])):
            continue
        python = bc.GPU_PYTHON if name in ('gggs', 'tsdf') else bc.PYTHON
        command = [str(bc.UV), 'run', '--no-project', '--python', str(python),
                   str(Path(__file__).resolve()), '--plant', ctx.plant, '--stage-worker', name]
        log_path = logs / f'{name}.log'
        record = dict(plant=ctx.plant, stage=name, status='running', log=str(log_path),
                      driver_pid=os.getpid(), physical_gpu=1, started_unix=time.time(), command=command)
        bc.write(progress, record)
        print(f'TOMATOPGT {ctx.plant}: {name}', flush=True)
        with log_path.open('a') as log:
            result = subprocess.run(command, cwd=bc.ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(status='failed' if result.returncode else 'complete',
                      returncode=result.returncode, ended_unix=time.time())
        bc.write(logs / f'{name}_process.json', record)
        bc.write(progress, record)
        result.check_returncode()
    bc.write(progress, dict(plant=ctx.plant, status='finished', stage=stages[-1],
                           output=str(ctx.out), ended_unix=time.time()))


if __name__ == '__main__':
    main()
