"""TomatoPGT compound-leaf candidates and reference comparisons."""
import csv
import fcntl
import html
import io
import sys

import numpy as np
from plyfile import PlyData

import batch_common as bc

ORGANS = 'compound_leaf_instances'
PROMPTS = ('compound leaf', 'tomato compound leaf', 'compound tomato leaf with leaflets and central rachis',
           'tomato leaf')
PROMPT_VERSION = 'sam3-compound-proposal-v2'


def compound_proposals(ctx, folder):
    import torch
    from PIL import Image

    path = folder / 'candidates.json'
    if path.is_file() and bc.read(path).get('protocol') == PROMPT_VERSION:
        return
    sam = bc.read(ctx.out / 'semantic_masks/complete.json')
    views = sam['key_views'] if 'key_views' in sam else sam['keyviews']
    results = []
    folder.mkdir(parents=True, exist_ok=True)
    old_path = folder / 'candidates_v1.json'
    if path.exists() and not old_path.exists():
        path.replace(old_path)
    with (bc.PRE / 'summary/gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.set_num_threads(8)
        torch.manual_seed(0)
        sys.path.insert(0, str(bc.ROOT / '第三方源码/SAM3-latest'))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        weight = bc.ROOT / 'sam3/sam3.pt'
        original = torch.load

        def load(path, *args, **kwargs):
            if str(getattr(path, 'name', '')) == str(weight):
                path = io.BytesIO(ctx.meter.read(weight))
            return original(path, *args, **kwargs)

        torch.load = load
        try:
            model = build_sam3_image_model(checkpoint_path=str(weight), load_from_HF=False, device='cuda')
        finally:
            torch.load = original
        processor = Sam3Processor(model, device='cuda', confidence_threshold=.3)
        for view in views:
            cache = folder / 'candidates' / view / 'complete.json'
            if cache.is_file() and bc.read(cache).get('protocol') == PROMPT_VERSION:
                results.extend(bc.read(cache)['proposals'])
                continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(ctx.out / 'semantic_masks' / view / 'plant.png') as image:
                plant = np.array(image) > 0
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                state = processor.set_image(ctx.image(view))
                detections = []
                for prompt in PROMPTS:
                    processor.reset_all_prompts(state)
                    obj = processor.set_text_prompt(prompt=prompt, state=state)
                    detections.extend((float(score), prompt, mask.squeeze(0).cpu().numpy() > 0)
                                      for mask, score in zip(obj['masks'], obj['scores']))
            saved, accepted, rejected = [], [], dict(low_score=0, invalid_area=0, low_plant_overlap=0, duplicate=0)
            pixels = max(1, plant.sum())
            for score, prompt, mask in sorted(detections, key=lambda entry: -entry[0]):
                count = mask.sum()
                clipped = mask & plant
                if score < .3:
                    rejected['low_score'] += 1
                    continue
                if not 100 <= count <= pixels*.8:
                    rejected['invalid_area'] += 1
                    continue
                if clipped.sum()/count < .4:
                    rejected['low_plant_overlap'] += 1
                    continue
                if any((clipped & x).sum() / (clipped | x).sum() > .8 for x in accepted):
                    rejected['duplicate'] += 1
                    continue
                candidate = cache.parent / f'candidate_{len(saved):03}.png'
                bc.png(candidate, clipped.astype(np.uint8)*255, ctx.meter)
                accepted.append(clipped)
                saved.append(dict(view=view, mask=str(candidate.relative_to(ctx.out)), score=score,
                                  prompt_type='SAM3 compound-leaf text; completeness unverified', text_prompt=prompt,
                                  mask_area_pixels=int(clipped.sum()), plant_mask_area_pixels=int(pixels)))
            bc.write(cache, dict(protocol=PROMPT_VERSION, proposals=saved, raw_detections=len(detections),
                                 rejected=rejected, prompt_scores={p: [score for score, name, mask in detections if p==name] for p in PROMPTS}))
            results.extend(saved)
            print('SAM3_COMPOUND', ctx.plant, view, len(saved), 'raw',len(detections), 'rejected',rejected, flush=True)
    bc.write(path, dict(protocol=PROMPT_VERSION, proposals=results, key_views=views, prompts=PROMPTS,
                        independent_from_leaflet_association=True))


def compound(ctx):
    from batch_instances import associate
    from batch_report import viewer

    folder = ctx.out / ORGANS
    done = folder / 'complete.json'
    ctx.status('compound', 'running')
    compound_proposals(ctx, folder)
    mesh = ctx.mesh()
    data = bc.read(done) if done.is_file() else associate(ctx, mesh,
        np.load(ctx.out / 'plant_only/triangle_semantics.npy'), organ_dir=ORGANS, id_prefix='Compound')
    items = []
    for row in data['instances']:
        viewer(ctx, row, mesh, catalog_href='../index.html', organ_dir=ORGANS)
        leaf = row['leaf_id']
        unit = 'cm' if row['metric'] else 'units'
        items.append(f'<article><a href="{leaf}/viewer.html"><img loading="lazy" src="{leaf}/reference.png" '
                     f'alt="{leaf}"><br>{leaf}</a> · <a href="{leaf}/leaf_{unit}.ply">PLY</a>'
                     f'<br>{row["faces"]} faces · {len(row["views"])} views</article>')
    (folder / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<style>body{font:15px system-ui;margin:24px}section{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}'
        'img{width:100%;aspect-ratio:3/2;object-fit:contain}</style>'
        f'<title>Compound leaves / {ctx.plant}</title><h1>{ctx.plant} / 复叶候选</h1>'
        f'<p>{len(items)} 个候选；复叶叶轴、器官完整性及官方身份待核对。</p>'
        '<p><a href="../summary/index.html">本扫描</a> · <a href="../../summary/batch_index.html">全部扫描</a> · '
        '<a href="../phenotype/compound_review.csv">配对复核表</a></p><section>' + ''.join(items) + '</section>', encoding='utf-8')


def visible_height(ctx, path, metric):
    unit = 'cm' if metric else 'units'
    verts = PlyData.read(ctx.out / f'plant_only/plant_only_{unit}.ply')['vertex']
    xyz = np.stack([verts[k] for k in ('x', 'y', 'z')], axis=1)
    vertical = np.asarray(bc.read(path)['vertical_unit_vector'])
    return float(np.ptp(xyz @ vertical))


def metrics(rows, pred, ref):
    xy = np.array([[r[pred], r[ref]] for r in rows if r.get(pred) is not None and r.get(ref) is not None])
    if not xy.size:
        return dict(n=0, mae=None, rmse=None, r2=None)
    error = xy[:, 0] - xy[:, 1]
    ss = np.square(xy[:, 1]-xy[:, 1].mean()).sum()
    return dict(n=len(xy), mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                r2=float(1 - np.square(error).sum()/ss) if len(xy) >= 2 and ss > 0 else None)


def comparison(ctx):
    from batch_measure import csvread, csvwrite
    from tomatopgt_pilot import official_reference

    folder = ctx.out / 'phenotype'
    summary = ctx.out / 'summary'
    metric = bc.read(ctx.out / 'mesh/complete.json')['metric']
    height = visible_height(ctx, ctx.out / 'calibration/reference_frame.json', metric)
    standard = folder / 'plant_measurements.csv'
    plant = csvread(standard)[0]
    if 'height_definition' not in plant:
        archive = folder / 'plant_measurements_pot_baseline.csv'
        if not archive.exists():
            csvwrite(archive, [plant])
    plant.update(plant_height_cm=height if metric else '', plant_height_model_units='' if metric else height,
                 height_definition='Visible plant mesh extent along estimated vertical: max(z)-min(z)',
                 measurement_status='VISIBLE_EXTENT_ALONG_ESTIMATED_VERTICAL' if metric else 'UNCALIBRATED_NO_CM_VALUES')
    csvwrite(standard, [plant])
    official = official_reference(ctx)
    observed = bc.read(ctx.out / ORGANS / 'complete.json')['instances']
    review = folder / 'compound_review.csv'
    reference_rows = {x['ID']: x for x in official['compound_leaf_traits']} if official else {}
    csvwrite(folder / 'official_compound_traits.csv', list(reference_rows.values()),
             fields=list(next(iter(reference_rows.values()))) if reference_rows else ['Organ','ID','Length (cm)','Thickness (cm)'])
    fields = ['candidate_id', 'official_id', 'identity_confirmed', 'same_cloudgraph_path_definition', 'length_cm', 'notes']
    if not review.exists():
        csvwrite(review, [dict(candidate_id=x['leaf_id']) for x in observed], fields=fields)
    pairs = []
    used = set()
    known = {x['leaf_id'] for x in observed}
    for row in csvread(review):
        if row.get('identity_confirmed','').strip().lower() not in ('true','1','yes'):
            continue
        ident, ref = row['candidate_id'], row['official_id']
        if ident not in known or ref not in reference_rows or ('candidate',ident) in used or ('reference',ref) in used:
            raise ValueError(f'Invalid or repeated confirmed compound identity {ident} -> {ref}')
        used.update([('candidate',ident), ('reference',ref)])
        same = row.get('same_cloudgraph_path_definition','').strip().lower() in ('true','1','yes')
        reference_length = reference_rows[ref].get('Length (cm)')
        length = float(row['length_cm']) if same and metric and row.get('length_cm') else None
        if length is not None and not (np.isfinite(length) and length > 0):
            raise ValueError(f'Invalid compound length for {ident}')
        pairs.append(dict(candidate_id=ident, official_id=ref, length_cm=length,
                          reference_length_cm=float(reference_length) if reference_length else None))
    height_pair = dict(prediction_cm=height if metric else None, official_cm=official['plant_height_cm'] if official else None,
                       difference_cm=height-official['plant_height_cm'] if metric and official else None,
                       prediction_definition=plant['height_definition'], reference_definition='TomatoPGT CloudGraph true_height')
    if official:
        bc.write(summary / 'official_reference.json', official)
    bc.write(summary / 'comparison.json', dict(metric=metric, visible_height=height_pair,
        official_compound_count=len(reference_rows) if official else None, compound_candidates=len(observed),
        compound_pairs=pairs, compound_matching_status=('confirmed_pairs' if pairs else
            'no_sam_compound_candidates' if official and not observed else
            'awaiting_identity_review' if official else 'reference_unavailable')))
    ctx.status('comparison','complete', metric=metric, compound_pairs=len(pairs))
