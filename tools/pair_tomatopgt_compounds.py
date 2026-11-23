"""Match existing ForeSplat compound meshes to TomatoPGT labeled organs (CPU only)."""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import csv
import html
from io import BytesIO
import json
from pathlib import Path
import os
from urllib.parse import quote, unquote

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/ForeSplat_TomatoPGT_Independent'
REF = Path('/fj/end-paper/tomato_3d_benchmark/03_standardized/TomatoPGT')
OUT = RUN / 'summary/compound_auto_review'
VERSION = 'geometry-v3'


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fields or ['scene', 'official_id', 'candidate_id'])
        writer.writeheader()
        writer.writerows(rows)


def xyz_ply(path, with_faces=False):
    ply = PlyData.read(path) if with_faces else PlyData.read(path, known_list_len={'face': {'vertex_indices': 3}})
    v = ply['vertex']
    p = np.column_stack([v[k] for k in 'xyz']).astype(float)
    if not with_faces:
        return p
    return p, np.stack(ply['face'].data['vertex_indices'])


def sample(p, n, step, rng):
    if not len(p):
        return p
    _, ids = np.unique(np.floor(p / step).astype(np.int32), axis=0, return_index=True)
    return p[rng.choice(ids, min(n, len(ids)), replace=False)]


def basis(up):
    z = np.asarray(up) / np.linalg.norm(up)
    x = np.eye(3)[np.argmin(np.abs(z))]
    x -= x.dot(z) * z
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def similarity(a, b, allow_scale=True):
    ac, bc = a.mean(0), b.mean(0)
    ar, br = a - ac, b - bc
    u, s, vt = np.linalg.svd(ar.T @ br)
    flip = np.sign(np.linalg.det(u @ vt))
    r = (u * [1, 1, flip]) @ vt
    k = (s @ [1, 1, flip]) / max(np.sum(ar**2), 1e-15) if allow_scale else 1.0
    return r, k, bc - k * (ac @ r)


def apply(p, tr):
    r, k, t = tr
    return k * (p @ r) + t


def distances(a, b, at, bt, tr):
    r, k, t = tr
    ab, ix = bt.query(apply(a, tr))
    ba, iy = at.query(((b - t) / k) @ r.T)
    return ab, ix, ba * k, iy


def fit(a, b, tr, iterations, trim):
    at, bt = cKDTree(a), cKDTree(b)
    for _ in range(iterations):
        ab, ix, ba, iy = distances(a, b, at, bt, tr)
        ma = ab <= np.quantile(ab, trim)
        mb = ba <= np.quantile(ba, trim)
        proposed = similarity(np.concatenate((a[ma], a[iy[mb]])), np.concatenate((b[ix[ma]], b[mb])))
        if not 0.55 <= proposed[1] <= 1.8 or proposed[0][2, 2] < 0.84:
            break
        tr = proposed
    ab, _, ba, _ = distances(a, b, at, bt, tr)
    # Keep outliers in the score, with bounded influence from incomplete geometry.
    score = np.mean(np.minimum(ab, .16)) + np.mean(np.minimum(ba, .16))
    return score / 2, tr


def register(a, b, rng):
    a0, b0 = a.mean(0), b.mean(0)
    ha = max(np.ptp(np.quantile(a[:, 2], [.02, .98])), 1e-10)
    hb = max(np.ptp(np.quantile(b[:, 2], [.02, .98])), 1e-10)
    a, b = (a - a0) / ha, (b - b0) / hb
    a1, b1 = sample(a, 1000, .022, rng), sample(b, 1300, .018, rng)
    a2, b2 = sample(a, 3800, .010, rng), sample(b, 4500, .008, rng)
    starts = []
    for angle in np.arange(0, 360, 10):
        r = Rotation.from_euler('z', angle, degrees=True).as_matrix()
        tr = r, 1., np.median(b1, 0) - np.median(a1 @ r, 0)
        starts.append(fit(a1, b1, tr, 7, .82))
    picked = []
    for _, tr in sorted(starts, key=lambda x: x[0]):
        if not any(np.trace(tr[0] @ r.T) > 2.88 for r, _, _ in picked):
            picked.append(tr)
        if len(picked) == 7:
            break
    refined = sorted((fit(a2, b2, tr, 25, .88) for tr in picked), key=lambda x: x[0])
    score, (r, k, t) = refined[0]
    alternative = next((ds for ds, (rr, _, _) in refined[1:] if np.trace(r @ rr.T) < 2.5), None)
    ab, _, ba, _ = distances(a2, b2, cKDTree(a2), cKDTree(b2), refined[0][1])
    stats = dict(normalized_distance=round(score, 5), median_pred_cm=round(float(np.median(ab) * hb), 4),
                 median_ref_cm=round(float(np.median(ba) * hb), 4), pred_near=float(np.mean(ab < .06)),
                 ref_near=float(np.mean(ba < .06)), alternate_gap=None if alternative is None else round(alternative - score, 5),
                 robust_ref_height_cm=float(hb))
    scale = k * hb / ha
    return (r, scale, b0 + t * hb - scale * (a0 @ r)), stats


def graph_reference(path, labels, groups):
    graph = json.loads(path.read_text()) if path else {}
    edges = [e for e in graph.get('edges', []) if e['type'] == 'CL' and len(e.get('anchors', [])) >= 2]
    meta = graph.get('meta', {}).get('transform', {})
    r = np.asarray(meta.get('rotate', np.eye(3)))
    t = np.asarray(meta.get('translate', np.zeros(3)))
    div = float(meta.get('scale_div', 1))
    if div <= 0:
        raise ValueError('invalid CloudGraph scale')
    nodes = {n['type']: n for n in graph.get('nodes', []) if n['type'] == 'ROOT'}
    root = np.array(nodes.get('ROOT', {}).get('pos', [0, 0, 0])) * 100
    if not edges or not groups:
        return (labels + t) @ r.T * (100 / div), {}, root, None
    best = None
    for rotation in (r.T, r):
        cloud = (labels + t) @ rotation * (100 / div)
        trees = [cKDTree(cloud[g]) for g in groups.values()]
        anchors = [np.asarray(e['anchors']) * 100 for e in edges]
        cost = np.array([[np.quantile(tree.query(a)[0], .65) + .35 * tree.query(a[-1])[0]
                          for a in anchors] for tree in trees])
        ir, ic = linear_sum_assignment(cost)
        value = np.median(cost[ir, ic])
        if best is None or value < best[0]:
            best = value, cloud, anchors, cost, dict(zip(ir, ic))
    value, cloud, anchors, cost, assignments = best
    mapping = {}
    height = max(np.ptp(cloud[:, 2]), 1.)
    for i, label in enumerate(groups):
        if i not in assignments:
            continue
        j = assignments[i]
        others = np.delete(cost[i], j)
        residual = cost[i, j]
        if residual <= max(.22, .032 * height) and (not len(others) or residual < .70 * np.min(others)):
            mapping[label] = (f'{edges[j]["source"]}-{edges[j]["target"]}', anchors[j], float(residual))
    return cloud, mapping, root, float(value)


def surface_samples(points, faces, n, rng):
    tri = points[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    total = area.sum()
    if not np.isfinite(total) or total <= 0:
        return points, None
    pick = rng.choice(len(tri), n, p=area / total)
    uv = rng.random((n, 2))
    flip = uv.sum(1) > 1
    uv[flip] = 1 - uv[flip]
    a, b, c = tri[pick].transpose(1, 0, 2)
    return np.concatenate((points, a + uv[:, :1] * (b-a) + uv[:, 1:] * (c-a))), float(total)


def shared_endpoints(ref, anchors):
    base, tip = anchors[0], anchors[-1]
    endpoints = {'length': np.stack((base, tip))}
    length = np.linalg.norm(tip - base)
    if length <= .15:
        return endpoints
    axis = (tip - base) / length
    delta = ref - (base + tip) / 2
    along = delta @ axis
    band = ref[np.abs(along) < .09 * length]
    if len(band) >= 25:
        trans = band - band.mean(0)
        trans -= (trans @ axis)[:, None] * axis
        _, _, vt = np.linalg.svd(trans, full_matrices=False)
        order = np.argsort(trans @ vt[0])
        a, b = band[order[int(.02 * (len(order)-1))]], band[order[int(.98 * (len(order)-1))]]
        if np.linalg.norm(a - b) > .15:
            endpoints['width'] = np.stack((a, b))
    return endpoints


def fmt(value):
    return '' if value is None else round(float(value), 4)


def link(path, target):
    return quote(os.path.relpath(path, target), safe='/')


def sketch(path, official, predicted, ref_ends=(), pred_ends=()):
    img = Image.new('RGB', (636, 225), '#f6f8f7')
    d = ImageDraw.Draw(img)
    points = np.concatenate([p for p in (official, predicted) if len(p)])
    center = (points.max(0) + points.min(0)) / 2
    scl = 171 / max(np.ptp(points, axis=0).max(), .001)
    for i, (x, y, title) in enumerate(((0, 2, 'XZ'), (1, 2, 'YZ'), (0, 1, 'XY'))):
        ox, oy = i * 212 + 106, 105
        d.rectangle((i*212+3, 4, i*212+208, 202), fill='white', outline='#d8e0db')
        d.text((i*212+10, 11), title, fill='#25342c')
        for cloud, color in ((official, '#156a9a'), (predicted, '#d36222')):
            xy = (cloud[:, [x, y]] - center[[x, y]]) * [scl, -scl] + [ox, oy]
            d.point([tuple(p) for p in xy.astype(int)], fill=color)
        for ends, color in ((ref_ends, '#267845'), (pred_ends, '#af2b83')):
            for ep in ends:
                xy = (ep[:, [x, y]] - center[[x, y]]) * [scl, -scl] + [ox, oy]
                d.line([tuple(p) for p in xy], fill=color, width=2)
                for px, py in xy:
                    d.ellipse((px-2, py-2, px+2, py+2), fill=color)
    d.text((8, 209), 'Blue: official   Orange: prediction   Green: official endpoints   Pink: predicted endpoints', fill='#25342c')
    img.save(path)


def pair_scene(record, out):
    scene = record['scene']
    date = RUN / scene
    dest = out / scene
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(413)
    metric = record['metric_mesh'] == 'True'
    unit = 'cm' if metric else 'units'
    annotations = REF / scene / 'annotations'
    labels_file = next((annotations / 'semantic').glob('*_annotated_ext.txt'))
    graph_path = next((annotations / 'skeletons').glob('*_annotated_ext_graph.json'), None)
    raw = np.loadtxt(labels_file, usecols=(0, 1, 2, 6, 7), dtype=np.float64)
    cls, ids = raw[:, 3:].astype(np.int32).T
    groups = {(int(c), int(i)): np.flatnonzero((cls == c) & (ids == i))
              for c in (1, 3) for i in np.unique(ids[cls == c]) if i >= 0}
    ref, official_ids, _, graph_error = graph_reference(graph_path, raw[:, :3], groups)
    official = ref[np.isin(cls, [0, 1, 2, 3, 6, 7, 8, 9])]
    if len(official) < 100:
        raise ValueError('insufficient labeled plant points')
    vertical = json.loads((date / 'calibration/reference_frame.json').read_text())['vertical_unit_vector']
    pred_basis = basis(vertical)
    full_path = date / 'plant_only' / f'plant_only_{unit}.ply'
    full = xyz_ply(full_path)
    fit_tr, qc = register(full @ pred_basis, official, rng)
    r, k, t = fit_tr
    tr = pred_basis @ r, k, t
    h = qc['robust_ref_height_cm']
    qc.update(scene=scene, metric=metric, version=VERSION, graph_label_median_residual_cm=graph_error,
              alignment_only_scale=float(k), rotation=tr[0].tolist(), translation_cm=t.tolist())
    good = qc['normalized_distance'] < .074 and min(qc['pred_near'], qc['ref_near']) >= .48
    clear = good and (qc['alternate_gap'] is None or qc['alternate_gap'] >= .009)
    qc['registration_status'] = 'usable' if clear else 'ambiguous' if good else 'weak'
    threshold = max(.14, .045 * h)
    qc['instance_distance_tolerance_cm'] = threshold
    p_files = sorted((date / 'compound_leaf_instances').glob(f'Compound_*/leaf_{unit}.ply'))
    labels = list(groups)
    refs = [sample(ref[groups[i]], 3500, max(.017, .003 * h), rng) for i in labels]
    refs_tree = [cKDTree(p) for p in refs]
    preds, areas = [], []
    for f in p_files:
        p, faces = xyz_ply(f, with_faces=True)
        points, area = surface_samples(p, faces, 3200, rng)
        preds.append(sample(apply(points, tr), 3500, max(.017, .003 * h), rng))
        areas.append(area)
    if refs and preds:
        pred_tree = [cKDTree(p) for p in preds]
        recall = np.array([[np.mean(pt.query(a)[0] <= threshold) for pt in pred_tree] for a in refs])
        precision = np.array([[np.mean(rt.query(b)[0] <= threshold) for b in preds] for rt in refs_tree])
        score = 2 * recall * precision / np.maximum(recall + precision, 1e-12)
        gain = np.maximum(score - .27, 0)
        ir, ic = linear_sum_assignment(np.column_stack((gain, np.zeros((len(refs), len(refs))))), maximize=True)
        assigned = {i:j for i, j in zip(ir, ic) if j < len(preds) and gain[i, j] > 0}
    else:
        recall = precision = score = np.empty((len(refs), 0))
        assigned = {}
        pred_tree = []
    traits = {x['ID']: x for x in read_csv(date / 'phenotype/official_compound_traits.csv')}
    rows = []
    overlay = dest / 'registration.png'
    sketch(overlay, sample(official, 9000, .015 * h, rng), sample(apply(full, tr), 9000, .015 * h, rng))
    for i, key in enumerate(labels):
        kind, label = key
        scope = 'compound_leaf' if kind == 1 else 'cotyledon'
        name = f'{kind}_{label}'
        ident, anchors, graph_residual = official_ids.get(key, ('', None, None))
        ref_trait = traits.get(ident, {})
        j = assigned.get(i)
        alt = sorted(range(len(preds)), key=lambda n: score[i, n], reverse=True)
        ranking = '; '.join(f'{p_files[n].parent.name}:{score[i,n]:.2f}' for n in alt[:3])
        row = dict(scene=scene, cultivar=record['cultivar'], metric_scan=metric, official_point_instance=label,
                   official_id=ident, reference_scope=scope, candidate_id='', status='', human_confirmed=False,
                   registration=qc['registration_status'], overlap_score='', reference_coverage='', prediction_coverage='',
                   runner_up_gap='', alternate_candidates=ranking, graph_link_residual_cm=fmt(graph_residual),
                   official_cloudgraph_length_cm=ref_trait.get('Length (cm)', ''),
                   official_cloudgraph_thickness_cm=ref_trait.get('Thickness (cm)', ''),
                   official_chord_cm='', model_chord_cm='', official_transverse_width_cm='', model_transverse_width_cm='',
                   model_observed_mesh_area_cm2='', reviewed_chord_pair=False, reviewed_width_pair=False,
                   endpoint_distances_cm='', endpoints_json='', model_ply='', reference_rgb='', model_viewer='', overlay='')
        ep = shared_endpoints(refs[i], anchors) if anchors is not None and metric else {}
        if 'length' in ep:
            row['official_chord_cm'] = fmt(np.linalg.norm(np.diff(ep['length'], axis=0)))
        if 'width' in ep:
            row['official_transverse_width_cm'] = fmt(np.linalg.norm(np.diff(ep['width'], axis=0)))
        if j is None:
            row['status'] = ('no_candidate' if not preds else 'registration_uncertain' if not good else
                             'competing_or_merged' if alt and score[i, alt[0]] >= .34 else 'no_reliable_match')
            if alt and score[i, alt[0]] >= .2:
                image = dest / f'label_{name}_alternative.png'
                sketch(image, refs[i], preds[alt[0]], list(ep.values()))
                row['overlay'] = image.name
        else:
            other = [score[i,n] for n in alt if n != j]
            rivals = np.delete(score[:,j], i)
            identity_gap = score[i,j] - max(rivals, default=0)
            gap = score[i,j] - max(other, default=0)
            strong = (clear and score[i,j] >= .72 and recall[i,j] >= .63 and precision[i,j] >= .70
                      and identity_gap >= .14 and gap >= -.012)
            plausible = good and score[i,j] >= .43 and gap >= -.04
            status = 'strong_suggestion' if strong else 'review_suggestion' if plausible else 'ambiguous_assignment'
            model = p_files[j]
            row.update(candidate_id=model.parent.name, status=status, overlap_score=fmt(score[i,j]),
                       reference_coverage=fmt(recall[i,j]), prediction_coverage=fmt(precision[i,j]),
                       runner_up_gap=fmt(gap), competing_reference_gap=fmt(identity_gap), model_ply=link(model, dest),
                       model_viewer=link(model.parent/'viewer.html', dest), reference_rgb=link(model.parent/'reference.png', dest),
                       model_observed_mesh_area_cm2=fmt(areas[j]) if metric else '')
            distances_by_trait, endpoints, mapped_ends = {}, {}, []
            if strong and metric and kind == 1:
                for trait, points in ep.items():
                    err, idx = pred_tree[j].query(points)
                    mapped = preds[j][idx]
                    length = np.linalg.norm(points[1] - points[0])
                    original = ((mapped - t) / k) @ tr[0].T
                    distances_by_trait[trait] = np.round(err, 4).tolist()
                    endpoints[trait] = dict(reference_cm=points.tolist(), prediction_cm=original.tolist())
                    if idx[0] != idx[1] and max(err) <= min(threshold, max(.12, .16 * length)):
                        key = 'model_chord_cm' if trait == 'length' else 'model_transverse_width_cm'
                        row[key] = fmt(np.linalg.norm(original[1] - original[0]))
                        mapped_ends.append(mapped)
            row['endpoint_distances_cm'] = json.dumps(distances_by_trait) if distances_by_trait else ''
            row['endpoints_json'] = json.dumps(endpoints) if endpoints else ''
            image = dest / f'label_{name}.png'
            sketch(image, refs[i], preds[j], list(ep.values()), mapped_ends)
            row['overlay'] = image.name
        rows.append(row)
    known = set(assigned.values())
    unused = [dict(candidate_id=p.parent.name, model_ply=link(p, dest),
                   viewer=link(p.parent/'viewer.html', dest), best_overlap=fmt(np.max(score[:,i])) if len(refs) else '')
              for i, p in enumerate(p_files) if i not in known]
    write_csv(dest / 'suggestions.csv', rows)
    write_csv(dest / 'unused_predictions.csv', unused)
    (dest / 'registration.json').write_text(json.dumps(qc, indent=2), encoding='utf-8')
    render_scene(dest, scene, rows, qc, unused, annotations)
    return rows, qc


STYLE = '''body{font:15px system-ui,sans-serif;color:#172925;margin:24px;background:#f7faf8}
a{color:#075985}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #d7e0db;vertical-align:top}
th{background:#e4efe8}img{max-width:100%;height:auto}.strong_suggestion{background:#e8f5ec}
.scroll{overflow:auto}.evidence{width:320px}small{color:#41524c}h1{font-size:25px}
#review{border-bottom:1px solid #d7e0db;margin-bottom:24px;padding-bottom:16px}
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}button,select{font:inherit;padding:5px 9px}
#accept{background:#176650;color:white;border:1px solid #176650;border-radius:4px}#accept:disabled{opacity:.45}
.pictures{display:grid;grid-template-columns:1.3fr 1fr;align-items:center;gap:16px;max-width:1080px}
.pictures img{aspect-ratio: 3 / 1;object-fit:contain;background:white;width:100%}.pictures #rgb{aspect-ratio:3 / 2}
[hidden]{display:none!important}.preview-only .interactive{display:none}
@media(max-width:650px){body{margin:12px}.pictures{grid-template-columns:1fr}}'''


def text(x):
    return html.escape(str(x))


def anchor(url, label):
    return f'<a href="{text(url)}">{text(label)}</a>' if url else ''


def page(dest, heading, body):
    dest.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>{text(heading)}</title><style>{STYLE}</style>'
                    f'<main><h1>{text(heading)}</h1>{body}</main></html>', encoding='utf-8')


def render_scene(dest, scene, rows, qc, unused, annotations):
    body = '<p>' + ' | '.join((anchor('../index.html','总览'), anchor('suggestions.csv','本扫描配对表'),
        anchor('unused_predictions.csv','未分配预测'), anchor('registration.json','配准质控'),
        anchor(link(annotations/'reference_cloud/official_point_cloud.ply', dest), '官方参考点云'))) + '</p>'
    body += f'<p>配准：{text(qc["registration_status"])}；蓝 = 官方；橙 = 预测；绿 = 参考端点；粉 = 预测端点。长度 / 宽度：模型 / 参考（cm）。数值为待复核估计。</p>'
    body += '<img src="registration.png" alt="整株配准对照"><div class="scroll"><table><thead><tr>'
    body += ''.join(f'<th>{x}</th>' for x in ('官方 ID / 点标签','预测及证据','状态','重叠 / 覆盖','直线叶长','横向叶宽','配准局部对照')) + '</tr></thead><tbody>'
    for r in rows:
        links = ' '.join((anchor(r['model_viewer'], r['candidate_id']), anchor(r['model_ply'],'PLY'), anchor(r['reference_rgb'],'RGB / mask')))
        pic = f'<a href="{text(r["overlay"])}"><img class="evidence" src="{text(r["overlay"])}" loading="lazy" alt="叶片对照"></a>' if r['overlay'] else ''
        body += (f'<tr class="{text(r["status"])}"><td>{text(r["official_id"] or "-")} / {r["official_point_instance"]}</td>'
                 f'<td>{links or "-"}<br><small>{text(r["alternate_candidates"])}</small></td><td>{text(r["status"])} / {r["reference_scope"]}</td>'
                 f'<td>{r["overlap_score"]} / {r["reference_coverage"]}, {r["prediction_coverage"]}</td>'
                 f'<td>{r["model_chord_cm"] or "-"} / {r["official_chord_cm"] or "-"}</td>'
                 f'<td>{r["model_transverse_width_cm"] or "-"} / {r["official_transverse_width_cm"] or "-"}</td><td>{pic}</td></tr>')
    page(dest/'index.html', scene, body+f'</tbody></table></div><p>未分配预测：{len(unused)}</p>')


def image_data(path, thumbnail=False):
    if thumbnail:
        with Image.open(path) as source, BytesIO() as buffer:
            preview = source.convert('RGB')
            preview.thumbnail((960, 720))
            preview.save(buffer, format='JPEG', quality=82, optimize=True)
            payload = buffer.getvalue()
        mime = 'image/jpeg'
    else:
        payload, mime = path.read_bytes(), 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(payload).decode('ascii')


def verification(rows, out):
    rows = sorted(rows, key=lambda r: ({'strong_suggestion':0, 'review_suggestion':1}.get(r['status'],2),
                                       r['metric_scan'] not in (True, 'True'), r['scene']))
    cache = {}
    previews = []
    for row in rows:
        item = dict(row)
        for field, thumbnail in (('overlay', False), ('reference_rgb', True)):
            if not row[field]:
                continue
            path = (out / row['scene'] / unquote(row[field])).resolve()
            if path not in cache:
                cache[path] = image_data(path, thumbnail) if path.is_file() else ''
            item[f'{field}_image'] = cache[path]
        previews.append(item)
    first = next((item for item in previews if item['status'] == 'strong_suggestion'
                  and item['reference_scope'] == 'compound_leaf' and item['metric_scan'] in (True, 'True')), {})
    title = f'{first["scene"]} · {first["official_id"] or first["official_point_instance"]} → {first["candidate_id"]}' if first else ''
    data = json.dumps(previews, ensure_ascii=False).replace('<', '\\u003c')
    script = Path(__file__).with_name('tomatopgt_compound_review.js').read_text(encoding='utf-8')

    def image_attr(field):
        return f'src="{first[field]}"' if first.get(field) else 'hidden'

    return f'''<section id="review" class="preview-only" data-version="{VERSION}">
      <p id="preview-note">预览可看首条图片；在浏览器中打开此 HTML 可切换叶片和保存核验。</p>
      <div class="toolbar interactive">
      <label>扫描 <select id="scene"><option value="">全部扫描</option></select></label>
      <label>范围 <select id="scope"><option value="metric">公制复叶</option><option value="">全部叶实例</option></select></label>
      <label>状态 <select id="status"><option value="strong_suggestion">优先核验</option>
      <option value="suggestions">全部建议</option><option value="">全部参考叶</option></select></label>
      <button id="prev">上一条</button><span id="position"></span><button id="next">下一条</button>
      <button id="export">导出核验 CSV</button></div>
      <div id="item"><h2 id="organ">{text(title)}</h2><p id="detail"></p><div class="pictures">
      <img id="overlay" {image_attr('overlay_image')} alt="蓝：官方；橙：预测；绿：参考端点；粉：预测端点">
      <img id="rgb" {image_attr('reference_rgb_image')} alt="预测叶片参考视图"></div>
      <p class="toolbar interactive"><a id="mesh" target="_blank" rel="noopener">预测 3D</a><a id="ply">预测 PLY</a>
      <a id="scan-page" target="_blank" rel="noopener">本扫描完整证据</a><span id="alternates"></span></p>
      <p id="dimensions"></p><div class="toolbar interactive"><label><input type="checkbox" id="verify-chord">已核验叶长端点</label>
      <label><input type="checkbox" id="verify-width">已核验叶宽端点</label></div>
      <p class="toolbar interactive"><button id="accept">确认对应</button><button id="reject">否决</button>
      <button id="defer">无法判定</button><button id="clear">撤销记录</button><span id="save-status" role="status"></span></p></div>
      <p id="error-summary" aria-live="polite"></p></section><script id="review-data" type="application/json">{data}</script><script>{script}</script>'''


def run(item):
    record, out, resume = item
    dest = out / record['scene']
    try:
        if resume and (dest/'index.html').exists() and (dest/'suggestions.csv').exists():
            qc = json.loads((dest/'registration.json').read_text())
            if qc.get('version') == VERSION:
                return read_csv(dest/'suggestions.csv'), qc, None
        rows, qc = pair_scene(record, out)
        print(record['scene'], qc['registration_status'], len(rows),
              sum(r['status'] == 'strong_suggestion' for r in rows), flush=True)
        return rows, qc, None
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'
        print(record['scene'], msg, flush=True)
        return [], {'scene':record['scene'], 'registration_status':'error'}, msg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = read_csv(RUN/'summary/batch_results.csv')
    records = [r for r in records if not args.scene or r['scene'] in args.scene]
    if not records:
        parser.error('no matching scenes')
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, [(r, out, args.resume) for r in records]))
    rows, summaries = [], []
    for matched, qc, error in results:
        for row in matched:
            row['model_local_path'] = str((out/row['scene']/row['model_ply']).resolve()) if row['model_ply'] else ''
            ratio = qc['alignment_only_scale']
            row['alignment_only_scale'] = fmt(ratio)
            row['metric_pair_qc'] = ('scale_review' if abs(ratio - 1) > .2 else 'eligible_for_review') if qc['metric'] else 'not_metric'
        rows.extend(matched)
        summaries.append(dict(scene=qc['scene'], registration=qc['registration_status'], error=error or '',
                              metric=qc.get('metric', ''), distance=qc.get('normalized_distance', ''),
                              reference_count=len(matched), official_graph_ids=sum(bool(r['official_id']) for r in matched),
                              strong=sum(r['status']=='strong_suggestion' for r in matched),
                              review=sum(r['status']=='review_suggestion' for r in matched),
                              ambiguous=sum(r['status']=='ambiguous_assignment' for r in matched),
                              not_assigned=sum(not r['candidate_id'] for r in matched),
                              chord_rows=sum(r['model_chord_cm'] != '' for r in matched),
                              width_rows=sum(r['model_transverse_width_cm'] != '' for r in matched)))
    write_csv(out/'compound_match_suggestions.csv', rows)
    write_csv(out/'provisional_leaf_measurements.csv', [r for r in rows if r['metric_pair_qc'] == 'eligible_for_review'
              and (r['model_chord_cm'] != '' or r['model_transverse_width_cm'] != '')])
    write_csv(out/'scan_summary.csv', summaries)
    body = '<p>自动建议不代表人工确认。优先核验强建议；低覆盖和配准不唯一的结果单列。原始 Mesh、标定和历史配对表未改。</p>'
    body += '<p>' + ' | '.join((anchor('scan_summary.csv','扫描汇总'), anchor('compound_match_suggestions.csv','全部建议'),
                                anchor('provisional_leaf_measurements.csv','暂定测量值'))) + '</p>'
    columns = ('scene','metric','registration','reference_count','official_graph_ids','strong','review','ambiguous','not_assigned','chord_rows','width_rows','error')
    labels = ('扫描','已标定','配准','参考点标签','有官方图 ID','强建议','待复核','歧义','无分配','叶长待复核','叶宽待复核','错误')
    body += verification(rows, out)
    body += '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in labels)+'</tr></thead><tbody>'
    for r in summaries:
        vals = [anchor(f'{r["scene"]}/index.html', r['scene']) if not r['error'] else text(r['scene'])] + [text(r[k]) for k in columns[1:]]
        body += '<tr>'+''.join(f'<td>{x}</td>' for x in vals)+'</tr>'
    page(out/'index.html', 'TomatoPGT 复叶配对核验', body+'</tbody></table></div>')
    print(f'{VERSION}: {len(summaries)} scans, {len(rows)} labeled organs, output {out}', flush=True)


if __name__ == '__main__':
    main()
