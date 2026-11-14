"""Common native/viewer export and fixed-camera visualization for all methods."""
from pathlib import Path
import json
import shutil
import argparse
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from ..io import save_ply, bake_viewer, write_json, sha
from ...evaluation.metrics import arrays


def export(output, original, rows, ids):
    output = Path(output)
    save_ply(output/'cleaned_gaussians.ply', rows)
    viewer, report = bake_viewer(rows)
    save_ply(output/'cleaned_gaussians_supersplat.ply', viewer)
    removed = np.ones(len(original), bool)
    removed[ids] = False
    save_ply(output/'removed_gaussians.ply', original[removed])
    # Removed native points also get a correctly baked viewing version.
    rv, _ = bake_viewer(original[removed])
    save_ply(output/'removed_gaussians_supersplat.ply', rv)
    np.save(output/'original_gaussian_ids.npy', ids)
    np.save(output/'removed_original_gaussian_ids.npy', np.flatnonzero(removed))
    report['native_sha256'] = sha(output/'cleaned_gaussians.ply')
    write_json(output/'viewer_baking.json', report)
    return report


def parity(scene, model, output):
    viewer, _ = scene.load_model(Path(output)/'cleaned_gaussians_supersplat.ply')
    records = []
    for vi in scene.evidence_indices()[::max(1, len(scene.evidence_indices())//4)][:4]:
        c, *_ = scene.get_view(vi)
        with torch.no_grad():
            a, b = scene.render(c, model), scene.render(c, viewer)
        rec = dict(index=vi)
        for key in ['render', 'mask']:
            d = (a[key]-b[key]).abs().flatten()
            rec[key] = dict(max=float(d.max()), mean=float(d.mean()), p99=float(torch.quantile(d, .99)),
                            fraction_gt_1_over_255=float((d > 1/255).float().mean()))
        records.append(rec)
    # Raster cutoff discontinuities may affect a few pixels after float32 bake.
    # Do not mistake one cutoff pixel for incorrect determinant compensation.
    passed = all(v['mean'] < 1e-5 and v['p99'] < 1e-4 and v['fraction_gt_1_over_255'] < 1e-4
                 for r in records for k, v in r.items() if k != 'index')
    report = json.loads((Path(output)/'viewer_baking.json').read_text())
    report.update(render_parity=records, parity_passed=passed,
                  parity_policy='bulk equivalence plus explicit cutoff-pixel fraction; maximum difference disclosed')
    write_json(Path(output)/'viewer_baking.json', report)
    if not passed:
        raise ValueError('GGGS filtered/viewer render bulk parity failed')


def image_pair(folder, name, a):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for background, value in [('black', a['rgb']), ('white', a['rgb']+(1-a['alpha'][..., None]))]:
        Image.fromarray(np.uint8(np.clip(value, 0, 1)*255)).save(folder/f'{name}_{background}.png')
    Image.fromarray(np.uint8(np.clip(a['alpha'], 0, 1)*255)).save(folder/f'{name}_alpha.png')


def visuals(scene, model, output, surface):
    from ...evaluation.free_view_evaluation import generate, proxy
    from ..runtime import camera
    output = Path(output)
    for vi in scene.evidence_indices()[::max(1, len(scene.evidence_indices())//8)][:8]:
        c, *_ = scene.get_view(vi)
        with torch.no_grad():
            a = arrays(scene.render(c, model))
        image_pair(output/'training', c.image_name, a)
    records = []
    definitions = generate(scene, surface)
    for uid, v in enumerate(definitions):
        c = camera(v['K'], v['E'], v['width'], v['height'], v['name'], uid)
        with torch.no_grad():
            a = arrays(scene.render(c, model))
        image_pair(output/'free', v['name'], a)
        mask, box = proxy(v, surface)
        occupied = ~box & (a['alpha'] > .5)
        _, count = ndimage.label(occupied)
        records.append(dict(name=v['name'], external_alpha=float(a['alpha'][~box].sum()),
            external_high_alpha_pixels=int(occupied.sum()), external_components=count,
            silhouette_proxy_high_alpha_pixels=int((~mask & (a['alpha'] > .5)).sum())))
    write_json(output/'automatic_free_views.json', dict(definitions=definitions, metrics=records,
        protocol='automatic_depth_trajectory_landscape_v2' if scene.config.get('free_view_aspect') else 'automatic_depth_trajectory_v1',
        viewport_aspect=scene.config.get('free_view_aspect'),
        reference='External depth AABB: conservative diagnostic proxy, not foreground ground truth'))
    # Historical eight cameras are evaluation-only; no IDs/ROIs from them enter C5.
    if scene.spec.get('historical_free_protocol'):
        protocol = json.loads(Path(scene.spec['historical_free_protocol']).read_text())
        hist = []
        for uid, v in enumerate(protocol['views']):
            c = argparse.Namespace(uid=uid, image_name=v['name'], image_width=v['width'], image_height=v['height'],
                FoVx=2*np.arctan(v['width']/(2*v['Fx'])), FoVy=2*np.arctan(v['height']/(2*v['Fy'])))
            for k in ['world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center']:
                setattr(c, k, torch.tensor(v[k], dtype=torch.float32, device='cuda'))
            with torch.no_grad():
                a = arrays(scene.render(c, model))
            image_pair(output/'historical_free', v['name'], a)
            cache = Path(scene.spec['historical_free_silhouettes'])/(v['name']+'.npz')
            with np.load(cache) as z:
                outside = ndimage.distance_transform_edt(~z['reference_mesh_silhouette']) > 5
            gray = a['rgb']+(1-a['alpha'][..., None])*protocol['background']
            hist.append(dict(name=v['name'], black_pixels=int((outside & (gray @ [.2126, .7152, .0722] < .12)).sum()),
                             background_alpha=float(a['alpha'][outside].sum())))
        write_json(output/'historical_free_metrics.json', dict(views=hist, black_pixels=sum(x['black_pixels'] for x in hist),
            reference='Fixed historical P3 silhouette proxy, evaluation only', protocol=scene.spec['historical_free_protocol']))
