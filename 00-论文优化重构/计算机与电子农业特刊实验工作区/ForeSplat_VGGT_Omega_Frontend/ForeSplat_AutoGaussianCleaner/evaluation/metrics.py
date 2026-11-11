from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..cleaner.foreground_protection import regions
from ..cleaner.io import write_json
from .artifact_metrics import artifact_metrics
from .foreground_metrics import photometry


def arrays(out):
    return {key: out[native].detach().cpu().numpy().transpose(1, 2, 0) if key in ['rgb', 'normal']
        else out[native].detach().cpu().numpy().squeeze()
        for key, native in [('rgb', 'render'), ('alpha', 'mask'), ('depth', 'median_depth'), ('normal', 'normal')]}


def evaluate(scene, model, folder, baseline=None, lpips_model=None, save_images=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(len(scene.views)):
        c, target, mask, depth, conf = scene.get_view(index)
        r = regions(mask, scene.config['boundary_relative_width'])
        with torch.no_grad():
            a = arrays(scene.render(c, model))
        if not all(np.isfinite(x).all() for x in a.values()):
            raise ValueError(f'Nonfinite render: {c.image_name}')
        metrics = dict(index=index, name=c.image_name, **photometry(a['rgb'], target, mask, r, a['alpha']),
            **artifact_metrics(a['alpha'], a['rgb'], r))
        if lpips_model is not None:
            # Both images have the same zero background; this is a foreground-masked LPIPS proxy.
            x = torch.as_tensor((a['rgb'] * mask[..., None]).transpose(2, 0, 1)[None].copy(), device='cuda') * 2 - 1
            y = torch.as_tensor((target * mask[..., None]).transpose(2, 0, 1)[None].copy(), device='cuda') * 2 - 1
            with torch.no_grad():
                metrics['fg_masked_lpips'] = float(lpips_model(x, y).mean())
        else:
            metrics['fg_masked_lpips'] = None
        if baseline is not None:
            with np.load(Path(baseline)/f'{index:04d}.npz') as z:
                old_depth = z['depth'].astype(np.float32)
                old_normal = z['normal'].astype(np.float32)
                old_alpha = z['alpha'].astype(np.float32)
            before_valid = mask & (old_alpha > .5) & (old_depth > 0)
            after_valid = mask & (a['alpha'] > .5) & (a['depth'] > 0)
            valid = before_valid & after_valid
            unit = max(float(np.median(old_depth[before_valid])), 1e-9) if before_valid.any() else 1.
            delta = np.abs(a['depth'][valid]-old_depth[valid])/unit
            metrics['depth_relative_p95'] = float(np.quantile(delta, .95)) if len(delta) else 0.
            metrics['depth_changed_fraction'] = float((delta > scene.config['quality']['depth_relative_p95']).mean()) if len(delta) else 0.
            metrics['new_depth_invalid_fraction'] = float((before_valid & ~after_valid).sum()/max(before_valid.sum(), 1))
            norms_old = np.linalg.norm(old_normal, axis=2)
            norms_new = np.linalg.norm(a['normal'], axis=2)
            nv = valid & (norms_old > .1) & (norms_new > .1)
            cos = np.sum(old_normal[nv]*a['normal'][nv], axis=1)/np.maximum(norms_old[nv]*norms_new[nv], 1e-9)
            metrics['normal_angle_p95_deg'] = float(np.quantile(np.degrees(np.arccos(np.clip(cos, -1, 1))), .95)) if len(cos) else 0.
        else:
            np.savez(folder/f'{index:04d}.npz', depth=a['depth'], normal=a['normal'], alpha=a['alpha'])
            metrics.update(depth_relative_p95=0., depth_changed_fraction=0., normal_angle_p95_deg=0., new_depth_invalid_fraction=0.)
        if save_images and index in scene.evidence_indices()[::max(1, len(scene.evidence_indices())//8)]:
            Image.fromarray((np.clip(a['rgb'], 0, 1)*255).astype(np.uint8)).save(folder/f'{c.image_name}_rgb.png')
            Image.fromarray((np.clip(a['alpha'], 0, 1)*255).astype(np.uint8)).save(folder/f'{c.image_name}_alpha.png')
            overlay = np.clip(a['rgb'], 0, 1).copy()
            overlay[r['boundary']] = [1, 0, 0]
            Image.fromarray((overlay*255).astype(np.uint8)).save(folder/f'{c.image_name}_boundary.png')
        records.append(metrics)
        if (index+1) % 32 == 0 or index+1 == len(scene.views):
            print(f'evaluation {folder.name} {index+1}/{len(scene.views)}', flush=True)
    keys = [k for k, v in records[0].items() if isinstance(v, (int, float)) and k != 'index']
    mean = {k: float(np.mean([r[k] for r in records if r[k] is not None])) for k in keys}
    for key in ['fg_masked_lpips', 'dark_foreground_alpha', 'dark_foreground_mse']:
        vals = [r[key] for r in records if r[key] is not None]
        mean[key] = float(np.mean(vals)) if vals else None
    result = dict(mean=mean, views=records, resolution_max_side=scene.config['evaluation_max_side'],
        thin_structure_definition='RAP skeleton low-thickness subset; no independent petiole/leaf semantic labels',
        lpips_status='measured_foreground_masked_proxy' if lpips_model is not None else 'unavailable')
    write_json(folder/'metrics.json', result)
    return result
