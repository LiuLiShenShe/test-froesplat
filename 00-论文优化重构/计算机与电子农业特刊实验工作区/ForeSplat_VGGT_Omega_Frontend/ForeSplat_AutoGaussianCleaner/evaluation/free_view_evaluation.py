"""Automatically generated trajectory views with external-geometry support proxies.

Projected external point maps provide a diagnostic silhouette. A projected
external bounding box supplies a more conservative outside-support metric.
Neither proxy is used to certify foreground photometry or prune Gaussians.
"""
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
from ..cleaner.runtime import camera
from ..cleaner.io import write_json
from .metrics import arrays


def generate(scene, surface):
    if surface is None:
        return []
    centers = np.array([np.linalg.inv(v['E'])[:3, 3] for v in scene.views])
    target = np.median(surface['points'], axis=0)
    up = -np.median([v['E'][1, :3] for v in scene.views], axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    radial = centers-target
    planar = radial-np.outer(radial@up, up)
    axis = planar[np.argmax(np.linalg.norm(planar, axis=1))]
    axis /= max(np.linalg.norm(axis), 1e-9)
    second = np.cross(up, axis)
    distance = max(float(np.median(np.linalg.norm(radial, axis=1))), surface['extent']*.8)
    v = scene.views[len(scene.views)//2]
    definitions = [(f'orbit_{i:02d}', i*np.pi/2, .10, 1.) for i in range(4)]
    definitions += [('oblique', .4, .75, 1.), ('top', 0., 2., 1.), ('far', 1., .2, 1.5), ('near', 2., .15, .9)]
    result = []
    for name, angle, elevation, scale in definitions:
        direction = np.cos(angle)*axis+np.sin(angle)*second+elevation*up
        direction /= np.linalg.norm(direction)
        center = target + direction*distance*scale
        z = target-center
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        E = np.eye(4)
        E[:3, :3] = np.stack([x, y, z])
        E[:3, 3] = -E[:3, :3] @ center
        result.append(dict(name=name, E=E, K=v['K'], width=v['width'], height=v['height']))
    return result


def proxy(v, surface):
    p = surface['points'] @ v['E'][:3, :3].T + v['E'][:3, 3]
    p = p[p[:, 2] > .01]
    projected = p @ v['K'].T
    xy = projected[:, :2]/projected[:, 2, None]
    x, y = np.rint(xy).astype(int).T
    valid = (x >= 0) & (x < v['width']) & (y >= 0) & (y < v['height'])
    mask = np.zeros((v['height'], v['width']), bool)
    mask[y[valid], x[valid]] = True
    radius = max(3, round(surface['spacing']*v['K'][0, 0]/max(float(np.median(p[:, 2])), 1e-9)*4))
    radius = min(radius, round(np.sqrt(mask.size)*.03))
    mask = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=radius))
    # ALL external points, including extreme tips, define the box; robust quantiles don't remove them.
    lo, hi = surface['points'].min(0), surface['points'].max(0)
    corners = np.array([[i, j, k] for i in [lo[0], hi[0]] for j in [lo[1], hi[1]] for k in [lo[2], hi[2]]])
    corners = corners @ v['E'][:3, :3].T + v['E'][:3, 3]
    if np.any(corners[:, 2] <= .01):
        conservative = np.ones_like(mask)
    else:
        h = corners @ v['K'].T
        xy = h[:, :2]/h[:, 2, None]
        hull = ConvexHull(xy)
        im = Image.new('L', (v['width'], v['height']))
        ImageDraw.Draw(im).polygon([tuple(p) for p in xy[hull.vertices]], fill=255)
        conservative = ndimage.binary_dilation(np.asarray(im)>0, iterations=radius)
    return mask, conservative


def evaluate_free(scene, model, surface, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    definitions = generate(scene, surface)
    for uid, v in enumerate(definitions):
        c = camera(v['K'], v['E'], v['width'], v['height'], v['name'], uid)
        with torch.no_grad():
            a = arrays(scene.render(c, model))
        mask, box = proxy(v, surface)
        outside = ndimage.distance_transform_edt(~mask) > 3
        safe_outside = ~box
        records.append(dict(name=v['name'], proxy_external_alpha_mass=float(a['alpha'][outside].sum()),
            conservative_external_alpha_mass=float(a['alpha'][safe_outside].sum()),
            proxy_external_high_alpha_pixels=int((outside & (a['alpha'] > .5)).sum()),
            dark_proxy_external_high_alpha_pixels=int((outside & (a['alpha'] > .5) & (a['rgb'] @ [.2126, .7152, .0722] < .12)).sum()),
            support_proxy_pixels=int(mask.sum()), conservative_support_pixels=int(box.sum())))
        Image.fromarray((np.clip(a['rgb'], 0, 1)*255).astype(np.uint8)).save(output/f'{v["name"]}_rgb.png')
        Image.fromarray((a['alpha']*255).astype(np.uint8)).save(output/f'{v["name"]}_alpha.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(output/f'{v["name"]}_external_proxy.png')
    result = dict(status='measured_proxy' if definitions else 'unavailable_no_external_surface',
        views=records, definitions=definitions, reference='Independent front-end depth point map; sparse projection and conservative external AABB, neither is ground truth',
        proxy_high_alpha_pixels_sum=sum(r['proxy_external_high_alpha_pixels'] for r in records),
        dark_proxy_high_alpha_pixels_sum=sum(r['dark_proxy_external_high_alpha_pixels'] for r in records))
    write_json(output/'free_metrics.json', result)
    return result
