"""Current-model native T*alpha attribution; no fitted mesh or fixed camera IDs."""
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from ..contribution_analysis import contribution
from ..foreground_protection import regions
from ..runtime import camera
from ..io import sha, write_json
from .camera_quadrature import representative_weights


def contribution_weights(scene,indices):
    if scene is not None and scene.config.get('contribution_aggregation')=='trajectory_voronoi':
        return representative_weights(scene,indices)
    return np.full(len(indices),1/max(len(indices),1))


def camera_indices(scene, count):
    previous = scene.config['evidence_views']
    try:
        scene.config['evidence_views'] = count
        return scene.evidence_indices()
    finally:
        scene.config['evidence_views'] = previous


def adaptive_floor(values):
    """Estimate a numerical/support floor in normalized alpha-pixel units."""
    positive = np.asarray(values)[np.asarray(values) > 1e-8]
    if not len(positive):
        return float('inf')
    # Low-tail mass rejects insignificant compositor tails. Unlike the old .05,
    # this follows each current model's measured contribution distribution.
    return max(float(np.quantile(positive, .20)), float(positive.sum())*1e-9)


def foreground_envelope(scene, model, indices, folder, identity):
    """A single broad original-model FG measurement; never a candidate208 loop."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    meta=folder/'identity.json';cache=folder/'foreground.npy'
    import json
    weights=contribution_weights(scene,indices)
    manifest={**identity,'indices':indices,'view_weights':weights.tolist(),'resolution_max_side':scene.config['evaluation_max_side']}
    if meta.exists() and cache.exists() and json.loads(meta.read_text())==manifest:
        return np.load(cache)
    fg=np.zeros(len(model.get_xyz),np.float64)
    for count,(index,weight) in enumerate(zip(indices,weights)):
        c,_,mask,*_=scene.get_view(index)
        g,_,_=contribution(c,model,np.stack([mask,mask,mask]))
        fg+=g[:,0]/(mask.size/(1080*1920))*weight
        if (count+1)%16==0 or count+1==len(indices):
            print(f'original foreground envelope {count+1}/{len(indices)}',flush=True)
    np.save(cache,fg);write_json(meta,manifest)
    return fg


def measure(scene, model, rows, ids, indices, free_views, folder, identity):
    """Only exact model+protocol identities may reuse attribution after restart."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ply_digest = identity['model_sha256']
    weights=contribution_weights(scene,indices)
    manifest = {**identity, 'indices': indices, 'view_weights':weights.tolist(), 'row_ids': ids.tolist(),
                'free_views': [s['view']['name'] for s in free_views]}
    meta, cache = folder/'identity.json', folder/'contribution.npz'
    import json
    if meta.exists() and cache.exists() and json.loads(meta.read_text()) == manifest:
        with np.load(cache) as z:
            return {k: z[k].copy() for k in z.files}
    n = len(ids)
    e = {k: np.zeros(n) for k in ['fg', 'bg', 'black', 'boundary', 'thin', 'free_bg', 'free_black', 'max_radius']}
    for k in ['fg_views', 'bg_views', 'depth_support', 'depth_observed']:
        e[k] = np.zeros(n, np.int32)
    xyz = np.column_stack([rows[k] for k in ['x', 'y', 'z']])
    scale = np.hypot(np.exp(np.column_stack([rows[f'scale_{k}'] for k in range(3)])), rows['filter_3D'][:, None])
    errors = []
    for vi,weight in zip(indices,weights):
        c, _, mask, depth, conf = scene.get_view(vi)
        r = regions(mask, scene.config['boundary_relative_width'])
        with torch.no_grad():
            out = scene.render(c, model)
            rgb = out['render'].detach().cpu().numpy().transpose(1, 2, 0)
            alpha = out['mask'].detach().cpu().numpy().squeeze()
        # Foreground and background share the SAME normalizer. A small leaf
        # silhouette must not silently rescale the historical harm ratio.
        norm = mask.size/(1080*1920)
        dark = r['bg'] & (alpha > .5) & ((rgb+(1-alpha[..., None])*.5) @ [.2126, .7152, .0722] < .12)
        g, radii, err = contribution(c, model, np.stack([mask, r['bg'], dark]))
        for j, k in enumerate(['fg', 'bg', 'black']):
            e[k] += g[:, j]/norm*weight
        e['fg_views'] += g[:, 0] > max(mask.sum(), 1)*1e-7
        e['bg_views'] += g[:, 1] > max(mask.sum(), 1)*1e-7
        e['max_radius'] = np.maximum(e['max_radius'], radii/np.sqrt(mask.size))
        g2, _, err2 = contribution(c, model, np.stack([r['boundary'], r['thin'], mask]))
        e['boundary'] += g2[:, 0]/norm*weight
        e['thin'] += g2[:, 1]/norm*weight
        errors.append(float(max(err.max(), err2.max())))
        if depth is not None:
            v = scene.views[vi]
            p = xyz @ v['E'][:3, :3].T+v['E'][:3, 3]
            h = p @ v['K'].T
            uv = np.rint(h[:, :2]/np.maximum(h[:, 2:3], 1e-9)).astype(np.int64)
            x, y = uv.T
            inside = (p[:, 2] > .01) & (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
            ii = np.flatnonzero(inside)
            x, y = x[ii], y[ii]
            valid = mask[y, x] & np.isfinite(depth[y, x]) & (depth[y, x] > 0) & np.isfinite(conf[y, x])
            all_valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
            if all_valid.any():
                valid &= conf[y, x] >= np.quantile(conf[all_valid], .2)
                tol = np.maximum(.03*np.median(depth[all_valid]), 2*scale[ii].max(1))
                # Hidden/back-side Gaussians are uncertain, not invalid.
                e['depth_observed'][ii] += valid & (p[ii, 2] <= depth[y, x]+tol)
                e['depth_support'][ii] += valid & (np.abs(p[ii, 2]-depth[y, x]) <= tol)
        print(f'historical attribution {len(errors)}/{len(indices)}', flush=True)
    for uid, support in enumerate(free_views):
        v = support['view']
        c = camera(v['K'], v['E'], v['width'], v['height'], v['name'], uid)
        with torch.no_grad():
            out = scene.render(c, model)
            rgb = out['render'].detach().cpu().numpy().transpose(1, 2, 0)
            alpha = out['mask'].detach().cpu().numpy().squeeze()
        outside = support['outside_surface']
        dark = outside & (alpha > .5) & ((rgb+(1-alpha[..., None])*.5) @ [.2126, .7152, .0722] < .12)
        g, _, err = contribution(c, model, np.stack([support['surface_mask'], outside, dark]))
        norm = outside.size/(1080*1920)*max(len(free_views), 1)
        e['free_bg'] += g[:, 1]/norm
        e['free_black'] += g[:, 2]/norm
        errors.append(float(err.max()))
    e['conservation_max_error'] = np.array(max(errors, default=0.))
    np.savez(cache, **e)
    write_json(meta, manifest)
    return e


def component_features(rows, e, surface):
    xyz = np.column_stack([rows[k] for k in ['x', 'y', 'z']]).astype(float)
    tree = cKDTree(xyz)
    d, neighbors = tree.query(xyz, k=min(5, len(xyz)), workers=8)
    d = np.atleast_2d(d) if len(xyz) == 1 else d
    if len(xyz) < 2:
        return dict(isolated=np.zeros(len(rows), bool), surface_distance=np.zeros(len(rows)), spacing=np.ones(len(rows)), components=np.zeros(len(rows), int))
    spacing = np.maximum(np.median(d[:, 1:], axis=1), 1e-12)
    # Mutual local-radius edges avoid connecting detached clusters through one
    # long KNN edge. Clusters are detected on every row, not only dark candidates.
    a = np.repeat(np.arange(len(xyz)), neighbors.shape[1]-1)
    b = neighbors[:, 1:].ravel()
    distance = d[:, 1:].ravel()
    link = distance <= 3*np.minimum(spacing[a], spacing[b])
    graph = coo_matrix((np.ones(int(link.sum()), bool), (a[link], b[link])), shape=(len(xyz), len(xyz))).tocsr()
    count, labels = connected_components(graph, directed=False)
    fg = np.bincount(labels, weights=e['fg'], minlength=count)
    bg = np.bincount(labels, weights=e['bg']+e['free_bg'], minlength=count)
    supported = np.bincount(labels, weights=e['depth_support'] >= 2, minlength=count)
    ratio = bg/np.maximum(fg+bg, 1e-12)
    isolated = (ratio[labels] > .98) & (supported[labels] == 0)
    if surface is None:
        dist = np.full(len(rows), np.inf)
    else:
        dist = surface['tree'].query(xyz, workers=8)[0]
    return dict(isolated=isolated, surface_distance=dist, spacing=spacing, components=labels)


def historical_candidates(e, original_fg, ids, iteration, spatial=None):
    fg = np.maximum(e['fg'], original_fg[ids])
    bg = e['bg']+e['free_bg']
    ratio = bg/np.maximum(fg+bg, 1e-12)
    floor = adaptive_floor(bg)
    # These are dimensionless historical recipe constants, not plant-specific
    # pixel thresholds. The absolute harm floor follows each distribution.
    limit = .99 if iteration == 0 else .95
    remove = (ratio > limit) & (bg > floor) & (e['bg_views'] >= 2)
    if spatial is not None:
        sigma = spatial['sigma']
        needle = sigma[:, 2]/np.maximum(sigma[:, 1], 1e-12)
        isolation = (spatial['surface_distance'] > 4*np.maximum(spatial['spacing'], sigma[:, 1])) | ((needle > 10) & (e['depth_support'] < 2))
        remove |= (ratio > .8) & (bg > floor) & (fg < np.quantile(fg[fg > 0], .25) if (fg > 0).any() else False) & isolation & (e['depth_support'] < 2) & (e['bg_views'] >= 2)
    return remove, dict(harm_threshold=limit, contribution_floor=floor if np.isfinite(floor) else None,
        foreground_envelope=True, removed_foreground_fraction=float(fg[remove].sum()/max(fg.sum(), 1e-12)))
