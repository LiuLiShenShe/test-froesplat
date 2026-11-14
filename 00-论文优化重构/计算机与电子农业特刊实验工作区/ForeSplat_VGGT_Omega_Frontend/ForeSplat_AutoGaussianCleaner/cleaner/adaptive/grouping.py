"""Automatic spatial connected groups, with foreground risk strata and recursive splits."""
import numpy as np


def split_group(ids, xyz, weights=None):
    if len(ids) <= 1:
        return []
    p = xyz[ids]
    axis = np.argmax(np.ptp(p, axis=0))
    order = np.argsort(p[:, axis], kind='stable')
    cut = len(ids)//2
    return [ids[order[:cut]], ids[order[cut:]]]


def groups(ids, f, evidence, config):
    ids = np.asarray(ids, np.int64)
    if not len(ids):
        return []
    xyz = f['xyz']
    maximum = max(config['group_max_gaussians'], min(config.get('group_population_cap',65536),
        int(len(xyz)*config.get('group_population_fraction',0))))
    cell = max(f['extent']*config['group_relative_cell'], np.median(f['spacing'])*4, 1e-12)
    bins = np.floor((xyz[ids]-np.median(xyz, axis=0))/cell).astype(np.int64)
    # Keep zero/negligible FG and mixed-risk contributors in separate graph layers.
    risk = evidence['fg'][ids]/np.maximum(evidence['fg'][ids]+evidence['bg'][ids], 1e-15)
    strata = np.digitize(risk, [1e-6, .001, .01])
    keys = np.column_stack([strata, bins])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    lookup = {tuple(k): i for i, k in enumerate(unique)}
    parent = np.arange(len(unique))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, key in enumerate(unique):
        for axis in range(1, 4):
            other = key.copy()
            other[axis] += 1
            j = lookup.get(tuple(other))
            if j is not None:
                parent[find(j)] = find(i)
    labels = np.array([find(i) for i in inverse])
    result = []
    for label in np.unique(labels):
        stack = [ids[labels == label]]
        while stack:
            group = stack.pop()
            if len(group) > maximum:
                stack.extend(split_group(group, xyz))
            else:
                result.append(group)
    return sorted(result, key=lambda g: -float(evidence['bg'][g].sum()))
