"""Preserve the original Gaussian PLY schema and every retained scalar value."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import numpy as np
from plyfile import PlyData, PlyElement


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_model(input_model, iteration=None):
    path = Path(input_model).expanduser().resolve(strict=True)
    if path.is_file():
        if path.suffix.lower() != '.ply':
            raise ValueError('Provide a Gaussian PLY or a native model directory')
        return path
    if iteration is not None:
        return (path / 'point_cloud' / f'iteration_{int(iteration)}' / 'point_cloud.ply').resolve(strict=True)
    if (path / 'pruned_model.ply').is_file():
        return (path / 'pruned_model.ply').resolve(strict=True)
    candidates = []
    for candidate in path.glob('point_cloud/iteration_*/point_cloud.ply'):
        match = re.fullmatch(r'iteration_(\d+)', candidate.parent.name)
        if match:
            candidates.append((int(match[1]), candidate))
    if not candidates:
        raise ValueError(f'No native Gaussian PLY found in {path}')
    return max(candidates)[1].resolve(strict=True)


@dataclass
class GaussianPLY:
    data: PlyData
    vertices: np.ndarray
    sh_degree: int

    @property
    def count(self):
        return len(self.vertices)

    @property
    def schema(self):
        return [{'name': name, 'dtype': self.vertices.dtype.fields[name][0].str}
                for name in self.vertices.dtype.names]


def _numbered(names, prefix, expected=None):
    fields = [n for n in names if n.startswith(prefix)]
    if any(not n[len(prefix):].isdigit() for n in fields):
        raise ValueError(f'Invalid numbered PLY property: {prefix}')
    fields.sort(key=lambda n: int(n[len(prefix):]))
    count = len(fields) if expected is None else expected
    if fields != [f'{prefix}{i}' for i in range(count)]:
        raise ValueError(f'Noncontiguous or unexpected PLY fields: {prefix}')
    return fields


def read_model(path):
    data = PlyData.read(str(path), mmap=False)
    # Mesh elements can contain vertex references; silently retaining them would
    # create invalid topology after row deletion. This API accepts Gaussian PLY.
    if len(data.elements) != 1 or data.elements[0].name != 'vertex':
        raise ValueError('Only a single vertex element is supported; Mesh/list indices need an explicit remapping adapter')
    vertices = data['vertex'].data
    if not len(vertices):
        raise ValueError('Empty Gaussian model')
    names = vertices.dtype.names
    required = {'x','y','z','opacity'}
    if not required.issubset(names):
        raise ValueError(f'Missing Gaussian properties: {required-set(names)}')
    _numbered(names, 'scale_', 2)
    _numbered(names, 'rot_', 4)
    _numbered(names, 'f_dc_', 3)
    rest = _numbered(names, 'f_rest_')
    coeffs, remainder = divmod(len(rest), 3)
    degree = math.isqrt(coeffs + 1) - 1
    if remainder or (degree + 1)**2 != coeffs + 1:
        raise ValueError('Invalid SH rest coefficient count')
    for name in names:
        dtype = vertices.dtype.fields[name][0]
        if dtype.subdtype or dtype.kind not in 'fiu':
            raise ValueError(f'Unsupported list or nonnumeric property: {name}')
        if not np.isfinite(vertices[name]).all():
            raise ValueError(f'Nonfinite PLY property: {name}')
    # Native 2DGS converts these to float32. Check the activated representation
    # as well as raw PLY values; finite log-scales can still overflow on exp.
    for name in ['x','y','z','opacity'] + _numbered(names,'f_dc_') + rest + _numbered(names,'rot_'):
        with np.errstate(over='ignore', invalid='ignore'):
            activated = vertices[name].astype(np.float32)
        if not np.isfinite(activated).all():
            raise ValueError(f'Property overflows native float32: {name}')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        scales = np.exp(np.column_stack([vertices['scale_0'],vertices['scale_1']]).astype(np.float32))
    if not (np.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError('Nonpositive or overflowing activated 2D scale')
    rotations = np.column_stack([vertices[f'rot_{i}'] for i in range(4)]).astype(np.float64)
    norms = np.linalg.norm(rotations, axis=1)
    if not (np.isfinite(norms).all() and (norms > 1e-12).all()):
        raise ValueError('Invalid rotation quaternion')
    return GaussianPLY(data, vertices, degree)


def write_subset(model, indices, path):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices):
        raise ValueError('Retained indices must be a nonempty integer vector')
    if (indices < 0).any() or (indices >= model.count).any() or len(np.unique(indices)) != len(indices):
        raise ValueError('Duplicate or out-of-range retained indices')
    # Fancy indexing applies the same row map to every known and extra property.
    element = PlyElement.describe(model.vertices[indices].copy(), 'vertex',
                                  comments=list(model.data['vertex'].comments))
    output = PlyData([element], text=model.data.text, byte_order=model.data.byte_order,
                     comments=list(model.data.comments), obj_info=list(model.data.obj_info))
    output.write(str(path))
    loaded = read_model(path)
    if loaded.schema != model.schema or loaded.sh_degree != model.sh_degree:
        raise ValueError('Writing changed the PLY schema or SH degree')
    if loaded.vertices.tobytes() != model.vertices[indices].tobytes():
        # Text PLY need not be byte-preserving, but scalar values must be.
        if not np.array_equal(loaded.vertices, model.vertices[indices]):
            raise ValueError('Retained values changed during serialization')
    return loaded
