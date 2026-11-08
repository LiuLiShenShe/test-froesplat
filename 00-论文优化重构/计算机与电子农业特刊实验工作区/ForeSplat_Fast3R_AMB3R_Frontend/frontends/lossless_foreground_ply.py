"""Complete per-frame foreground exports by losslessly splitting saved PLYs.

This never runs a model, transforms coordinates or samples/averages points.
Only three saved prediction archives are read for original-value spot checks.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from frontends.common import json_write, sha256

RECORD_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


def ply_header(count):
    return ('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'end_header\n' % count).encode()


def read_layout(path):
    path = Path(path)
    with path.open('rb') as stream:
        lines = []
        for _ in range(32):
            line = stream.readline()
            lines.append(line)
            if line == b'end_header\n':
                break
        else:
            raise ValueError(f'Unsupported PLY header: {path}')
        count_lines = [line for line in lines if line.startswith(b'element vertex ')]
        if len(count_lines) != 1:
            raise ValueError(f'Missing unique vertex count: {path}')
        count = int(count_lines[0].split()[-1])
        header = b''.join(lines)
        if count < 0 or header != ply_header(count):
            raise ValueError(f'Unsupported PLY record layout: {path}')
        if path.stat().st_size != len(header) + count * RECORD_DTYPE.itemsize:
            raise ValueError(f'PLY count/file size mismatch: {path}')
    return count, len(header)


def validate_per_frame_exports(directory):
    """Validate required real files; no prediction payload reads."""
    directory = Path(directory)
    bundle = json.loads((directory / 'bundle.json').read_text())
    with (directory / 'per_frame_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if [row['name'] for row in rows] != bundle['input_names']:
        raise ValueError('Per-frame CSV order differs from original bundle order')
    totals = {'mask': 0, 'mask_confidence': 0}
    for row in rows:
        stem = Path(row['name']).stem
        for branch in totals:
            path = directory / 'pointcloud' / 'global_per_frame' / f'{stem}_{branch}.ply'
            count, _ = read_layout(path)
            if branch == 'mask_confidence' and count != int(row['valid_foreground_points']):
                raise ValueError(f'Foreground-valid point count differs: {path}')
            if branch == 'mask' and count > int(row['foreground_pixels']):
                raise ValueError(f'Mask point count exceeds foreground pixels: {path}')
            totals[branch] += count
    for branch, total in totals.items():
        complete = directory / 'pointcloud' / f'global_{branch}.ply'
        if read_layout(complete)[0] != total or bundle['cloud_counts'][branch] != total:
            raise ValueError(f'Per-frame count sum differs from complete {branch} cloud')
    return dict(frames=len(rows),required_files=2 * len(rows),point_counts=totals)


def _copy_segment(stream, target, offset, count):
    """Write identical source vertex bytes, then verify the actual saved file."""
    header = ply_header(count)
    stream.seek(offset)
    remaining = count * RECORD_DTYPE.itemsize
    temporary = target.with_name(target.name + '.lossless_split_tmp')
    digest = hashlib.sha256(header)
    try:
        with temporary.open('wb') as output:
            output.write(header)
            while remaining:
                data = stream.read(min(1024 * 1024, remaining))
                if not data:
                    raise ValueError('Unexpected EOF in complete PLY')
                output.write(data)
                digest.update(data)
                remaining -= len(data)
        expected_sha = digest.hexdigest()
        if sha256(temporary) != expected_sha:
            raise ValueError(f'Saved PLY differs from source segment: {target}')
        if target.exists():
            if sha256(target) != expected_sha:
                raise FileExistsError(f'Existing per-frame cloud differs: {target}')
            temporary.unlink()
            operation = 'existing_byte_identical'
        else:
            os.replace(temporary, target)
            operation = 'created_lossless_split'
        if read_layout(target)[0] != count:
            raise ValueError(f'Saved per-frame vertex count differs: {target}')
        return expected_sha, operation
    finally:
        if temporary.exists():
            temporary.unlink()


def complete_lossless_exports(directory):
    directory = Path(directory)
    bundle = json.loads((directory / 'bundle.json').read_text())
    csv_path = directory / 'per_frame_metrics.csv'
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    names = bundle['input_names']
    if [row['name'] for row in rows] != names:
        raise ValueError('CSV order differs from original input order')
    sources = {branch: directory / 'pointcloud' / f'global_{branch}.ply'
               for branch in ('mask', 'mask_confidence')}
    counts = {'mask': [int(row['foreground_pixels']) for row in rows],
              'mask_confidence': [int(row['valid_foreground_points']) for row in rows]}
    layouts = {branch: read_layout(path) for branch, path in sources.items()}
    for branch in sources:
        if sum(counts[branch]) != layouts[branch][0] or sum(counts[branch]) != bundle['cloud_counts'][branch]:
            raise ValueError(f'Cannot prove exact per-frame {branch} partition from saved CSV; no files changed')
    # The mask branch is a subset of all foreground pixels. Equality of total
    # counts proves every foreground pixel is finite and fixes each frame count.
    print('Checking complete foreground PLY SHA256 for byte-identical branch reuse', flush=True)
    source_hashes = {branch: sha256(path) for branch, path in sources.items()}
    equivalent = source_hashes['mask'] == source_hashes['mask_confidence'] and counts['mask'] == counts['mask_confidence']
    output = directory / 'pointcloud' / 'global_per_frame'
    output.mkdir(exist_ok=True)
    records = []
    with sources['mask'].open('rb') as mask_stream, sources['mask_confidence'].open('rb') as confidence_stream:
        offsets = {branch: layouts[branch][1] for branch in sources}
        for index, name in enumerate(names):
            stem = Path(name).stem
            mask_path = output / f'{stem}_mask.ply'
            mask_sha, operation = _copy_segment(mask_stream, mask_path, offsets['mask'], counts['mask'][index])
            row = dict(name=name,original_frame_index=index,source_index=int(rows[index]['source_index']),
                       mask_points=counts['mask'][index],mask_confidence_points=counts['mask_confidence'][index],
                       mask_sha256=mask_sha,mask_operation=operation)
            confidence_path = output / f'{stem}_mask_confidence.ply'
            if equivalent:
                if confidence_path.exists():
                    if sha256(confidence_path) != mask_sha:
                        raise FileExistsError(f'Existing confidence branch differs: {confidence_path}')
                    confidence_operation = 'existing_byte_identical'
                else:
                    os.link(mask_path, confidence_path)
                    confidence_operation = 'hardlink_of_identical_mask_branch'
                confidence_sha = mask_sha
            else:
                confidence_sha, confidence_operation = _copy_segment(
                    confidence_stream, confidence_path, offsets['mask_confidence'], counts['mask_confidence'][index])
            if read_layout(confidence_path)[0] != counts['mask_confidence'][index]:
                raise ValueError(f'Saved confidence point count differs: {confidence_path}')
            row.update(mask_confidence_sha256=confidence_sha,mask_confidence_operation=confidence_operation)
            records.append(row)
            for branch in sources:
                offsets[branch] += counts[branch][index] * RECORD_DTYPE.itemsize
            if (index + 1) % 32 == 0:
                print(f'Lossless foreground split: {index + 1}/{len(names)} complete', flush=True)
    validation = validate_per_frame_exports(directory)
    probes = []
    for index in sorted({0, len(names) // 2, len(names) - 1}):
        name = names[index];stem = Path(name).stem
        with np.load(io.BytesIO((directory / 'raw_predictions' / f'{stem}.npz').read_bytes())) as raw:
            xyz = raw['world_points'];conf = raw['world_points_conf'].reshape(xyz.shape[:2])
        rgb = np.asarray(Image.open(directory / 'rgb' / f'{stem}.png').convert('RGB'))
        mask = np.asarray(Image.open(directory / 'mask' / f'{stem}.png')) > 0
        finite = np.isfinite(xyz).all(-1)
        comparisons = {}
        for branch, selected in [('mask', finite & mask),
                                  ('mask_confidence', finite & mask & np.isfinite(conf) & (conf > 1))]:
            expected = np.empty(int(selected.sum()), dtype=RECORD_DTYPE)
            selected_points=xyz[selected];selected_colors=rgb[selected]
            for i, key in enumerate(('x', 'y', 'z')):
                expected[key] = selected_points[:, i]
            for i, key in enumerate(('red', 'green', 'blue')):
                expected[key] = selected_colors[:, i]
            path = output / f'{stem}_{branch}.ply';count, offset = read_layout(path)
            with path.open('rb') as stream:
                stream.seek(offset);actual = stream.read()
            identical = count == len(expected) and actual == expected.tobytes()
            if not identical:
                raise ValueError(f'Original raw XYZ/RGB differs from split: {path}')
            comparisons[branch] = dict(points=count,all_foreground_xyz_rgb_bytes_identical=True)
        probes.append(dict(name=name,original_frame_index=index,branches=comparisons))
    audit = dict(schema='foresplat.lossless_foreground_split.v1',operation='exact sequential binary PLY partition',
                 frame_order=names,per_frame_csv_sha256=sha256(csv_path),
                 source_plys={branch:dict(path=str(path),vertices=layouts[branch][0],sha256=source_hashes[branch])
                              for branch,path in sources.items()},
                 record_layout='little-endian float32 XYZ + uint8 RGB;15bytes/vertex',
                 complete_branches_byte_identical=equivalent,frames=records,validation=validation,
                 raw_xyz_rgb_spot_checks=probes,raw_prediction_archives_read=len(probes),
                 point_sampling=False,point_fusion=False,coordinate_transform=False,
                 source_arrays_cameras_complete_plys_modified=False,
                 interpretation='Original concat order is the bundle input order/CSV order. Source segment bytes and saved per-frame SHA256 are identical; all vertices are retained. Mask total equals foreground-pixel total, proving the mask partition counts. Only3raw frames were read for full foreground XYZ/RGB byte comparisons.')
    json_write(directory / 'lossless_split_audit.json',audit)
    print(f'Lossless exports verified: {validation}',flush=True)
    return audit
