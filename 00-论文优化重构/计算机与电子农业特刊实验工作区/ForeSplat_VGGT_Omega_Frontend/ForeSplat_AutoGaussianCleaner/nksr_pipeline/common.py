"""Bounded I/O and runtime adapters; historical geometry functions stay unchanged."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT.parent
HISTORY = FRONTEND / 'experiments/KongQueZhuYu/geometry_mesh_final_selection'
PHENOTYPE = FRONTEND.parent / '阶段二十F_植物表型与尺度恢复'
OUT = ROOT / 'experiments/C1_NKSR_Phenotype'
PLANTS = ['DouBanLv1', 'HongZhang', 'ChangShouHua3']
LABELS = dict(DouBanLv1='豆瓣绿1', HongZhang='红掌', ChangShouHua3='长寿花3')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HISTORY / 'scripts'))
from fast_ply.io import Meter, proc_io
from surface_protocol import PROTOCOL, normals_from_grid, unproject, unique_observations, write
from gpu_observation import GPUMonitor


class PacedFile:
    """Seekable file for NumPy/ZIP/Torch/PIL, with actual bytes sharing one meter."""
    def __init__(self, path, mode, meter):
        self.stream = Path(path).open(mode, buffering=0)
        self.meter = meter
    def read(self, n=-1):
        if n < 0:
            n = os.fstat(self.stream.fileno()).st_size - self.stream.tell()
        parts = []
        for start in range(0, n, self.meter.chunk_bytes):
            size = min(self.meter.chunk_bytes, n-start)
            self.meter.pace(size, 'read')
            data = self.stream.read(size)
            self.meter.record(len(data), 'read')
            parts.append(data)
            if len(data) < size:
                break
        return b''.join(parts)
    def readinto(self, array):
        self.meter.readinto(self.stream, array)
        return memoryview(array).nbytes
    def write(self, data):
        self.meter.write(self.stream, data)
        return memoryview(data).nbytes
    def seek(self, *args):
        return self.stream.seek(*args)
    def tell(self):
        return self.stream.tell()
    def flush(self):
        return self.stream.flush()
    def seekable(self):
        return True
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.stream.close()


def load_npy(path, meter):
    with PacedFile(path, 'rb', meter) as stream:
        return np.load(stream, allow_pickle=False)


def save_npz(path, meter, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with PacedFile(temporary, 'xb', meter) as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            for name, value in arrays.items():
                with archive.open(name + '.npy', 'w', force_zip64=True) as entry:
                    np.save(entry, np.asarray(value), allow_pickle=False)
    os.link(temporary, path)
    temporary.unlink()


def load_image(path, size, meter, mask=False):
    with PacedFile(path, 'rb', meter) as stream:
        with Image.open(stream) as image:
            result = np.asarray(image.convert('L' if mask else 'RGB').resize(size,
                Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR)).copy()
    return result > 127 if mask else result.astype(np.float32) / 255


def save_image(path, array, meter):
    with PacedFile(path, 'wb', meter) as stream:
        Image.fromarray(np.uint8(np.clip(array, 0, 1)*255)).save(stream, format='PNG')


def stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def hash_file(path, meter=None):
    digest = hashlib.sha256()
    manager = PacedFile(path, 'rb', meter) if meter else Path(path).open('rb')
    with manager as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guard_memory(limit_gib=24, minimum_available_gib=10):
    values = dict((k, int(v.split()[0])*1024) for k, v in
        (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()))
    rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    if rss > limit_gib*2**30 or values['MemAvailable'] < minimum_available_gib*2**30:
        raise MemoryError(f'Resource guard: RSS={rss/2**30:.2f} GiB, available={values["MemAvailable"]/2**30:.2f} GiB')


class Stage:
    def __init__(self, path, gpu=False):
        self.path, self.gpu, self.meter = Path(path), gpu, Meter(20)
    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.wall, self.cpu, self.io0 = time.perf_counter(), time.process_time(), proc_io()
        if self.gpu:
            import torch
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats()
            self.monitor = GPUMonitor(self.path/'gpu_memory.csv')
            self.monitor.__enter__()
        return self
    def __exit__(self, kind, value, trace):
        if self.gpu:
            import torch
            torch.cuda.synchronize()
            self.monitor.__exit__(kind, value, trace)
        delta = {k:v-self.io0.get(k, 0) for k,v in proc_io().items()}
        self.result = dict(status='complete' if kind is None else 'failed',
            wall_s=time.perf_counter()-self.wall, process_cpu_s=time.process_time()-self.cpu,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            logical_read_bytes=self.meter.read_bytes, logical_write_bytes=self.meter.write_bytes,
            observed_peak_one_second_mib_s=self.meter.peak_window_bytes/2**20,
            configured_io_limit_mib_s=20, throttle_wait_s=self.meter.wait_s, proc_io_delta=delta,
            physical_gpu=1 if self.gpu else None,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if self.gpu else None,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if self.gpu else None,
            gpu_peak_process_mib=self.monitor.peak_mib if self.gpu else None,
            measurement_scope='This stage process only; logical large-artifact I/O plus paced images; /proc device counters separate')
        if value is not None:
            self.result['error'] = str(value)
        write(self.path/'resources.json', self.result)


def make_scene(spec, max_side=1920):
    from cleaner.runtime import Scene
    copy = dict(spec)
    # Depth is rendered by GGGS; saved external geometry is used only for scale
    # and pose checks, never loaded into a scene-wide image cache.
    copy.pop('geometry_cameras', None)
    copy.pop('geometry_dir', None)
    return Scene(copy, dict(evaluation_max_side=max_side, evidence_views=4))


def get_camera(scene, index):
    from cleaner.runtime import camera
    view = scene.views[index]
    return camera(view['K'], view['E'], view['width'], view['height'], view['name'], view['uid'])


def camera_record(camera):
    result = dict(name=camera.image_name, width=camera.image_width, height=camera.image_height,
        Fx=camera.Fx, Fy=camera.Fy, Cx=camera.Cx, Cy=camera.Cy)
    for key in ['world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center']:
        result[key] = getattr(camera, key).detach().cpu().tolist()
    return result


def load_gaussians(scene, path, meter):
    import torch
    from fast_ply import evaluate
    evaluate.torch = torch
    header, rows = evaluate.read_rows(path, meter)
    model = evaluate.model_from_rows(scene, header, rows)
    return model, header.count


def observation_keys(data, scale):
    # Exact key construction used by historical unique_observations().
    grid = np.floor((data['xyz']-np.asarray(scale['origin']))/scale['sampling_voxel']).astype(np.int32)
    dominant = np.argmax(np.abs(data['normal']), axis=1)
    sign = data['normal'][np.arange(len(grid)), dominant] >= 0
    keys = np.column_stack([grid, (dominant*2+sign).astype(np.int32)]).copy()
    return keys.view(np.dtype((np.void, 16))).ravel()


class StreamingObservations:
    """Exact first-observation selection without storing duplicate full views."""
    def __init__(self, scale):
        self.scale, self.parts = scale, []
        self.keys = np.empty(0, 'V16')
        self.order = np.empty(0, np.int64)
        self.count = 0
    def add(self, data):
        part = unique_observations(data, self.scale)
        keys = observation_keys(part, self.scale)
        if not len(keys):
            return 0
        if len(self.keys):
            positions = np.searchsorted(self.keys, keys)
            duplicate = (positions < len(self.keys)) & (self.keys[np.minimum(positions, len(self.keys)-1)] == keys)
            keys = keys[~duplicate]
            part = {k:v[~duplicate] for k,v in part.items()}
        n = len(keys)
        if not n:
            return 0
        positions = np.searchsorted(self.keys, keys) + np.arange(n)
        old_slots = np.ones(self.count+n, bool)
        old_slots[positions] = False
        merged = np.empty(self.count+n, 'V16')
        order = np.empty(self.count+n, np.int64)
        merged[positions], merged[old_slots] = keys, self.keys
        order[positions], order[old_slots] = np.arange(self.count, self.count+n), self.order
        self.keys, self.order = merged, order
        self.parts.append(part)
        self.count += n
        return n
    def finish(self):
        result = {}
        for key in ['xyz', 'normal', 'sensor', 'rgb']:
            values = np.concatenate([part[key] for part in self.parts])
            result[key] = values[self.order]
            for part in self.parts:
                del part[key]
        self.parts.clear()
        self.keys = np.empty(0, 'V16')
        self.order = np.empty(0, np.int64)
        return result


def write_mesh(path, mesh, meter):
    """Open3D-equivalent binary triangle PLY, written through the shared meter."""
    vertices, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    colours, faces = np.asarray(mesh.vertex_colors), np.asarray(mesh.triangles)
    if len(normals) != len(vertices) or len(colours) != len(vertices):
        raise ValueError('Expected native mesh with vertex normals and colours')
    header = 'ply\nformat binary_little_endian 1.0\ncomment C1 NKSR native geometry\nelement vertex '+str(len(vertices))+'\n'
    header += ''.join('property double '+key+'\n' for key in ['x','y','z','nx','ny','nz'])
    header += ''.join('property uchar '+key+'\n' for key in ['red','green','blue'])
    header += f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n'
    dtype = np.dtype([(k,'<f8') for k in ['x','y','z','nx','ny','nz']]+[(k,'u1') for k in ['red','green','blue']])
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('xb', buffering=0) as stream:
        meter.write(stream, header.encode())
        for start in range(0, len(vertices), 16384):
            stop = min(start+16384, len(vertices))
            block = np.empty(stop-start, dtype)
            for axis,key in enumerate(['x','y','z']): block[key] = vertices[start:stop,axis]
            for axis,key in enumerate(['nx','ny','nz']): block[key] = normals[start:stop,axis]
            rgb = np.rint(np.clip(colours[start:stop], 0, 1)*255).astype(np.uint8)
            for axis,key in enumerate(['red','green','blue']): block[key] = rgb[:,axis]
            meter.write(stream, block)
        for start in range(0, len(faces), 16384):
            block = np.empty(min(16384, len(faces)-start), [('n','u1'),('idx','<i4',(3,))])
            block['n'] = 3
            block['idx'] = faces[start:start+len(block)]
            meter.write(stream, block)
    os.link(temporary, path)
    temporary.unlink()
    return True


def read_mesh(path, meter):
    """Read this adapter's fixed triangle layout without Python face objects."""
    import open3d as o3d
    with Path(path).open('rb', buffering=0) as stream:
        lines = []
        while True:
            line = stream.readline(4096)
            if not line or len(lines) > 100:
                raise ValueError('Invalid mesh header')
            lines.append(line)
            if line.strip() == b'end_header': break
        size = sum(map(len, lines))
        meter.pace(size, 'read'); meter.record(size, 'read')
        text = b''.join(lines).decode('ascii')
        nv = int(next(x for x in text.splitlines() if x.startswith('element vertex')).split()[-1])
        nf = int(next(x for x in text.splitlines() if x.startswith('element face')).split()[-1])
        dtype = np.dtype([(k,'<f8') for k in ['x','y','z','nx','ny','nz']]+[(k,'u1') for k in ['red','green','blue']])
        vertices = np.empty(nv, dtype)
        faces = np.empty(nf, [('n','u1'),('idx','<i4',(3,))])
        meter.readinto(stream, vertices); meter.readinto(stream, faces)
        assert (faces['n'] == 3).all()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.column_stack([vertices[k] for k in ['x','y','z']])),
        o3d.utility.Vector3iVector(faces['idx'].copy()))
    mesh.vertex_normals = o3d.utility.Vector3dVector(np.column_stack([vertices[k] for k in ['nx','ny','nz']]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.column_stack([vertices[k] for k in ['red','green','blue']])/255.)
    return mesh


def stage_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--input-ply', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()
