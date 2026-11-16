"""Shared experiment paths and bounded application I/O (no algorithm changes)."""
import io
import json
import os
import time
from collections import deque
from pathlib import Path

ROOT = Path('/data/fj/F2DMAS')
OUT = (ROOT / 'experiments/COLMAP_Gaussian_Mesh_Selection').resolve()
FRONT = OUT.parents[2]
CLEANER = OUT.parents[1]
HISTORY = FRONT / 'experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection'
DATA = OUT / 'common_input/dataset'
SOURCES = {
    '3DGS': HISTORY / 'P3_GS2Mesh/official/third_party/gaussian-splatting',
    '2DGS': OUT / 'sources/2dgs_official',
    'GGGS': FRONT / 'gggs_community',
    'gsplat': OUT / 'sources/gsplat_official',
    'MCMC_3DGS': OUT / 'sources/mcmc_official',
}
PYTHONS = {
    '3DGS': OUT / '.venv-3dgs/bin/python',
    '2DGS': ROOT / '2d-gaussian-splatting-main/venv/bin/python',
    'GGGS': FRONT / '.venv/bin/python',
    'gsplat': OUT / '.venv-gsplat/bin/python',
    'MCMC_3DGS': OUT / '.venv-mcmc/bin/python',
}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


class Meter:
    def __init__(self):
        self.window = deque()
        self.read_bytes = self.write_bytes = self.peak = 0
        self.limit = int(19.5 * 1024**2)

    def record(self, n, mode):
        while True:
            now = time.monotonic()
            while self.window and now - self.window[0][0] >= 1:
                self.window.popleft()
            total = sum(b for _, b in self.window)
            if total + n <= self.limit:
                break
            time.sleep(max(.001, 1 - (now - self.window[0][0])))
        self.window.append((time.monotonic(), n))
        self.peak = max(self.peak, total + n)
        if mode == 'read': self.read_bytes += n
        else: self.write_bytes += n

    def read(self, path):
        result = bytearray()
        with open(path, 'rb') as f:
            while True:
                block = f.read(256 * 1024)
                if not block: break
                self.record(len(block), 'read')
                result.extend(block)
        return bytes(result)

    def save(self, path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            view = memoryview(data)
            for offset in range(0, len(view), 256 * 1024):
                block = view[offset:offset+256*1024]
                self.record(len(block), 'write')
                f.write(block)

    def write(self, stream, data):
        view=memoryview(data).cast('B')
        for offset in range(0,len(view),256*1024):
            block=view[offset:offset+256*1024]
            self.record(len(block),'write');stream.write(block)

    def readinto(self, stream, array):
        view=memoryview(array).cast('B')
        for offset in range(0,len(view),256*1024):
            size=min(256*1024,len(view)-offset)
            self.record(size,'read')
            count=stream.readinto(view[offset:offset+size])
            if count!=size:raise EOFError('Incomplete artifact')

    def pace(self, n, mode):
        # Compatibility with existing header readers: record performs the wait.
        pass

    def stats(self):
        return dict(application_read_bytes=self.read_bytes, application_write_bytes=self.write_bytes,
                    peak_one_second_mib_s=self.peak/1024**2, target_mib_s=20)


def install_io(meter):
    import torch
    from PIL import Image
    from plyfile import PlyData
    original_save, original_image, original_ply = torch.save, Image.open, PlyData.write
    original_read=PlyData.read

    def save(obj, file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            buffer = io.BytesIO()
            original_save(obj, buffer, *args, **kwargs)
            meter.save(file, buffer.getbuffer())
        else: original_save(obj, file, *args, **kwargs)

    def image(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and str(DATA) in str(Path(file).resolve()):
            file = io.BytesIO(meter.read(file))
        return original_image(file, *args, **kwargs)

    def ply(self, file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            buffer = io.BytesIO()
            original_ply(self, buffer, *args, **kwargs)
            meter.save(file, buffer.getbuffer())
        else: original_ply(self, file, *args, **kwargs)

    def read_ply(file,*args,**kwargs):
        if isinstance(file,(str,Path)):
            file=io.BytesIO(meter.read(file));kwargs['mmap']=False
        return original_read(file,*args,**kwargs)

    torch.save, Image.open, PlyData.write = save, image, ply
    PlyData.read=staticmethod(read_ply)
