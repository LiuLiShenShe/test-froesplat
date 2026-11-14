"""Explicit bounded binary PLY I/O; no mmap, fromfile, hashing or scratch files."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

_TYPES = {'char':'i1','int8':'i1','uchar':'u1','uint8':'u1','short':'i2','int16':'i2',
          'ushort':'u2','uint16':'u2','int':'i4','int32':'i4','uint':'u4','uint32':'u4',
          'float':'f4','float32':'f4','double':'f8','float64':'f8'}


class Meter:
    """Shared stream limiter. No accumulated idle credit; each call <= chunk size.

    Peak is observed application bytes in a trailing one-second window. This
    cannot cap kernel readahead/writeback or other processes; /proc counters are
    reported separately. Each transfer is paced BEFORE touching the file.
    """
    def __init__(self, mib_s=20.0, chunk_bytes=256*1024):
        self.rate = mib_s * 2**20
        self.chunk_bytes = chunk_bytes
        self.read_bytes = self.write_bytes = 0
        self.wait_s = self.read_wait_s = self.write_wait_s = 0.0
        self.events = deque()
        self.window_bytes = self.peak_window_bytes = 0
        self.read_calls = self.write_calls = 0
        self.max_transfer_bytes = 0

    def pace(self, n, kind):
        if self.rate > 0:
            start = time.perf_counter()
            time.sleep(n/self.rate)
            wait = time.perf_counter()-start
            self.wait_s += wait
            if kind == 'read': self.read_wait_s += wait
            else: self.write_wait_s += wait

    def record(self, n, kind):
        if kind == 'read':
            self.read_bytes += n; self.read_calls += 1
        else:
            self.write_bytes += n; self.write_calls += 1
        now = time.perf_counter()
        self.events.append((now,n)); self.window_bytes += n
        while self.events and self.events[0][0] <= now-1:
            self.window_bytes -= self.events.popleft()[1]
        self.peak_window_bytes = max(self.peak_window_bytes, self.window_bytes)
        self.max_transfer_bytes = max(n,self.max_transfer_bytes)

    def readinto(self, stream, target):
        view = memoryview(target).cast('B')
        for start in range(0,len(view),self.chunk_bytes):
            part = view[start:start+self.chunk_bytes]
            self.pace(len(part),'read')
            got = 0
            while got < len(part):
                n = stream.readinto(part[got:])
                if not n: raise ValueError('Truncated PLY payload')
                self.record(n,'read'); got += n

    def write(self, stream, data):
        view = memoryview(data).cast('B')
        for start in range(0,len(view),self.chunk_bytes):
            part = view[start:start+self.chunk_bytes]
            self.pace(len(part),'write')
            sent = 0
            while sent < len(part):
                n = stream.write(part[sent:])
                if not n: raise OSError('Short PLY write')
                self.record(n,'write'); sent += n


@dataclass
class Header:
    lines: list[bytes]
    dtype: np.dtype
    count: int
    offset: int

    def output(self, count):
        return b''.join(f'element vertex {count}\n'.encode() if line.split()[:2] == [b'element',b'vertex'] else line for line in self.lines)


def read_header(stream, meter):
    # Header stays under 1 MiB, is counted, and is never a full-model scan.
    lines=[]; props=[]; count=None; endian=None; total=0
    while total < 2**20:
        line=stream.readline(4096)
        if not line or len(line)==4096: raise ValueError('Invalid PLY header')
        total += len(line); lines.append(line)
        tokens=line.decode('ascii').split()
        if len(lines)==1 and tokens != ['ply']: raise ValueError('Missing PLY magic')
        if tokens[:1]==['format']:
            endian={'binary_little_endian':'<','binary_big_endian':'>'}.get(tokens[1])
            if endian is None: raise ValueError('Fast I/O requires binary PLY; ASCII is unsupported')
        elif tokens[:1]==['element']:
            if tokens[1]!='vertex' or count is not None:
                raise ValueError('Native GGGS vertex-only PLY required; refusing to discard other elements')
            count=int(tokens[2])
        elif tokens[:1]==['property']:
            if len(tokens)!=3 or tokens[1] not in _TYPES: raise ValueError('Scalar PLY properties required')
            props.append((tokens[2],_TYPES[tokens[1]]))
        elif tokens==['end_header']: break
    else: raise ValueError('PLY header exceeds 1 MiB')
    meter.pace(total,'read');meter.record(total,'read')
    if endian is None or count is None or count <= 0: raise ValueError('Empty or invalid native PLY')
    dtype=np.dtype([(name,endian+kind) for name,kind in props])
    return Header(lines,dtype,count,total)


def chunks(stream, header, meter, block_rows):
    for start in range(0,header.count,block_rows):
        rows=np.empty(min(block_rows,header.count-start),dtype=header.dtype)
        meter.readinto(stream,rows)
        yield start,rows


def proc_io():
    try:
        return {key:int(value) for key,value in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    except OSError: return {}
