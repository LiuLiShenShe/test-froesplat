"""CPU-only upstream filtering, followed by byte-preserving throttled GGGS export."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import types
import numpy as np
from fast_ply.io import Meter, read_header, chunks, proc_io

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(__file__).resolve().parent / 'upstream'
PINS = {'CorbeauSplat': '79bda917ddedfc1a26d02ae618c73296e4b133c5',
        '3dgsconverter': '99bec766bb074d49da89614d077fba58fdd843ae',
        'splat-slim': '91e08c8b39a2f6b062eb3208eac12dc7b1c58de7'}
FIELDS = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2']


def namespace(name, path):
    # Avoid importing the converter's unrelated codecs/UI and their dependencies.
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod


def check_source(name):
    repo = UPSTREAM / name
    sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != PINS[name]:
        raise ValueError(f'Expected pinned {name} {PINS[name]}, got {sha}')
    if subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--'], text=True):
        raise ValueError('Upstream tracked source was modified')
    return repo, sha


def filter_ids(rows, method, strength):
    n = len(rows)
    if method == 'corbeau':
        repo, sha = check_source('CorbeauSplat')
        namespace('app', repo / 'app')
        mod = importlib.import_module('app.core.ply_cleaner')
        ids_mask, stats = mod.compute_clean_mask(*(rows[k] for k in FIELDS), **mod.PRESETS[strength])
        return np.flatnonzero(ids_mask), dict(commit=sha, parameters=mod.PRESETS[strength],
            functions=['app.core.ply_cleaner.compute_clean_mask'], upstream_stats=stats)
    if method == 'converter':
        repo, sha = check_source('3dgsconverter')
        if importlib.util.find_spec('taichi') is not None:
            raise RuntimeError('Use a CPU environment without taichi: upstream auto-initializes GPU on import')
        namespace('gsconverter', repo / 'gsconverter')
        mod = importlib.import_module('gsconverter.processing.data_processor')
        data = np.empty(n, dtype=[(k, 'f4') for k in FIELDS[:4]] + [('original_id', 'i8')])
        for k in FIELDS[:4]:
            data[k] = rows[k]
        data['original_id'] = np.arange(n, dtype=np.int64)
        processor = mod.DataProcessor(data)
        processor.apply_alpha_filter(5)
        after_alpha = len(processor.data)
        result = processor.remove_flyers(intensity=8)
        return result['original_id'].copy(), dict(commit=sha, parameters=dict(min_opacity=5, sor_intensity=8),
            functions=['DataProcessor.apply_alpha_filter', 'DataProcessor.remove_flyers'],
            after_alpha=after_alpha, after_sor=len(result),
            upstream_cpu_sor_defect='CPU computes mask but never assigns self.data = vertices[mask]; executed unchanged')
    repo, sha = check_source('splat-slim')
    sys.path.insert(0, str(repo / 'src'))
    from splat_slim.stages.prune import prune_opacity
    from splat_slim.stages.outliers import remove_outliers
    fields = {k: rows[k].astype(np.float32) for k in FIELDS}
    fields['original_id'] = np.arange(n, dtype=np.int64)
    pruned = prune_opacity(fields, percentile=5)
    result = remove_outliers(pruned, scale_cap=1.0)
    return result['original_id'], dict(commit=sha,
        parameters=dict(opacity_percentile=5, spatial_low=.5, spatial_high=99.5, scale_low=1., scale_high=99., scale_cap=1.),
        functions=['splat_slim.stages.prune.prune_opacity', 'splat_slim.stages.outliers.remove_outliers'],
        after_prune=len(pruned['original_id']), after_outliers=len(result['original_id']))


def run(args):
    source, output = args.input.resolve(), args.output.resolve()
    report_path = output.parent / 'cleanup_report.json'
    if source == output or source == report_path:
        raise ValueError('Input/output/report must differ')
    if output.exists() or report_path.exists():
        raise FileExistsError('Use new output paths')
    if args.io_limit_mib <= 0 or args.cpus < 1:
        raise ValueError('Positive I/O limit and CPU count required')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpus])
    started, cpu0, io0 = time.perf_counter(), time.process_time(), proc_io()
    meter = Meter(args.io_limit_mib)
    initial = source.stat()
    with source.open('rb', buffering=0) as stream:
        header = read_header(stream, meter)
        if not set(FIELDS + ['filter_3D']).issubset(header.dtype.names):
            raise ValueError('Required native GGGS fields missing')
        if initial.st_size != header.offset + header.count * header.dtype.itemsize:
            raise ValueError('Invalid PLY payload size')
        if initial.st_size * 8 > args.memory_budget_gib * 2**30:
            raise MemoryError('Conservative 8x input-size budget exceeded; no implicit large-scene run')
        rows = np.empty(header.count, header.dtype)
        for start, block in chunks(stream, header, meter, 16384):
            rows[start:start+len(block)] = block
    read_done, decision_cpu0 = time.perf_counter(), time.process_time()
    ids, algorithm = filter_ids(rows, args.method, args.strength)
    decision_cpu = time.process_time() - decision_cpu0
    if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(rows)) or np.any(np.diff(ids) <= 0):
        raise ValueError('Upstream IDs must be unique, increasing, original row indices')
    decision_done = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    if partial.exists():
        raise FileExistsError(partial)
    try:
        with partial.open('xb', buffering=0) as dest:
            meter.write(dest, header.output(len(ids)))
            for start in range(0, len(ids), 16384):
                meter.write(dest, rows[ids[start:start+16384]])
            os.fsync(dest.fileno())
        current = source.stat()
        if (current.st_size, current.st_mtime_ns) != (initial.st_size, initial.st_mtime_ns):
            raise ValueError('Input changed during cleanup')
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    ended, cpu1, io1 = time.perf_counter(), time.process_time(), proc_io()
    report = dict(input_ply=str(source), output_ply=str(output), method=args.method,
        strength=args.strength if args.method=='corbeau' else None, algorithm=algorithm,
        original_gaussians=len(rows), remaining_gaussians=len(ids), removed_gaussians=len(rows)-len(ids),
        retained_ids_sha256=hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest(),
        timing=dict(total_s=ended-started, process_cpu_s=cpu1-cpu0, decision_cpu_s=decision_cpu,
                    read_s=read_done-started, decision_wall_s=decision_done-read_done,
                    export_s=ended-decision_done, throttle_wait_s=meter.wait_s),
        resource=dict(peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      cpu_affinity=sorted(os.sched_getaffinity(0)), gpu_used=False, renderer_used=False),
        io=dict(logical_read_bytes=meter.read_bytes, logical_write_bytes=meter.write_bytes,
                configured_limit_mib_s=args.io_limit_mib,
                observed_peak_one_second_mib_s=meter.peak_window_bytes/2**20,
                proc_io_delta={k:io1[k]-io0.get(k,0) for k in io1},
                complete_model_reads=1, complete_model_writes=1, intermediate_model_bytes=0,
                measurement_scope='Application trailing-one-second PLY throughput; kernel/device peaks not measured'),
        native_policy='Only select original records; original dtype, order, all SH, rotation, filter_3D and unknown fields unchanged')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=['corbeau','converter','slim'], required=True)
    p.add_argument('--strength', choices=['medium','strong'], default='medium')
    p.add_argument('--io-limit-mib', type=float, default=20.)
    p.add_argument('--memory-budget-gib', type=float, default=8.)
    p.add_argument('--cpus', type=int, default=4)
    run(p.parse_args())


if __name__ == '__main__':
    main()
