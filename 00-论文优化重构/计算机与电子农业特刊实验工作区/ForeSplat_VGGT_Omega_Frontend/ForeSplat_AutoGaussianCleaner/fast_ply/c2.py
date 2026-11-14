"""Frozen legacy C2_raw decisions with bounded CPU working sets and PLY I/O."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import resource
import time

import numpy as np
from scipy.special import expit
from scipy.spatial import cKDTree

from ..cleaner.adaptive.legacy import rules
from .features import REQUIRED
from .io import Meter, read_header, chunks, proc_io


@dataclass(frozen=True)
class C2Config:
    io_limit_mib: float = 20.0
    memory_budget_gib: float = 8.0
    block_rows: int = 16384

    def validate(self):
        if not np.isfinite(self.io_limit_mib) or self.io_limit_mib < 0:
            raise ValueError('I/O limit must be finite and nonnegative')
        if not np.isfinite(self.memory_budget_gib) or self.memory_budget_gib <= 0:
            raise ValueError('Memory budget must be finite and positive')
        if self.block_rows < 1:
            raise ValueError('Block rows must be positive')


def allocate_features(n, dtype):
    # Match legacy arithmetic: DC stays in its original precision; XYZ, exp,
    # opacity and global reductions use float64. Never pre-average SH values.
    dc_dtype = np.result_type(dtype['f_dc_0'], dtype['f_dc_1'], dtype['f_dc_2'])
    rgb_dtype = (np.zeros((1, 3), dc_dtype) * .28209479177387814 + .5).dtype
    return dict(xyz=np.empty((n, 3), 'f8'), scale_max=np.empty(n, 'f8'),
                raw_alpha=np.empty(n, 'f8'), brightness=np.empty(n, rgb_dtype),
                color_range=np.empty(n, rgb_dtype), knn_original=np.zeros(n, bool))


def fill_features(f, start, rows):
    for name in rows.dtype.names:
        if not np.isfinite(rows[name]).all():
            raise ValueError('Nonfinite native field: ' + name)
    sl = slice(start, start + len(rows))
    f['xyz'][sl] = np.column_stack([rows[k] for k in ['x', 'y', 'z']]).astype(np.float64)
    sigma = np.exp(np.column_stack([rows[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    rgb = np.column_stack([rows[f'f_dc_{i}'] for i in range(3)]) * .28209479177387814 + .5
    f['scale_max'][sl] = sigma.max(1)
    f['raw_alpha'][sl] = expit(rows['opacity'].astype(np.float64))
    f['brightness'][sl] = rgb.mean(1)
    f['color_range'][sl] = np.ptp(rgb, axis=1)


def decide_c2(f, progress=None):
    xyz = f['xyz']; bright = f['brightness']; n = len(bright)
    f['extent'] = max(float(np.linalg.norm(np.diff(np.quantile(xyz, [.01, .99], axis=0), axis=0))), 1e-12)
    if n > 21:
        tree = cKDTree(xyz)
        candidates = np.flatnonzero(bright < .35)
        for start in range(0, len(candidates), 16384):
            ids = candidates[start:start + 16384]
            # Deliberately identical to legacy, including k=65, default argsort
            # and ties/duplicate centers. k=21 is NOT an equivalent shortcut.
            d, ni = tree.query(xyz[ids], k=min(65, n), workers=8)
            d[ni == ids[:, None]] = np.inf
            order = np.argsort(d, axis=1)
            d, ni = np.take_along_axis(d, order, 1), np.take_along_axis(ni, order, 1)
            f['knn_original'][ids] = bright[ids] < .4 * np.median(bright[ni[:, :20]], axis=1)
            if progress and start % (16384 * 32) == 0:
                progress(f'KNN {min(start + len(ids), len(candidates))}/{len(candidates)}')
        del tree
    # All thresholds, strict inequalities, masks, bit ordering and union are
    # owned by the unchanged historical function, not a second implementation.
    return rules(f, version=3, normalized=False)


def cleanup_c2(input_path, output_dir, config=C2Config(), historical_ids=None,
               expected_retained=None, progress=None):
    config.validate()
    source = Path(input_path).resolve(); out = Path(output_dir).resolve()
    output = out / 'cleaned_gaussians.ply'; report_file = out / 'cleanup_report.json'
    partial = out / 'cleaned_gaussians.ply.partial'
    if source.parent == out or source in (output, report_file):
        raise ValueError('Output must be separate from original checkpoint')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory; preserve all previous results')
    started = time.perf_counter(); cpu_start = time.process_time(); initial_io = proc_io()
    meter = Meter(config.io_limit_mib); source_stat = source.stat()
    budget = config.memory_budget_gib * 2**30
    verification = None
    try:
        with source.open('rb', buffering=0) as stream:
            header = read_header(stream, meter)
            if not REQUIRED.issubset(header.dtype.names):
                raise ValueError('Missing native GGGS fields')
            n = header.count
            if header.offset + n * header.dtype.itemsize != source_stat.st_size:
                raise ValueError('PLY payload size does not match header')
            # Includes compact features, tree, quantile/median/norm copies,
            # eight rule masks, reason bits, KNN scratch and interpreter reserve.
            compact_estimate = n * 256 + config.block_rows * header.dtype.itemsize * 4 + 160 * 2**20
            full_estimate = compact_estimate + n * header.dtype.itemsize
            if compact_estimate > budget:
                raise MemoryError(f'C2 compact estimate {compact_estimate/2**30:.2f} GiB exceeds budget')
            full = full_estimate <= budget
            rows = np.empty(n, header.dtype) if full else None
            f = allocate_features(n, header.dtype)
            if progress: progress(f'Read {n} Gaussians; '+('one-read' if full else 'two-read'))
            for start, block in chunks(stream, header, meter, config.block_rows):
                fill_features(f, start, block)
                if full: rows[start:start + len(block)] = block
            del block
        read_done = time.perf_counter()
        removed, bits, parameters = decide_c2(f, progress)
        keep = ~removed; remaining = int(keep.sum())
        if expected_retained is not None and remaining != expected_retained:
            raise ValueError(f'Expected {expected_retained} retained IDs, got {remaining}')
        if historical_ids is not None:
            old = np.load(historical_ids, allow_pickle=False)
            ids = np.flatnonzero(keep)
            same = np.array_equal(ids, old)
            verification = dict(historical_ids=str(Path(historical_ids).resolve()),
                                historical_count=len(old), new_count=len(ids), identical_ids=bool(same))
            if not same:
                raise ValueError('Retained Gaussian IDs differ from historical C2_raw')
            del ids, old
        del f, removed, bits
        decision_done = time.perf_counter()
        if progress: progress(f'Export {remaining}/{n} retained Gaussians')
        with partial.open('xb', buffering=0) as dest:
            meter.write(dest, header.output(remaining))
            if full:
                for start in range(0, n, config.block_rows):
                    block = rows[start:start + config.block_rows]
                    meter.write(dest, block[keep[start:start + len(block)]])
            else:
                with source.open('rb', buffering=0) as stream:
                    second = read_header(stream, meter)
                    if second.lines != header.lines:
                        raise ValueError('Source header changed during cleanup')
                    for start, block in chunks(stream, second, meter, config.block_rows):
                        meter.write(dest, block[keep[start:start + len(block)]])
            os.fsync(dest.fileno())
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError('Source changed during cleanup')
        # Hard-link publication refuses to overwrite even a concurrently created
        # result, then unlinks the staging name without copying any PLY bytes.
        os.link(partial, output); partial.unlink()
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    ended = time.perf_counter(); final_io = proc_io()
    total = ended - started
    report = dict(schema='fast_c2_raw_v1', algorithm='legacy.rules(version=3, normalized=False)',
        input_ply=str(source), output_ply=str(output), config=asdict(config),
        original_gaussians=n, remaining_gaussians=remaining, removed_gaussians=n-remaining,
        removed_fraction=(n-remaining)/n, parameters=parameters, historical_verification=verification,
        execution_path='one_read_in_memory' if full else 'two_sequential_reads_compact_features',
        timing=dict(process_cpu_s=time.process_time()-cpu_start, total_s=total,
                    read_and_features_s=read_done-started, decision_s=decision_done-read_done,
                    export_s=ended-decision_done, throttle_wait_s=meter.wait_s,
                    active_wall_excluding_throttle_s=total-meter.wait_s),
        io=dict(complete_model_reads=1 if full else 2, complete_model_writes=1,
                logical_read_bytes=meter.read_bytes, logical_write_bytes=meter.write_bytes,
                configured_limit_mib_s=config.io_limit_mib,
                observed_peak_one_second_mib_s=meter.peak_window_bytes/2**20,
                average_aggregate_mib_s=(meter.read_bytes+meter.write_bytes)/total/2**20,
                max_transfer_bytes=meter.max_transfer_bytes, read_calls=meter.read_calls,
                write_calls=meter.write_calls, proc_io_delta={k:final_io[k]-initial_io.get(k,0) for k in final_io},
                temporary_data_bytes=0, final_ply_bytes=output.stat().st_size,
                measurement_scope='Aggregate application PLY bytes in trailing one-second windows; NOT measured device physical peak. Single final payload staged under .partial; no intermediate models or SHA scans.'),
        resource=dict(gpu_used=False, renderer_used=False,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      budget_bytes=int(budget), estimated_peak_bytes=int(full_estimate if full else compact_estimate)),
        native_record_policy='All fields, order, byte order and retained record bytes unchanged; only vertex count changes.',
        quality_boundary='Cleanup timing excludes separate GPU render evaluation; visual/metric review required.')
    with report_file.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return report
