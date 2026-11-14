"""Independent CPU-only, rate-limited native GGGS cleanup orchestration."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import os
import resource
import time
import numpy as np

from .features import REQUIRED,WIDTH,extract_features,pack,unpack
from .io import Meter,read_header,chunks,proc_io
from .spatial import build,query,quantile


@dataclass(frozen=True)
class CleanupConfig:
    io_limit_mib: float=20.0
    memory_budget_gib: float=8.0
    cell_scale: float=4.0
    min_local_support: int=3
    alpha_floor: float=1e-5
    block_rows: int=16384
    stage: str='joint'

    def validate(self):
        values=(self.io_limit_mib,self.memory_budget_gib,self.cell_scale,self.alpha_floor)
        if not all(np.isfinite(v) for v in values): raise ValueError('Configuration must be finite')
        if self.io_limit_mib<0 or self.memory_budget_gib<=0 or self.cell_scale<=0:
            raise ValueError('Rate must be nonnegative; memory budget and cell scale must be positive')
        if self.block_rows<1 or self.min_local_support<1 or self.alpha_floor<0:
            raise ValueError('Invalid block/support/alpha configuration')
        if self.stage not in ['attributes','appearance','isolation','joint']:
            raise ValueError('Unknown ablation stage')


def decide_keep(features,config=CleanupConfig(),spatial_index=None):
    """Conservative joint decision; brightness alone never deletes a row.

    Four ablation stages share identical input features and threshold sample.
    Relative brightness is a DC prior, never a claim about rendered SH color.
    Fine structures are protected by opposing occupied cells; supported dark
    regions are retained unless size and relative color both disagree locally.
    """
    config.validate()
    n=len(features['xyz']);b=features['brightness'];s=features['effective_max'];a=features['alpha_eff']
    b05,b20=quantile(b,[.05,.20]);s50,s995=quantile(s,[.50,.995]);a01=quantile(a,.01)
    scale_limit=max(float(s995),float(s50)*3)
    alpha_limit=max(config.alpha_floor,float(a01)*.25)
    dark=b<=b05;large=s>=scale_limit;weak=a<=alpha_limit
    candidates=dark|large|weak
    ids=np.flatnonzero(candidates)
    removed=np.zeros(n,bool)
    thresholds=dict(brightness_q05=float(b05),brightness_q20=float(b20),
                    effective_scale_large=scale_limit,effective_alpha_low=alpha_limit)
    stats=dict(candidate_count=len(ids),dark_dc=int(dark.sum()),large_effective_scale=int(large.sum()),
               low_effective_alpha=int(weak.sum()),thresholds=thresholds)
    # Attribute-only and appearance stages intentionally expose why unconditional
    # opacity or dark-scale pruning is aggressive; they are research ablations.
    if config.stage=='attributes': removed=weak & large
    elif config.stage=='appearance': removed=(weak & large)|(dark & large)
    else:
        index=spatial_index if spatial_index is not None else build(features,config.cell_scale,config.block_rows)
        local=query(index,features,ids)
        support=local['support'];nb=local['neighbor_brightness'];ns=local['neighbor_scale']
        sparse=support<config.min_local_support
        local_dark=np.isfinite(nb) & (b[ids]<nb-.02) & (b[ids]<.5*np.maximum(nb,.02))
        local_large=np.isfinite(ns) & (s[ids]>3*np.maximum(ns,np.finfo('f4').tiny))
        scale=large[ids]|local_large
        bad_dark=dark[ids]|local_dark
        delete=(weak[ids] & (sparse|scale)) | (bad_dark & scale & (sparse|local_dark))
        protected=np.zeros(len(ids),bool)
        if config.stage=='joint':
            continuous=local['opposite_pairs']>=1
            # A line or sheet can have low density; coherent occupied directions
            # plus a locally reasonable scale protect dark leaf/stem structures.
            protected=(continuous & ~local_large) | ((b[ids]>=b20) & (support>=config.min_local_support))
            delete &= ~protected
        removed[ids[delete]]=True
        stats.update(voxel_count=len(index['unique']),voxel_size=index['cell_size'],
                     sparse_candidate=int(sparse.sum()),local_dark_outlier=int(local_dark.sum()),
                     local_scale_outlier=int(local_large.sum()),continuous_candidates=int((local['opposite_pairs']>=1).sum()),
                     protected_candidates=int(protected.sum()),removed_low_alpha=int((delete&weak[ids]).sum()),
                     removed_dark_large=int((delete&bad_dark&scale).sum()))
    stats['removed_count']=int(removed.sum())
    return ~removed,stats


def _write_report(path,report):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def cleanup_ply(input_path,output_path,report_path=None,config=CleanupConfig()):
    config.validate()
    source=Path(input_path).resolve();output=Path(output_path).resolve()
    report_file=Path(report_path).resolve() if report_path else None
    if source==output or report_file in [source,output]:
        raise ValueError('Input, output and report must have different paths')
    if output.exists() or (report_file and report_file.exists()):
        raise FileExistsError('Choose new output paths; existing inputs and results are preserved')
    started=time.perf_counter();cpu_start=time.process_time();initial_io=proc_io()
    meter=Meter(config.io_limit_mib);source_stat=source.stat()
    budget=config.memory_budget_gib*2**30
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.name+'.partial')
    if partial.exists(): raise FileExistsError(partial)
    try:
        with source.open('rb',buffering=0) as stream:
            header=read_header(stream,meter)
            if not REQUIRED.issubset(header.dtype.names): raise ValueError('Missing native GGGS fields')
            expected=header.offset+header.count*header.dtype.itemsize
            if expected!=source_stat.st_size: raise ValueError('PLY payload size does not match header')
            n=header.count
            # Conservative worst-case numeric working set incl. unique sorting,
            # compact features, keep and candidate arrays; chunk scratch + Python.
            compact_estimate=n*240 + config.block_rows*header.dtype.itemsize*4 + 96*2**20
            full_estimate=compact_estimate+n*header.dtype.itemsize
            if compact_estimate>budget:
                raise MemoryError(f'Compact working-set estimate {compact_estimate/2**30:.2f} GiB exceeds budget; increase --memory-budget-gib')
            full=full_estimate<=budget
            rows=np.empty(n,dtype=header.dtype) if full else None
            compact=np.empty((n,WIDTH),'f4')
            for start,block in chunks(stream,header,meter,config.block_rows):
                compact[start:start+len(block)]=pack(block)
                if full: rows[start:start+len(block)]=block
            del block
        read_done=time.perf_counter()
        features=unpack(compact)
        keep,decision=decide_keep(features,config)
        decision_done=time.perf_counter()
        with partial.open('xb',buffering=0) as dest:
            meter.write(dest,header.output(int(keep.sum())))
            if full:
                for start in range(0,n,config.block_rows):
                    block=rows[start:start+config.block_rows]
                    meter.write(dest,block[keep[start:start+len(block)]])
            else:
                with source.open('rb',buffering=0) as stream:
                    # Header is re-read and counted; no mmap, seeks over data or scratch files.
                    second=read_header(stream,meter)
                    if second.lines!=header.lines: raise ValueError('Source header changed during cleanup')
                    for start,block in chunks(stream,header,meter,config.block_rows):
                        meter.write(dest,block[keep[start:start+len(block)]])
            os.fsync(dest.fileno())
        if (source.stat().st_size,source.stat().st_mtime_ns)!=(source_stat.st_size,source_stat.st_mtime_ns):
            raise ValueError('Source changed while cleaning')
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    ended=time.perf_counter();cpu_s=time.process_time()-cpu_start;final_io=proc_io()
    total=ended-started
    report=dict(schema='fast_ply_v2',algorithm='conservative filtered attributes + sparse voxel joint evidence',
                input_ply=str(source),output_ply=str(output),config=asdict(config),
                original_gaussians=n,remaining_gaussians=int(keep.sum()),removed_gaussians=int((~keep).sum()),
                removed_fraction=float((~keep).mean()),decision=decision,
                execution_path='one_read_in_memory' if full else 'two_sequential_reads_compact_features',
                timing=dict(total_s=total,read_and_features_s=read_done-started,decision_s=decision_done-read_done,
                            export_s=ended-decision_done,process_cpu_s=cpu_s,throttle_wait_s=meter.wait_s,
                            active_wall_excluding_throttle_s=total-meter.wait_s,
                            wall_s_per_million=total/n*1e6,cpu_s_per_million=cpu_s/n*1e6),
                io=dict(complete_model_reads=1 if full else 2,complete_model_writes=1,
                        logical_read_bytes=meter.read_bytes,logical_write_bytes=meter.write_bytes,
                        read_write_bytes=meter.read_bytes+meter.write_bytes,configured_limit_mib_s=config.io_limit_mib,
                        input_bytes=source_stat.st_size,output_bytes=output.stat().st_size,
                        average_aggregate_mib_s=(meter.read_bytes+meter.write_bytes)/total/2**20,
                        observed_peak_one_second_mib_s=meter.peak_window_bytes/2**20,
                        max_transfer_bytes=meter.max_transfer_bytes,read_calls=meter.read_calls,write_calls=meter.write_calls,
                        throttle_wait_s=meter.wait_s,input_throttle_wait_s=meter.read_wait_s,output_throttle_wait_s=meter.write_wait_s,
                        proc_io_delta={k:final_io[k]-initial_io.get(k,0) for k in final_io},
                        measurement_scope='Logical application PLY bytes; proc read_bytes/write_bytes are page-cache-dependent physical counters. Kernel/device peaks are not capped or measured by this application.',
                        temporary_data_bytes=0,final_ply_bytes=output.stat().st_size,
                        staging_file_peak_bytes=output.stat().st_size,
                        staging_scope='The single final PLY uses a partial name until atomic rename; no extra data copy or scratch files'),
                resource=dict(gpu_used=False,renderer_used=False,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                              budget_bytes=int(budget),estimated_peak_bytes=int(full_estimate if full else compact_estimate)),
                native_record_policy='All fields, order, byte order and retained record bytes preserved; only vertex count/header line changed.',
                quality_boundary='DC/voxel evidence is a proxy. Multi-view rendering and semantic review are required for artifact and true plant preservation claims.')
    if report_file:
        report_file.parent.mkdir(parents=True,exist_ok=True);_write_report(report_file,report)
    return report
