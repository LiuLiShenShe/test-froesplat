"""One shared raw feature pass, frozen legacy C1/C2 decisions, bounded I/O."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import resource
import time
import numpy as np
from .c2 import C2Config, allocate_features, fill_features, decide_c2
from .features import REQUIRED
from .io import Meter, read_header, chunks, proc_io
from ..cleaner.adaptive.legacy import rules


def stamp(path):
    path = Path(path).resolve(); s = path.stat()
    return dict(path=str(path), size_bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def snapshot(meter):
    return dict(read_bytes=meter.read_bytes, write_bytes=meter.write_bytes,
                wait_s=meter.wait_s, wall=time.perf_counter(), cpu=time.process_time())


def cleanup_pair(source, output_dir, config=C2Config(), historical_ids=None,
                 reuse=None, progress=print):
    config.validate(); source=Path(source).resolve(); out=Path(output_dir).resolve()
    reuse=reuse or {}; historical_ids=historical_ids or {}
    for method in ['C1_raw','C2_raw']:
        target=out/method
        if target.exists() and any(target.iterdir()): raise FileExistsError(target)
        target.mkdir(parents=True,exist_ok=True)
    if source in [out/m/'cleaned_gaussians.ply' for m in ['C1_raw','C2_raw']]:
        raise ValueError('Cannot overwrite source')
    meter=Meter(config.io_limit_mib); initial=stamp(source); begin=snapshot(meter); io0=proc_io()
    budget=config.memory_budget_gib*2**30
    rule_hash=hashlib.sha256((Path(__file__).parents[1]/'cleaner/adaptive/legacy.py').read_bytes()).hexdigest()
    with source.open('rb',buffering=0) as stream:
        header=read_header(stream,meter);n=header.count
        if not REQUIRED.issubset(header.dtype.names):raise ValueError('Missing native GGGS fields')
        if header.offset+n*header.dtype.itemsize!=initial['size_bytes']:raise ValueError('PLY payload size')
        compact=n*272+config.block_rows*header.dtype.itemsize*6+192*2**20
        full_estimate=compact+n*header.dtype.itemsize
        if compact>budget:raise MemoryError('Compact C1/C2 working set exceeds budget')
        full=full_estimate<=budget
        rows=np.empty(n,header.dtype) if full else None; f=allocate_features(n,header.dtype)
        progress(f'Read {n:,}; '+('one input pass' if full else 'two sequential input passes'))
        for start,block in chunks(stream,header,meter,config.block_rows):
            fill_features(f,start,block)
            if full:rows[start:start+len(block)]=block
        del block
    f['extent']=max(float(np.linalg.norm(np.diff(np.quantile(f['xyz'],[.01,.99],axis=0),axis=0))),1e-12)
    shared=snapshot(meter); decisions={};keeps={};info={};verification={}
    for method,version in [('C1_raw',2),('C2_raw',3)]:
        before=snapshot(meter)
        removed,bits,params=rules(f,version=2,normalized=False) if version==2 else decide_c2(f,progress)
        keeps[method]=~removed;info[method]=params
        decisions[method]=dict(cpu_s=time.process_time()-before['cpu'],wall_s=time.perf_counter()-before['wall'])
        if method in historical_ids:
            old=np.load(historical_ids[method],allow_pickle=False);ids=np.flatnonzero(keeps[method])
            if not np.array_equal(old,ids):raise ValueError('Historical retained IDs differ: '+method)
            verification[method]=dict(identical_ids=True,count=len(ids),path=str(historical_ids[method]))
            del old,ids
        del removed,bits
        progress(f'{method}: {int(keeps[method].sum()):,}/{n:,}')
    if np.any(keeps['C2_raw'] & ~keeps['C1_raw']):raise AssertionError('C2 must be subset of C1')
    del f
    exports={}; reuse_evidence={}; required=[]
    for method in ['C1_raw','C2_raw']:
        if method not in reuse:required.append(method);continue
        record=reuse[method];old_ply=Path(record['ply']);old_info=json.loads(Path(record['report']).read_text())
        if int(keeps[method].sum())!=old_info.get('remaining_gaussians',n-old_info.get('removed',0)):
            raise ValueError('Reuse count mismatch: '+method)
        old_params=old_info.get('parameters',old_info)
        for k in ['version','normalized','thresholds','reasons','removed','scene_extent']:
            if info[method][k]!=old_params[k]:raise ValueError('Reuse rule mismatch: '+method+' '+k)
        if method not in verification:
            if old_info.get('input_ply')!=str(source):raise ValueError('Reuse source path mismatch')
            if record['input_stamp']!=initial:raise ValueError('Reuse source stat mismatch')
            if record['legacy_sha256']!=rule_hash:raise ValueError('Reuse rule hash mismatch')
        old_stamp=stamp(old_ply)
        with old_ply.open('rb',buffering=0) as stream:h=read_header(stream,meter)
        if h.count!=int(keeps[method].sum()) or h.dtype!=header.dtype:raise ValueError('Reuse native schema mismatch')
        os.link(old_ply,out/method/'cleaned_gaussians.ply')
        reuse_evidence[method]=dict(ply=old_stamp,report=str(record['report']),
            validation='recomputed identical rule parameters/reason counts + current input stamp + frozen source hash, or historical exact IDs',
            publication='hard link, zero payload copy')
        exports[method]=dict(cpu_s=0.,wall_s=0.,logical_write_bytes=0,complete_model_writes=0)
    partials=[]
    try:
        if full:
            for method in required:
                before=snapshot(meter);partial=out/method/'cleaned_gaussians.ply.partial';partials.append(partial)
                with partial.open('xb',buffering=0) as dest:
                    meter.write(dest,header.output(int(keeps[method].sum())))
                    for start in range(0,n,config.block_rows):
                        block=rows[start:start+config.block_rows]
                        meter.write(dest,block[keeps[method][start:start+len(block)]])
                    os.fsync(dest.fileno())
                exports[method]=dict(cpu_s=time.process_time()-before['cpu'],wall_s=time.perf_counter()-before['wall'],
                    logical_write_bytes=meter.write_bytes-before['write_bytes'],complete_model_writes=1)
        else:
            from contextlib import ExitStack
            with ExitStack() as stack:
                streams={}
                for method in required:
                    partial=out/method/'cleaned_gaussians.ply.partial';partials.append(partial)
                    streams[method]=stack.enter_context(partial.open('xb',buffering=0))
                    meter.write(streams[method],header.output(int(keeps[method].sum())))
                    exports[method]=dict(cpu_s=0.,wall_s=0.,logical_write_bytes=len(header.output(int(keeps[method].sum()))),complete_model_writes=1)
                with source.open('rb',buffering=0) as stream:
                    second=read_header(stream,meter)
                    if second.lines!=header.lines:raise ValueError('Source header changed')
                    for start,block in chunks(stream,second,meter,config.block_rows):
                        for method,dest in streams.items():
                            before=snapshot(meter);meter.write(dest,block[keeps[method][start:start+len(block)]])
                            e=exports[method];e['cpu_s']+=time.process_time()-before['cpu'];e['wall_s']+=time.perf_counter()-before['wall'];e['logical_write_bytes']+=meter.write_bytes-before['write_bytes']
                for dest in streams.values():os.fsync(dest.fileno())
        if stamp(source)!=initial:raise ValueError('Source changed')
        for partial in partials:os.link(partial,partial.with_suffix(''));partial.unlink()
    except BaseException:
        for partial in partials:partial.unlink(missing_ok=True)
        raise
    end=snapshot(meter);io1=proc_io();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    batch=dict(schema='frozen_c1_c2_pair_v1',input=initial,config=asdict(config),legacy_sha256=rule_hash,
        original_gaussians=n,execution_path='one_read_in_memory' if full else 'two_sequential_reads_compact_features',
        timing=dict(total_s=end['wall']-begin['wall'],process_cpu_s=end['cpu']-begin['cpu'],
                    shared_read_features_wall_s=shared['wall']-begin['wall'],shared_read_features_cpu_s=shared['cpu']-begin['cpu'],
                    throttle_wait_s=meter.wait_s,decisions=decisions,exports=exports),
        io=dict(logical_read_bytes=meter.read_bytes,logical_write_bytes=meter.write_bytes,
            complete_source_model_reads=1 if full else 2,complete_model_writes=len(required),
            observed_peak_one_second_mib_s=meter.peak_window_bytes/2**20,configured_limit_mib_s=config.io_limit_mib,
            proc_io_delta={k:io1[k]-io0.get(k,0) for k in io1},
            scope='Shared limiter for all PLY read/write bytes; application one-second window, not disk-device physical peak; reuse only reads headers'),
        resource=dict(peak_process_rss_bytes=rss,budget_bytes=int(budget),estimated_peak_bytes=int(full_estimate if full else compact),gpu_used=False),
        historical_verification=verification,reuse=reuse_evidence)
    for method in ['C1_raw','C2_raw']:
        report=dict(schema='frozen_raw_method_v1',method=method,input=initial,input_ply=str(source),
            output_ply=str(out/method/'cleaned_gaussians.ply'),legacy_sha256=rule_hash,
            original_gaussians=n,remaining_gaussians=int(keeps[method].sum()),removed_gaussians=int((~keeps[method]).sum()),
            removed_fraction=float((~keeps[method]).mean()),parameters=info[method],
            timing=dict(shared_read_features_cpu_s=batch['timing']['shared_read_features_cpu_s'],
                decision_cpu_s=decisions[method]['cpu_s'],export_cpu_s=exports[method]['cpu_s'],
                process_cpu_s=batch['timing']['shared_read_features_cpu_s']+decisions[method]['cpu_s']+exports[method]['cpu_s'],
                total_s=batch['timing']['shared_read_features_wall_s']+decisions[method]['wall_s']+exports[method]['wall_s'],
                scope='Method contribution = measured shared read/features + its decision + actual export; shared costs appear in both method reports, do not sum; reused export costs zero this run'),
            resource=batch['resource'],io=dict(shared_logical_read_bytes=meter.read_bytes,**exports[method],
                observed_peak_one_second_mib_s=batch['io']['observed_peak_one_second_mib_s']),
            reused=method in reuse_evidence,reuse=reuse_evidence.get(method),historical_verification=verification.get(method),
            native_policy='Select original raw records, preserving every field, record order and endianness')
        (out/method/'cleanup_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'batch_report.json').write_text(json.dumps(batch,ensure_ascii=False,indent=2)+'\n')
    return batch
