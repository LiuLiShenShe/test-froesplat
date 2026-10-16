"""Repeated synchronized renderer timing in a process with no metric networks."""
from common import *
from native_context import benchmark_cameras
from argparse import Namespace
import gc
import time
import traceback


def telemetry():
    fields='uuid,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,utilization.gpu,memory.used'
    text=subprocess.check_output(['nvidia-smi','-i','1',f'--query-gpu={fields}','--format=csv,noheader,nounits'],text=True).strip()
    return dict(zip(fields.split(','),[s.strip() for s in text.split(',')]))


def main():
    assert_gpu();p=plan();import_paths()
    import numpy as np
    import torch
    from gaussian_renderer import render
    from plugins.gaussian_compression.evaluation.native import load_native
    torch.set_num_threads(4)
    if (STAGE/'reports/BENCHMARK.json').exists():raise FileExistsError('Benchmark exists')
    result={'status':'RUNNING','started_at':now(),'gpu_uuid':GPU_UUID,'protocol':p['benchmark'],'torch_threads':4}
    frames=[];blocks=[]
    try:
        s,views,gate=benchmark_cameras(p);gc.collect();torch.cuda.empty_cache()
        bg=torch.tensor(p['background'],dtype=torch.float32,device='cuda');pipe=Namespace(**p['pipeline'])
        runs=rows(STAGE/'COMPRESSION_RUN_MANIFEST.csv')
        rng=np.random.Generator(np.random.PCG64(p['benchmark']['order_seed']))
        with torch.no_grad():
            for repeat in range(p['benchmark']['rounds']):
                order=rng.permutation(len(runs)).tolist()
                for position,idx in enumerate(order):
                    run=runs[idx];gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                    g=load_native(run['pruned_model_path'],GS,p['SH_degree'])
                    for sweep in range(p['benchmark']['warmup_sweeps']):
                        for cam in views:
                            out=render(cam,g,pipe,bg);del out
                    torch.cuda.synchronize()
                    baseline_allocated=torch.cuda.memory_allocated();baseline_reserved=torch.cuda.memory_reserved()
                    before=telemetry();torch.cuda.reset_peak_memory_stats()
                    block=[]
                    for cam in views:
                        torch.cuda.synchronize()
                        tick=time.perf_counter_ns()
                        out=render(cam,g,pipe,bg)
                        torch.cuda.synchronize()
                        elapsed=(time.perf_counter_ns()-tick)/1e9
                        assert elapsed>0
                        block.append(elapsed)
                        frames.append({'run_id':run['run_id'],'round':repeat,'order_position':position,'camera':cam.image_name,
                            'width':cam.image_width,'height':cam.image_height,'render_seconds':elapsed,'frame_FPS':1/elapsed,
                            'physical_gpu':1,'gpu_uuid':GPU_UUID,'active_SH_degree':p['SH_degree'],'background':'0,0,0'})
                        del out
                    blocks.append({'run_id':run['run_id'],'round':repeat,'order_position':position,'frames':len(block),
                        'block_FPS':len(block)/sum(block),'mean_frame_seconds':float(np.mean(block)),
                        'p05_frame_seconds':float(np.quantile(block,.05)),'p50_frame_seconds':float(np.median(block)),
                        'p95_frame_seconds':float(np.quantile(block,.95)),
                        'baseline_allocated_bytes':baseline_allocated,'baseline_reserved_bytes':baseline_reserved,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                        **{f'GPU_before_{k}':v for k,v in before.items()},**{f'GPU_after_{k}':v for k,v in telemetry().items()}})
                    print('BENCHMARK_BLOCK',repeat,run['run_id'],round(blocks[-1]['block_FPS'],3),flush=True)
                    del g;gc.collect();torch.cuda.empty_cache()
        write_csv(STAGE/'reports/FPS_FRAME_SAMPLES.csv',frames)
        write_csv(STAGE/'reports/FPS_MEMORY_BLOCKS.csv',blocks)
        result.update(status='PASS',completed_at=now(),frame_samples=len(frames),blocks=len(blocks),
                      camera_loader_gate=str(gate),loaded_camera_count=28,heldout_camera_count=len(views),
                      timings_include=['native renderer full forward call','explicit completion synchronization'],
                      timings_exclude=['offline scoring','PLY loading','quality metrics','file IO','GPU warmup'],
                      peak_memory_scope='shared GPU camera transforms, background, current model, native render intermediates and outputs',
                      note='wall time includes CPU dispatch and waiting for all CUDA work; shared machine, no exclusive GPU lock',
                      GPU_name=torch.cuda.get_device_name(),torch_version=torch.__version__)
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:write_json(STAGE/'reports/BENCHMARK.json',result)

if __name__=='__main__':main()
