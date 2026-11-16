"""Observe a process tree and physical GPU1 without creating a CUDA context."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil
from runtime import write


def observe(pid, output):
    peak_rss=peak_gpu=0
    samples=0
    processes={}
    started=time.time()
    root=psutil.Process(pid)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.jsonl').open('a') as stream:
        while root.is_running() and root.status()!=psutil.STATUS_ZOMBIE:
            family=[root]+root.children(recursive=True)
            rss=0
            for proc in family:
                try:
                    rss+=proc.memory_info().rss
                    cpu=proc.cpu_times();disk=proc.io_counters()
                    processes[proc.pid]=dict(cpu_s=cpu.user+cpu.system,read_bytes=disk.read_bytes,
                        write_bytes=disk.write_bytes,rchar=disk.read_chars,wchar=disk.write_chars)
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            try:
                gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
                memory,utilization=map(float,gpu)
            except (subprocess.SubprocessError,ValueError):memory=utilization=None
            peak_rss=max(peak_rss,rss)
            if memory is not None:peak_gpu=max(peak_gpu,memory)
            stream.write(json.dumps(dict(unix=time.time(),rss_bytes=rss,gpu1_mib=memory,gpu1_utilization=utilization))+'\n')
            stream.flush();samples+=1
            time.sleep(2)
    write(output,dict(physical_gpu=1,wall_s=time.time()-started,samples=samples,
          peak_process_tree_rss_gib=peak_rss/2**30,peak_gpu_device_mib=peak_gpu,
          process_cpu_s=sum(p['cpu_s'] for p in processes.values()),
          last_observed_process_io=processes,device_memory_scope='Entire physical GPU1; this experiment is its sole compute job'))


if __name__=='__main__':observe(int(sys.argv[1]),sys.argv[2])
