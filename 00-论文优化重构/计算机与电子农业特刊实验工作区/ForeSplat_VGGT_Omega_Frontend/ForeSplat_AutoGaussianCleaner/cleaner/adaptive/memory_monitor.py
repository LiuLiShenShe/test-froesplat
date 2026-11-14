"""Sample this process's device memory, including native CUDA allocations."""
import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path


class ProcessGpuMemory:
    def __init__(self, output, interval=5.):
        self.path=Path(output)/'GPU_PROCESS_MEMORY.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.interval=interval;self.pid=os.getpid();self.stop_event=threading.Event()
        self.started=time.perf_counter();self.samples=0;self.peak=None;self.failures=0;self.gpu_uuids=set()
        self.thread=threading.Thread(target=self._loop,name='gggs-memory-sampler',daemon=True)

    def _sample(self):
        try:
            result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
            rows=[]
            for row in csv.reader(result.stdout.splitlines()):
                if len(row)!=3 or int(row[0].strip())!=self.pid:continue
                value=float(row[1].strip());uuid=row[2].strip();rows.append((value,uuid));self.gpu_uuids.add(uuid)
            memory=sum(value for value,_ in rows) if rows else None
            self.samples+=1
            if memory is not None:self.peak=max(self.peak or 0.,memory)
            record=dict(seconds=time.perf_counter()-self.started,pid=self.pid,memory_mib=memory,gpu_uuids=[u for _,u in rows])
        except (OSError,subprocess.SubprocessError,ValueError):
            self.failures+=1;record=dict(seconds=time.perf_counter()-self.started,pid=self.pid,status='measurement_unavailable')
        with self.path.open('a') as f:f.write(json.dumps(record)+'\n')

    def _loop(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start();return self

    def finish(self):
        self.stop_event.set();self.thread.join(timeout=5);self._sample()
        return dict(sampled_process_peak_mib=self.peak,sampling_interval_seconds=self.interval,
            samples=self.samples,unavailable_samples=self.failures,pid=self.pid,gpu_uuids=sorted(self.gpu_uuids),
            scope='NVIDIA device memory attributed to this process, including native rasterizer allocations. Sampled maximum is a lower bound on the true instantaneous peak; excludes other processes.',
            samples_file=str(self.path.resolve()))
