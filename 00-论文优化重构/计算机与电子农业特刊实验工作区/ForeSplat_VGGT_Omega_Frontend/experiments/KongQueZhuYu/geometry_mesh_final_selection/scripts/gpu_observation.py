"""Observe this process on physical GPU1 without counting other users' jobs."""
import csv,os,subprocess,threading,time

class GPUMonitor:
    def __init__(self,path):
        self.path=path;self.samples=[];self.stop=threading.Event()
    def observe(self):
        with self.path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['unix_time','physical_gpu','pid','nvidia_smi_used_mib'])
            while not self.stop.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                    values=[float(row.split(',')[1]) for row in raw.splitlines() if row.strip() and int(row.split(',')[0])==os.getpid()]
                    if values:
                        value=max(values);self.samples.append(value);writer.writerow([time.time(),1,os.getpid(),value]);f.flush()
                except (subprocess.SubprocessError,ValueError):pass
                self.stop.wait(1.)
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.worker=threading.Thread(target=self.observe,daemon=True);self.worker.start();return self
    def __exit__(self,*args):
        self.stop.set();self.worker.join(timeout=6.)
    @property
    def peak_mib(self):return max(self.samples) if self.samples else None
