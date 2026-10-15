"""Read-only resource observations; no changes to scheduling or other jobs."""
from common import *
import time

def main():
    path=STAGE/'logs/RESOURCE_OBSERVATIONS.csv'
    fresh=not path.exists()
    with path.open('a',newline='',buffering=1) as f:
        w=csv.writer(f,lineterminator='\n')
        if fresh:w.writerow(['time','io_pressure','memory_pressure','MemAvailable_kB','loadavg','GPU1_observation'])
        while not (STAGE/'reports/FINAL_ACCEPTANCE.json').exists():
            mem={r.split(':')[0]:r.split(':')[1].strip() for r in Path('/proc/meminfo').read_text().splitlines()}
            try:
                gpu=subprocess.check_output(['nvidia-smi','-i',GPU_INDEX,'--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True,timeout=8).strip()
            except Exception as e:gpu=repr(e)
            w.writerow([now(),Path('/proc/pressure/io').read_text().strip().replace('\n',' | '),Path('/proc/pressure/memory').read_text().strip().replace('\n',' | '),
                        mem['MemAvailable'],Path('/proc/loadavg').read_text().strip(),gpu])
            for _ in range(12):
                if (STAGE/'reports/FINAL_ACCEPTANCE.json').exists():return
                time.sleep(5)

if __name__=='__main__':main()
