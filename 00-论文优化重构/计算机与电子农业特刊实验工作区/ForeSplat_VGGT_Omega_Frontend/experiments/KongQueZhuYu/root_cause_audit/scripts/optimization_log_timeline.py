"""Recover historical point-count timelines without inventing missing CSV loss."""
import csv
import json
import re
from pathlib import Path

AUDIT=Path(__file__).resolve().parents[1]
OMEGA=AUDIT.parents[2]
STD=OMEGA.parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison'

def main():
    rows=list(csv.DictReader((OMEGA/'experiments/KongQueZhuYu/E1_GGGS_Baseline/metrics/train_metrics.csv').open()))
    e1={int(r['iteration']):r for r in rows}
    path=STD/'E5_Standard_VGGT_GGGS_Baseline/train_30000.log'
    text=path.read_text(errors='replace')
    records=[]
    for line in text.splitlines():
        match=re.search(r'(\d+)/30000.*?pts=(\d+)',line)
        if match:
            loss=re.search(r'Loss=([0-9.]+)',line); ncc=re.search(r'ncc=([0-9.]+)',line)
            records.append(dict(display_iteration=int(match[1]),num_gaussians=int(match[2]),ema_loss=float(loss[1]) if loss else None,ema_ncc=float(ncc[1]) if ncc else None))
    by={}
    for r in records:by.setdefault(r['display_iteration'],[]).append(r)
    matched=[]
    for t in [100,500,600,700,1000,3000,5000,6000,7000,7100,8000,10000,12000,14000,15000,30000]:
        candidates=by.get(t,[])
        counts=[r['num_gaussians'] for r in candidates]
        matched.append(dict(iteration=t,e1_csv_pre_update_count=int(e1[t]['num_gaussians']),
                            e5_tqdm_first_count=counts[0] if counts else None,e5_tqdm_last_count=counts[-1] if counts else None,
                            e5_tqdm_distinct_counts=list(dict.fromkeys(counts)),
                            e5_ema_loss=candidates[0]['ema_loss'] if candidates else None,e5_ema_ncc=candidates[0]['ema_ncc'] if candidates else None))
    output=dict(source=str(path),recovered_progress_records=len(records),matched=matched,
                e5_global_min_logged_count=min(r['num_gaussians'] for r in records),e5_global_max_logged_count=max(r['num_gaussians'] for r in records),
                loss_precision='E5 tqdm Loss/normal/ncc are 4-decimal moving averages, not exact unweighted component losses',
                count_timing='tqdm postfix is set before progress.update; duplicate display labels can bracket that iteration densification. Preserve both first/last rather than claiming a unique precise instant.',
                exact_saved_checkpoint_7000_count=132104,exact_final_count=5153165,
                event_history_limitation='No per-event clone/split/prune/Q values or gradient accumulators were saved; total-count deltas do not isolate their individual contributions.')
    p=AUDIT/'gaussian_analysis/optimization_count_timeline.json';p.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    for row in matched:print(row)

if __name__=='__main__':main()
