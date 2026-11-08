"""Recover exact E1-E4 CSV components and saved mask pruning events, CPU only."""
import csv
import json
from pathlib import Path
import numpy as np

AUDIT=Path(__file__).resolve().parents[1];OMEGA=AUDIT.parents[2]
def main():
    names=['E1_GGGS_Baseline','E2_GGGS_EO','E3_GGGS_Foreground','E4_ForeSplat_GGGS']
    results=[]
    for name in names:
        path=OMEGA/'experiments/KongQueZhuYu'/name
        rows=list(csv.DictReader((path/'metrics/train_metrics.csv').open()))
        phases={}
        for label,lo,hi in [('before_regularization',0,6999),('regularization_and_densification',7000,14999),('after_densification',15000,30000)]:
            selected=[r for r in rows if lo<=int(r['iteration'])<=hi]
            weighted={key:np.array([float(r[key])*weight for r in selected]) for key,weight in [('rgb_loss',1),('depth_normal_loss',.05),('ncc_loss',.6),('geo_loss',.02),('mask_loss',1),('bg_loss',1)]}
            total=np.array([float(r['total_loss']) for r in selected]);recovered=sum(weighted.values())
            phases[label]={'csv_samples':len(selected),'iteration_range':[int(selected[0]['iteration']),int(selected[-1]['iteration'])],
                'weighted_component_means':{key:float(val.mean()) for key,val in weighted.items()},
                'total_loss_mean':float(total.mean()),'max_total_loss_recovery_error_csv_rounding':float(np.max(np.abs(total-recovered)))}
        prune=path/'checkpoints/pruning_report.json'
        events=json.loads(prune.read_text()) if prune.exists() else []
        results.append(dict(id=name[:2],source_csv=str(path/'metrics/train_metrics.csv'),phases=phases,
                saved_mask_pruning_events=events,total_saved_mask_pruned=sum(r['removed'] for r in events),
                regular_densification_event_counts_saved=False))
    out=dict(schema='gggs.loss_pruning_evidence.v1',experiments=results,
        csv_semantics='Raw depth_normal/NCC/geo components are weighted by .05/.6/.02 here; mask_loss/bg_loss in CSV are already weighted. These are sampled instantaneous training losses, not evaluation or EMA.',
        e5_exact_component_csv_present=False,e5_note='Only 4-decimal EMA stdout fields exist; do not create an artificial E5 component decomposition.',
        gradient_or_Q_history_present=False,no_training=True,no_gpu=True)
    p=AUDIT/'gaussian_analysis/loss_pruning_summary.json';p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for r in results:print(r['id'],r['phases']['regularization_and_densification'],r['total_saved_mask_pruned'])

if __name__=='__main__':main()
