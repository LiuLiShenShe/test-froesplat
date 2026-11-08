"""Saved actual nearest-view graph and source-proven without-replacement access."""
import json
import re
from pathlib import Path
import numpy as np

AUDIT=Path(__file__).resolve().parents[1]
def main():
    data=json.loads((AUDIT/'gaussian_analysis/experiment_provenance.json').read_text())
    rows=[]
    for exp in data['experiments']:
        if exp['id']=='E6':continue
        path=Path(exp['path'])/'checkpoints/multi_view.json'
        graph=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sizes=[len(row['nearest_name']) for row in graph]
        gaps=[abs(int(re.search(r'(\d+)$',row['ref_name'])[1])-int(re.search(r'(\d+)$',other)[1])) for row in graph for other in row['nearest_name']]
        rows.append(dict(id=exp['id'],source=str(path),views=len(graph),empty_neighbor_views=sum(n==0 for n in sizes),
            empty_neighbor_fraction=float(np.mean(np.array(sizes)==0)),neighbors_count_distribution={str(n):sizes.count(n) for n in sorted(set(sizes))},
            source_index_gap_median=float(np.median(gaps)) if gaps else None,source_index_gap_p90=float(np.quantile(gaps,.9)) if gaps else None))
    output=dict(schema='gggs.multiview_sampling.v1',experiments=rows,
        primary_training_camera_selection=dict(with_replacement=False,source_lines='train_foresplat.py:188-192',
            mechanism='When empty, stack=getTrainCameras().copy(); pop randint from remaining stack. Every complete208 cycle visits every camera exactly once.',
            iterations=30000,views=208,complete_cycles=144,partial_last_cycle=48,
            per_camera_primary_visits='Each of208 cameras visited144 or145 times; exactly48 cameras145. Identities/order not saved, cannot recover last48 from10-step stdout/100-step CSV.',
            primary_view_order_saved=False),
        neighbors_selection=dict(max_num=8,max_angle_deg=30,min_distance_native_units=.01,max_distance_native_units=1.5,
            reference='scene/__init__.py:103-114',distance_thresholds_extent_scaled=False,
            note='Graph is saved actual training selection. Distance bounds are native-world absolute units, while position LR and clone/split size threshold scale with extent. This mechanism can depend on gauge; graph counts alone do not prove image overlap or causal blur.'),
        training_executed=False,no_gpu=True)
    path=AUDIT/'gaussian_analysis/multiview_sampling.json';path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for row in rows:print(row)

if __name__=='__main__':main()
