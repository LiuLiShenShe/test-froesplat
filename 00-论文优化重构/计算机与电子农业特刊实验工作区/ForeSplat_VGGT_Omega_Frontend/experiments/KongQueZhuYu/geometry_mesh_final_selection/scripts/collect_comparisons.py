"""Collect actual completed artifacts; no ranking is inferred from point count."""
import csv,json,math,hashlib,os
from pathlib import Path
import numpy as np
from surface_protocol import ROOT,write
def csvwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    summaries=[];structures=[];assets=[]
    for path in sorted((ROOT/'meshes').rglob('evaluation/complete.json')):
        folder=path.parent;structures.extend(csv.DictReader((folder/'structure.csv').open()))
        rows=list(csv.DictReader((folder/'all_references_per_view.csv').open()));groups={}
        for r in rows:groups.setdefault((r['method'],r['stage'],r['reference']),[]).append(r)
        for (method,stage,reference),group in groups.items():
            result=dict(method=method,stage=stage,reference=reference,views=len(group))
            for key in group[0]:
                if key in ['method','stage','reference','source_id']:continue
                values=[float(r[key]) for r in group if math.isfinite(float(r[key]))]
                result[key]=float(np.mean(values)) if values else float('nan')
            summaries.append(result)
    assert len([r for r in summaries if r['reference']=='RAP predicted mask'])>=14
    csvwrite(ROOT/'comparisons/mesh_structure_comparison.csv',structures);csvwrite(ROOT/'comparisons/mesh_reprojection_comparison.csv',summaries)
    methods=sorted(set(r['method'] for r in summaries))
    csvwrite(ROOT/'comparisons/mesh_scan_comparison.csv',[dict(method=m,status='NOT_AVAILABLE_CONFIRMED_PAIR',scan_path='',sim3='',pred_to_scan='',scan_to_pred='',chamfer='',fscore='',
        reason='No confirmed same-specimen/same-acquisition KongQueZhuYu laser scan or trustworthy correspondence was located in the project; arbitrary SfM/old TSDF PLY is not a scan ground truth') for m in methods])
    asset_paths=[]
    for base,dirs,files in os.walk(ROOT,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ['.venv-nksr','nksr_official','2dgs_corrected']]
        asset_paths.extend(Path(base)/name for name in files)
    for p in sorted(asset_paths):
        if not p.is_file() or any(x in p.parts for x in ['.venv-nksr','nksr_official','2dgs_corrected']):continue
        if p.suffix.lower() not in ['.ply','.npz','.pth','.json','.csv','.md','.png']:continue
        assets.append(dict(path=str(p),resolved_path=str(p.resolve()),bytes=p.stat().st_size,kind=p.suffix.lower(),is_symlink=p.is_symlink()))
    csvwrite(ROOT/'comparisons/LARGE_ASSET_INDEX.csv',assets)
    write(ROOT/'configs/comparison_collection.json',dict(status='complete',mesh_methods=methods,all_metrics_real_artifacts=True,single_plant_seed0=True,
        statistical_scope='208 views are correlated training observations, not 208 independent experiments; structure and reprojection are proxies, no invented independent scan precision'))

if __name__=='__main__':main()
