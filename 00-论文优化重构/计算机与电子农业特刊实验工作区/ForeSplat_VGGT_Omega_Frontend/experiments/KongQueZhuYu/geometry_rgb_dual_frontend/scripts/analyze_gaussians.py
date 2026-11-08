"""Saved-parameter geometry evidence using the existing audit implementation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OMEGA=ROOT.parents[2];AUDIT=ROOT.parent/'root_cause_audit'
spec=importlib.util.spec_from_file_location('existing_gaussian_properties',AUDIT/'scripts/optimization_gaussian_properties.py')
audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)
audit.AUDIT=ROOT
out=ROOT/'gaussian_analysis';out.mkdir(exist_ok=True)
names={'E7':'E7_Omega_Original_Interface','E8':'E8_Omega_Corrected_Interface','E9':'E9_Standard_Original_Interface','E10':'E10_Standard_Corrected_Interface'}
results=[];rows=[]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
for eid in ['E1','E5','E7','E8','E9','E10']:
    if eid=='E1':checkpoint=ROOT.parent/'E1_GGGS_Baseline/checkpoints'
    elif eid=='E5':checkpoint=OMEGA.parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison/E5_Standard_VGGT_GGGS_Baseline/checkpoints'
    else:checkpoint=ROOT/names[eid]/'checkpoints'
    ply=checkpoint/'final_gaussians/point_cloud.ply'
    assert (checkpoint/'point_cloud/iteration_30000/point_cloud.ply').exists()
    canonical=checkpoint/'point_cloud/iteration_30000/point_cloud.ply'
    final_sha=sha(ply);canonical_sha=sha(canonical)
    assert final_sha==canonical_sha, 'Final export differs from the actual rendered30K PLY: '+eid
    records=json.loads((ROOT/'render_comparison'/eid/'camera_runtime.json').read_text())['records']
    centers=np.array([r['camera_center'] for r in records]);extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
    result=audit.analyze(dict(id=eid,ply_checkpoints=[dict(path=str(ply))],camera_summary=dict(recomputed_camera_radius_float64=extent)),200000)
    result.update(actual_rendered30000_ply_sha256=canonical_sha,final_export_exact_identity=True)
    results.append(result)
    count=result['source']['vertex_count']
    rows.append(dict(experiment=eid,gaussian_count=count,max_sigma_per_extent_p50=result['max_sigma_over_extent']['quantiles']['0.5'],
        max_sigma_per_extent_p90=result['max_sigma_over_extent']['quantiles']['0.9'],raw_alpha_p50=result['raw_alpha']['quantiles']['0.5'],filtered_alpha_p50=result['filtered_alpha']['quantiles']['0.5'],
        filtered_alpha_lt005_fraction=result['filtered_alpha_lt005']/count,raw_ge005_filtered_lt005_fraction=result['raw_ge005_filtered_lt005']/count,
        near_black_large_raw_alpha_fraction=result['near_black_large_scale_raw_alpha_ge05']/count,nonfinite_sigma_count=result['max_sigma']['nonfinite_count']))
    (out/'final_gaussian_properties.json').write_text(json.dumps(dict(experiments=results,implementation=str(AUDIT/'scripts/optimization_gaussian_properties.py'),
        actual_30000_checkpoints=True,no_training_or_pruning=True,interpretation='Scalar opacity/scale/color evidence; floaters and double surfaces require actual rendering/point evidence, not scalar labels'),indent=2,allow_nan=False))
with (ROOT/'gaussian_geometry_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
