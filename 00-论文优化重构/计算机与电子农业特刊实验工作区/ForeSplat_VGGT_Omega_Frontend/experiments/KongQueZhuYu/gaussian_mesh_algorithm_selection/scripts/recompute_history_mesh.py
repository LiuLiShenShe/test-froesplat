"""Recompute same208 historical projection metrics from verified native ray caches.

Add a consistent enclosed-missing-pixel proxy without another geometry render.
All legacy files and metrics remain intact.
"""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MPLBACKEND='Agg')
import sys,json,time,hashlib,argparse,csv
from pathlib import Path
import numpy as np
from scipy import ndimage
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection';DEST=OUT/'comparison/history_mesh_uniform'
sys.path.insert(0,str(OLD/'scripts'))
from evaluate_meshes import metrics,csvwrite
index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text());assert len(index)==208
parser=argparse.ArgumentParser();parser.add_argument('--include-new',action='store_true');args=parser.parse_args()
prestarted=OUT/'comparison/history_mesh_prestarted_process.json'
if prestarted.is_file():
    earlier=json.loads(prestarted.read_text()).get('pid')
    while earlier and earlier!=os.getpid() and (Path('/proc')/str(earlier)).exists():
        proc=Path('/proc')/str(earlier)
        try:
            state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
            command=(proc/'cmdline').read_bytes()
        except FileNotFoundError:break
        if state=='Z' or b'recompute_history_mesh.py' not in command:break
        print('Waiting for identical prestarted cached-Mesh metric process',earlier,flush=True)
        time.sleep(5)
scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text());start=time.time()
entries=[(folder.name,folder) for folder in sorted((OLD/'meshes').glob('M*')) if folder.is_dir()]
if args.include_new:
    entries += [('P3_native',OUT/'P3_GS2Mesh/meshes'),('P3_shared',OUT/'P3_GS2Mesh/meshes_shared_depth_range')]
    entries += [('E14_'+phase,OUT/'E14_2D_SuGaR'/directory) for phase,directory in [('cluster_before','cluster_pruning/cluster_before_mesh'),('cluster_after','cluster_pruning/cluster_after_mesh'),('initial','initial_mesh'),('refined','refinement/mesh_post')]]
for name,folder in entries:
    if not folder.is_dir():continue
    out=DEST/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():continue
    records=[]
    stage_scale=json.loads((OLD/'configs/omega_spatial_scale.json').read_text()) if name.startswith('M1_') else scale
    for stage in ['raw','post']:
        rows=[];legacy={int(r['source_id']):r for r in csv.DictReader((folder/'evaluation'/f'{stage}_per_view.csv').open())}
        for item in index:
            sid=item['source_id'];path=folder/'evaluation/views'/f'{stage}_{sid:04d}.npz';assert path.exists()
            with np.load(path) as z:
                pred=z['mesh_silhouette'];depth=z['mesh_depth'];reference=z['reference_depth'] if 'reference_depth' in z else None
                fg=z['foreground_mask'] if 'foreground_mask' in z else None
            if fg is None:
                with np.load(item['path']) as truth:fg=truth['foreground_mask']
            assert hashlib.sha256(fg.tobytes()).hexdigest()==item['mask_sha256']
            miss=fg&~pred;labels,count=ndimage.label(miss);rim=fg&~ndimage.binary_erosion(fg,iterations=3);touch=set(np.unique(labels[rim]));touch.add(0);sizes=np.bincount(labels.ravel());valid=[i for i in range(1,count+1) if i not in touch and sizes[i]>=9];holes=np.isin(labels,valid)
            row=dict(method=name,stage=stage,source_id=sid,reference='RAP predicted mask',**metrics(pred,fg,depth,reference,stage_scale['diagonal']),hole_count=len(valid),hole_area_px=int(holes.sum()),hole_area_fg_ratio=float(holes.sum()/max(fg.sum(),1)),open_edge_missing_area_px=int(miss.sum()-holes.sum()),source_native_raycast=str(path.resolve()));rows.append(row);records.append(row)
            # Historical cache stores full reference-depth arrays for fixed8 only.
            # Keep its originally calculated208 depth measurements, while all
            # silhouette/boundary/missing-area measurements above are recalculated.
            for key in ['depth_common_pixels','depth_mae','depth_p95','depth_mae_normalized']:
                if key in legacy[sid]:row[key]=float(legacy[sid][key])
            for key in ['silhouette_iou','foreground_coverage','background_geometry_fraction','boundary_fscore_2px','boundary_chamfer_px']:
                assert np.isclose(row[key],float(legacy[sid][key]),atol=1e-10,equal_nan=True),(name,stage,sid,key)
        csvwrite(out/f'{stage}_per_view.csv',rows)
        print('historical208',folder.name,stage,flush=True)
    csvwrite(out/'all_references_per_view.csv',records)
    (out/'complete.json').write_text(json.dumps(dict(status='complete',views_per_stage=208,source_native_raycast_cache=str((folder/'evaluation/views').resolve()),historical_code=str((OLD/'scripts/evaluate_meshes.py').resolve()),same_predicted_RAP_masks_E10=True,hole_definition='Connected FG missing projections>=9px not touching3px RAP exterior boundary; knownRAP holes excluded; open leaf edges separately counted; not certified3D hole area',camera_cache='Existing actual per-source raycast, generated with same historical integer-centre projection; Omega has own W2C/gauge forsameRGB IDs',seconds=time.time()-start),indent=2))
print('Historical mesh proxies complete',flush=True)
