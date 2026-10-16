"""Freeze capacities, evaluation and mesh choices before any experiment metrics."""
from common import *

if __name__=='__main__':
    target=STAGE/'configs/EXPERIMENT_PLAN.json'
    if target.exists():raise FileExistsError(target)
    config_path=E1/'configs/WanNianQing1_B_R0_TRAIN_CONFIG.json'
    config=read_json(config_path)
    source=Path(config['model']['model_path'])/'point_cloud/iteration_30000/point_cloud.ply'
    refs=[r for r in rows(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv') if r['sample']=='WanNianQing1']
    legacy=[r for r in rows(E1/'HELDOUT_PER_VIEW_METRICS.csv') if r['sample']=='WanNianQing1' and r['group']=='B_R0']
    assert len(refs)==len(legacy)==27
    names=sorted(r['loader_image_name'] for r in refs)
    source_sha='436ec90adae95941f2228789825a3f21c4300a3a605796ea8da387e574c633ef'
    assert sha(source)==source_sha
    d={'created_at':now(),'baseline_commit':BASELINE,'scene':'WanNianQing1','group':'B_R0','training_seed':0,'M2M3':'M0',
       'source_model':str(source),'source_model_sha256':source_sha,'source_native_model_dir':config['model']['model_path'],
       'source_training_config':str(config_path),'source_training_config_sha256':sha(config_path),
       'model':config['model'],'pipeline':config['pipeline'],'methods':['random','opacity'],
       'ratios':['0','0.2','0.4','0.6'],'random_seed':20261016,'enabled':True,'recovery':False,'recovery_steps':0,
       'capacity_rule':'remove floor(N * Decimal(str(ratio))); stable high-score keep; ties lower source index',
       'runs':[{'run_id':f'{method}_p{int(float(ratio)*100):02d}','method':method,'prune_ratio':ratio}
               for method in ['random','opacity'] for ratio in ['0','0.2','0.4','0.6']],
       'heldout_camera_names':names,'SH_degree':3,'background':[0,0,0],'resolution':2,'depth_ratio':0.0,
       'reference_manifest':str(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv'),
       'reference_manifest_sha256':sha(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv'),
       'ROI_manifest':str(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv'),'ROI_manifest_sha256':sha(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv'),
       'ROI_group':'B_R0','ROI_status':'frozen SAM3 R0 proxy, NOT independent manual GT',
       'metric_protocol':{'raw':'native PSNR/SSIM; LPIPS VGG v0.1 normalize=True; clamp RGB [0,1]',
                          'ROI':'untouched E2 metric_core foreground_metrics',
                          'edge_diagnostic':'ROI inner boundary, 3-pixel Chebyshev erosion; proxy only',
                          'aggregation':'arithmetic mean across the same 27 held-out views'},
       'qualitative_camera_names':[names[0],names[len(names)//2],names[-1]],
       'benchmark':{'physical_gpu':1,'uuid':GPU_UUID,'rounds':7,'warmup_sweeps':2,'order_seed':20261016,
                    'sync':'torch.cuda.synchronize before/after every measured full renderer call',
                    'samples_per_run':7*len(names),'include_pruning_or_loading':False,
                    'render_outputs':'all native outputs, same as quality evaluation',
                    'peak_memory':'isolated benchmark process; shared camera transforms plus current model and render outputs; no LPIPS'},
       'mesh':{'run_id':'opacity_p40','protocol_path':str(E3/'configs/WanNianQing1_MESH.json'),
               'protocol_sha256':sha(E3/'configs/WanNianQing1_MESH.json'),'SH_degree':0,'full_training_cameras':True},
       'large_asset_root':str(DATA),'conference_algorithms':'not substituted by heuristics; blocked methods retain explicit failed attempts'}
    write_json(target,d);write_json(STAGE/'configs/PLAN_HASH.json',{'path':str(target),'sha256':sha(target),'frozen_at':now()})
    print('PLAN_FROZEN',sha(target),flush=True)
