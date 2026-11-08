"""Serial GPU1/uv execution with recorded commands and completion gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['MPLBACKEND']='Agg'
import argparse,csv,fcntl,hashlib,json,shlex,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OMEGA=ROOT.parents[2];SCRIPTS=ROOT/'scripts';ENV=OMEGA/'.venv/bin/python';STANDARD=OMEGA.parent/'ForeSplat_Standard_VGGT_Frontend';STD=STANDARD/'experiments/KongQueZhuYu/standard_vggt_comparison'
MODELS={'Omega':'VGGT_Omega_Original_RGB','Standard':'Standard_VGGT_Original_RGB'}
EXPS={'E8':('Omega','corrected','E8_Omega_Corrected_Interface'),'E10':('Standard','corrected','E10_Standard_Corrected_Interface'),'E7':('Omega','original','E7_Omega_Original_Interface'),'E9':('Standard','original','E9_Standard_Original_Interface')}
lock=(ROOT/'configs/GPU1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(**kw):
    (ROOT/'STATUS.json').write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2))
def run(stage,script,args,repo=None):
    command=['uv','run','--no-project','--python',str(ENV),'python',str(SCRIPTS/script),*map(str,args)]
    log=ROOT/'logs'/f'{stage}.log';log.parent.mkdir(exist_ok=True)
    record=dict(stage=stage,command=command,shell_display=shlex.join(command),cwd=str(repo or OMEGA),CUDA_VISIBLE_DEVICES='1',started_unix=time.time(),log=str(log))
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    state(status='running',stage=stage,log=str(log),command=command)
    with log.open('a') as f:
        f.write('\n'+shlex.join(command)+'\n');f.flush()
        process=subprocess.Popen(command,cwd=repo or OMEGA,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        state(status='running',stage=stage,pid=process.pid,log=str(log),command=command)
        rc=process.wait()
    record.update(ended_unix=time.time(),returncode=rc)
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if rc:raise RuntimeError(f'{stage} failed with {rc}; see {log}')
    print('STAGE COMPLETE',stage,flush=True)
def main():
    # prepare runs separately once; this runner refuses racing source writes.
    assert (ROOT/'configs/prepared_sources.json').exists()
    for model in MODELS:
        run('infer_'+model,'run_frontends.py',['infer','--model',model])
        run('export_'+model,'run_frontends.py',['export','--model',model])
    run('evaluate_frontends','evaluate_frontends.py',[])
    for interface in ['original','corrected']:run('validate_'+interface,'validate_interface.py',['--interface',interface])
    for eid,(model,interface,dirname) in EXPS.items():
        out=ROOT/dirname;out.mkdir(exist_ok=True);checkpoint=out/'checkpoints';repo=OMEGA/'gggs_community' if interface=='original' else ROOT/'camera_interface/gggs_corrected'
        dataset=ROOT/MODELS[model]/('COLMAP_Corrected' if interface=='corrected' else 'COLMAP_Original')
        if not (out/'densification_events.complete.json').exists():
            training=['--source_path',dataset,'--model_path',checkpoint,'--iterations','30000','--resolution','2','--sh_degree','3','--sg_degree','0',
                '--test_iterations','10000','20000','30000','--save_iterations','7000','15000','30000','--checkpoint_iterations','7000','15000','30000',
                '--metrics_csv',out/'train_metrics.csv','--metrics_interval','100']
            config=dict(id=eid,frontend=model,pixel_interface=interface,dataset=str(dataset),iterations=30000,views=208,seed=0,physical_gpu=1,eo=False,foreground_loss=False,alpha_loss=False,mask_pruning=False,
                initialization_ply=str(dataset/'sparse/0/points3D.ply'),training_images=str(ROOT/'training_images'),mask_unchanged=True,source_commit='23b418c2f0f19b30b251acad6fe13445dae968c7',optimizer_source_sha256=hashlib.sha256((repo/'scene/gaussian_model.py').read_bytes()).hexdigest(),argv=list(map(str,training)))
            (ROOT/'configs'/f'{eid}.json').write_text(json.dumps(config,indent=2))
            run('train_'+eid,'train_observed.py',['--repo',repo,'--events',out/'densification_events.jsonl','--',*training],repo=repo)
        assert (checkpoint/'point_cloud/iteration_30000/point_cloud.ply').exists() and (checkpoint/'chkpnt30000.pth').exists()
    for eid in ['E1','E5','E7','E8','E9','E10']:
        if eid=='E1':dataset=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset';checkpoint=ROOT.parent/'E1_GGGS_Baseline/checkpoints';repo=OMEGA/'gggs_community'
        elif eid=='E5':dataset=STD/'Standard_VGGT_Dataset';checkpoint=STD/'E5_Standard_VGGT_GGGS_Baseline/checkpoints';repo=OMEGA/'gggs_community'
        else:
            model,interface,dirname=EXPS[eid];dataset=ROOT/MODELS[model]/('COLMAP_Corrected' if interface=='corrected' else 'COLMAP_Original');checkpoint=ROOT/dirname/'checkpoints';repo=OMEGA/'gggs_community' if interface=='original' else ROOT/'camera_interface/gggs_corrected'
        run('evaluate_'+eid,'render_evaluate.py',['--experiment',eid,'--dataset',dataset,'--checkpoint-root',checkpoint,'--repo',repo])
    run('analyze_gaussians','analyze_gaussians.py',[])
    run('summarize','summarize.py',[])
    state(status='awaiting_visual_review',stage='all_208_frontends_four_30K_and_common208_evaluation_complete')
try:main()
except Exception as e:
    prior=json.loads((ROOT/'STATUS.json').read_text()) if (ROOT/'STATUS.json').exists() else {}
    state(**{k:v for k,v in prior.items() if k not in ('updated_unix','status')},status='failed',error=str(e),traceback=traceback.format_exc());raise
