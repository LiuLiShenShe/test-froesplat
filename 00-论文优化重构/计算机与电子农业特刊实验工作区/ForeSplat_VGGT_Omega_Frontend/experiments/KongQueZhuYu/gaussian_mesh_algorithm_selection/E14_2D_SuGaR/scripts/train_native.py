"""Observe full official 2D-SuGaR 2DGS training; preserve the clustering intervention exactly."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,time,json,csv,inspect,importlib.util,hashlib
from pathlib import Path
from argparse import ArgumentParser,Namespace
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1]
OFF=ROOT/'official/gaussian_splatting_2d'
DATA=ROOT/'input_colmap'
OUT=ROOT/'training'
sys.path.insert(0,str(OFF))
from arguments import ModelParams,PipelineParams,OptimizationParams
from utils.general_utils import safe_state
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def camera_record(c):
    d=dict(name=c.image_name,width=c.image_width,height=c.image_height,Fx=c.Fx,Fy=c.Fy,Cx=c.Cx,Cy=c.Cy)
    for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:d[k]=getattr(c,k).detach().cpu().tolist()
    return d
def validate(cameras):
    dual=ROOT.parent.parent/'geometry_rgb_dual_frontend'
    refs={r['name']:r for r in json.loads((dual/'render_comparison/E10/camera_runtime.json').read_text())['records']}
    hashes={int(r['source_id']):r['gt_sha256'] for r in csv.DictReader((dual/'render_comparison/E10/per_view_metrics.csv').open())}
    errors=[];records=[]
    for c in cameras:
        r=refs[c.image_name]
        for k in ['Fx','Fy','Cx','Cy']:assert abs(getattr(c,k)-r[k])<1e-8,(c.image_name,k)
        for k in ['world_view_transform','projection_matrix','full_proj_transform']:
            errors.append(float(np.max(np.abs(np.asarray(getattr(c,k).cpu())-r[k]))))
        rgb=c.original_image.cpu().numpy().transpose(1,2,0)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==hashes[int(c.image_name.split('_')[-1])]
        records.append(camera_record(c))
    assert len(records)==208 and max(errors)<1e-7
    write(ROOT/'runtime_camera_validation.json',dict(status='passed',all208_K_W2C_P_RGB_identical_to_E10_E11=True,matrix_max_abs_error=max(errors),records=records))
def main():
    OUT.mkdir(exist_ok=True)
    if (ROOT/'training_complete.json').exists():return
    assert (ROOT/'priors_complete.json').exists()
    assert not (OUT/'point_cloud/iteration_30000/point_cloud.ply').exists()
    spec=importlib.util.spec_from_file_location('native_sugar_training',OFF/'train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parser=ArgumentParser();mp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    argv=['-s',str(DATA),'-m',str(OUT),'-r','2','--iterations','30000','--depth_ratio','1.0','--lambda_dist','1000','--initialization_prior','monocular_depth_normal']
    args=parser.parse_args(argv)
    dataset,opt,pipe=mp.extract(args),op.extract(args),pp.extract(args)
    # ParamGroup list parser has no item type; preserve actual integer native prune schedule.
    opt.cluster_prune_iterations=[7000]
    module.args=args
    safe_state(False);started=time.time();torch.cuda.reset_peak_memory_stats()
    write(ROOT/'actual_training_parameters.json',dict(argv=argv,dataset=vars(dataset),opt=vars(opt),pipe=vars(pipe),seed=0,physical_gpu=1,extra_prior='Metric3Dv2 actual corrected intrinsics, native per-image median track depth scale',official_loss='0.8 L1 +0.2 DSSIM + native depth-distortion1000 after3000 +normal.05 after7000 +normal prior.1 fromstart',normal_prior_domain='Official mean over the complete resized world-normal map; no RAP/alpha multiplier in the native normal-prior loss. Prior RGB is the exact masked E11 training image. RAP separately restricts extra initialization and Mesh fusion; no added foreground-only photometric/mask loss.'))
    original_scene=module.Scene
    holder={}
    def scene_hook(*a,**kw):
        scene=original_scene(*a,**kw);validate(scene.getTrainCameras());holder['scene']=scene
        count=len(scene.gaussians.get_xyz)
        for attr in ['get_xyz','get_scaling','get_rotation']:
            assert torch.isfinite(getattr(scene.gaussians,attr)).all(),('Nonfinite native initialization',attr)
        write(ROOT/'initialization_complete.json',dict(status='complete',actual_initialized_gaussians=count,baseline_VGGT_points=1252383,actual_extra_depth_generated_points=count-1252383,initial_pointcloud=str(OUT/'input.ply'),depth_scale_archive=str(DATA/'prior_depth_scale.json'),prior_normal_dtype='float32',gaussian_initialization='Native neighbour-distance two scales and normal-to-quaternion rotation; same COLMAP points plus native sampling restricted RAP',CPU_sample_expectation=json.loads((ROOT/'initialization_cpu_audit.json').read_text())['expected_initial_points_before_any_depth_finiteness_failure']))
        return scene
    module.Scene=scene_hook
    metrics=(ROOT/'training_metrics.csv').open('w',newline='')
    writer=csv.writer(metrics);writer.writerow(['iteration','photo_loss','dist_loss','normal_loss','normal_prior_loss','total_loss','gaussians','elapsed_seconds'])
    state={'iteration':0,'gaussians':0}
    report=module.training_report
    def report_hook(*a,**kw):
        iteration=a[1];state['iteration']=iteration
        loc=inspect.currentframe().f_back.f_locals
        assert torch.isfinite(loc['total_loss']),('Nonfinite native total loss',iteration)
        state['gaussians']=len(loc['gaussians'].get_xyz)
        if iteration%100==0 or iteration==1:
            writer.writerow([iteration,float(loc['loss']),float(loc['dist_loss']),float(loc['normal_loss']),float(loc['normal_prior_loss']),float(loc['total_loss']),state['gaussians'],time.time()-started]);metrics.flush()
            write(ROOT/'STATUS.json',dict(status='running',stage='native_2DGS30K',**state,elapsed_seconds=time.time()-started))
        return report(*a,**kw)
    module.training_report=report_hook
    original_prune=module.GaussianModel.prune_points
    def prune_hook(gaussians,mask):
        if state['iteration']==7000:
            folder=ROOT/'cluster_pruning';folder.mkdir(exist_ok=True)
            removed=np.asarray(mask.detach().cpu() if torch.is_tensor(mask) else mask,dtype=bool)
            pre=OUT/'point_cloud/iteration_7000/point_cloud.ply'
            link=folder/'before.ply'
            if not link.exists():link.symlink_to(pre)
            # Upstream saves PLY immediately before save_clusters/prune_points;
            # this captures the exact pre-prune optimization state at that phase.
            torch.save((gaussians.capture(),7000),folder/'before_pruning_checkpoint.pth')
            gaussians.save_ply(str(folder/'removed.ply'),removed)
            before=len(gaussians.get_xyz)
            original_prune(gaussians,mask)
            gaussians.save_ply(str(folder/'after.ply'))
            np.savez_compressed(folder/'removed_indices.npz',removed=np.flatnonzero(removed))
            torch.save((gaussians.capture(),7000),folder/'after_pruning_checkpoint.pth')
            write(folder/'complete.json',dict(status='complete',iteration=7000,before=before,removed=int(removed.sum()),after=len(gaussians.get_xyz),native_rule='DBSCAN min_samples4 epsilon98th percentile 4th nearest-neighbour distance; keep largest cluster',before_path=str(pre),after_path=str(folder/'after.ply'),removed_path=str(folder/'removed.ply'),before_checkpoint=str(folder/'before_pruning_checkpoint.pth'),after_checkpoint=str(folder/'after_pruning_checkpoint.pth'),phase='Native Scene.save -> save_clusters -> prune_points; densification skipped at7K; optimizer.step occurs afterwards'))
            return
        return original_prune(gaussians,mask)
    module.GaussianModel.prune_points=prune_hook
    module.network_gui.try_connect=lambda *a,**kw:None
    module.training(dataset,opt,pipe,[],[1,7000,15000,30000],[7000,15000,30000],None)
    torch.cuda.synchronize();metrics.close()
    assert state['iteration']==30000
    for p in ['point_cloud/iteration_1/point_cloud.ply','point_cloud/iteration_7000/point_cloud.ply','point_cloud/iteration_30000/point_cloud.ply','chkpnt30000.pth','input.ply']:assert (OUT/p).is_file(),p
    assert (ROOT/'cluster_pruning/after.ply').is_file()
    write(ROOT/'training_complete.json',dict(status='complete',iterations=30000,gaussians=state['gaussians'],seed=0,training_wall_seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,native_checkpoint=str(OUT/'chkpnt30000.pth'),native_2d_gaussian_ply=str(OUT/'point_cloud/iteration_30000/point_cloud.ply')))
if __name__=='__main__':main()

