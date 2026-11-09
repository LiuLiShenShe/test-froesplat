"""Observe official 2DGS training without changing its numerical loop or RNG."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,time,json,hashlib,csv,inspect,importlib.util
from pathlib import Path
from argparse import ArgumentParser
ROOT=Path(__file__).resolve().parents[1];DUAL=ROOT.parent/'geometry_rgb_dual_frontend';REPO=ROOT/'camera_interface/2dgs_corrected';OUT=ROOT/'E11_2DGS_Corrected'
sys.path.insert(0,str(REPO))
import torch
from arguments import ModelParams,PipelineParams,OptimizationParams
from utils.general_utils import safe_state
from e11_camera_validation import check_cameras

def main():
    assert json.loads((ROOT/'camera_interface/E11_validation.json').read_text())['status']=='passed'
    assert not (OUT/'checkpoints/point_cloud/iteration_30000/point_cloud.ply').exists()
    spec=importlib.util.spec_from_file_location('official_training',REPO/'train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parser=ArgumentParser();mp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    argv=['-s',str(DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected'),'-m',str(OUT/'checkpoints'),'-r','2','--iterations','30000','--sh_degree','3','--lambda_normal','0.05','--lambda_dist','0.2','--depth_ratio','0.0']
    args=parser.parse_args(argv);dataset,opt,pipe=mp.extract(args),op.extract(args),pp.extract(args)
    safe_state(False);torch.cuda.reset_peak_memory_stats();started=time.time()
    params=dict(dataset=vars(dataset),opt=vars(opt),pipe=vars(pipe),argv=argv,seed=0,physical_gpu=1,torch=torch.__version__,source=str(REPO),source_sha256={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in ['train.py','scene/gaussian_model.py','scene/cameras.py','gaussian_renderer/__init__.py']})
    (OUT/'actual_training_parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    old_scene=module.Scene
    def observed_scene(*a,**kw):
        s=old_scene(*a,**kw);(OUT/'runtime_cameras.json').write_text(json.dumps(check_cameras(s.getTrainCameras()),indent=2)+'\n')
        assert s.gaussians.get_xyz.shape[0]==1252383
        return s
    module.Scene=observed_scene
    report=module.training_report;last={'iteration':0,'gaussians':0}
    metrics=(OUT/'train_metrics.csv').open('w',newline='');writer=csv.writer(metrics);writer.writerow(['iteration','rgb_loss','dist_loss','normal_loss','total_loss','surfel_count','elapsed_seconds'])
    def observed_report(tb,iteration,Ll1,loss,l1_loss,elapsed,testing,scene,renderFunc,renderArgs):
        if iteration%100==0:
            loc=inspect.currentframe().f_back.f_locals
            writer.writerow([iteration,float(loss),float(loc['dist_loss']),float(loc['normal_loss']),float(loc['total_loss']),len(scene.gaussians.get_xyz),time.time()-started]);metrics.flush()
            last.update(iteration=iteration,gaussians=len(scene.gaussians.get_xyz))
            (ROOT/'STATUS.json').write_text(json.dumps(dict(status='running',stage='E11_30K_training',iteration=iteration,gaussians=last['gaussians'],physical_gpu=1),indent=2)+'\n')
        return report(tb,iteration,Ll1,loss,l1_loss,elapsed,testing,scene,renderFunc,renderArgs)
    module.training_report=observed_report
    # No interactive GUI clients in this experiment; official numerical loop remains intact.
    module.network_gui.try_connect=lambda *a,**kw:None
    module.training(dataset,opt,pipe,[7000,15000,30000],[7000,15000,30000],[7000,15000,30000],None)
    torch.cuda.synchronize();metrics.close()
    done=dict(status='complete',iterations=last['iteration'],surfel_count=last['gaussians'],training_wall_seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,physical_gpu=1,seed=0)
    assert done['iterations']==30000
    for name in ['point_cloud/iteration_30000/point_cloud.ply','chkpnt30000.pth']:assert (OUT/'checkpoints'/name).is_file()
    (OUT/'training_complete.json').write_text(json.dumps(done,indent=2)+'\n');print(json.dumps(done),flush=True)

if __name__=='__main__':main()
