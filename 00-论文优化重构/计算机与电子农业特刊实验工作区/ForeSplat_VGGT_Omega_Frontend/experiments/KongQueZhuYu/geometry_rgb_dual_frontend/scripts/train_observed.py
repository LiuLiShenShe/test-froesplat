"""Full original GGGS training with read-only densification observation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args()
argv=a.argv[1:] if a.argv and a.argv[0]=='--' else a.argv
assert '--smoke' not in argv
sys.path.insert(0,str(a.repo))
spec=importlib.util.spec_from_file_location('actual_training',a.repo/'train_foresplat.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']=='1' and torch.cuda.device_count()==1
torch.set_num_threads(8)
old=m.GaussianModel.densify_and_prune;state={'event':0}
def observed(self,max_grad,min_opacity,extent,max_screen_size):
    state['event']+=1
    normal=self.xyz_gradient_accum/self.denom;normal[normal.isnan()]=0
    absolute=self.xyz_gradient_accum_abs/self.denom;absolute[absolute.isnan()]=0
    selected=torch.norm(normal,dim=-1)>=max_grad;ratio=selected.float().mean();quantile=torch.quantile(absolute.reshape(-1),1-ratio)
    row=dict(event=state['event'],iteration=500+100*state['event'],before=len(self._xyz),ratio=float(ratio),Q=float(quantile),denom_zero=int((self.denom==0).sum()),
        absolute_zero=int((absolute==0).sum()),zero_selected_by_abs=int(((absolute.reshape(-1)==0)&(absolute.reshape(-1)>=quantile)).sum()))
    result=old(self,max_grad,min_opacity,extent,max_screen_size)
    row.update(after=len(self._xyz),clone_added=int(result[0]),split_net_added=int(result[1]),pruned=int(result[2]))
    with a.events.open('a') as f:f.write(json.dumps(row)+'\n')
    return result
m.GaussianModel.densify_and_prune=observed
actual_training=m.training
def recorded_training(**kwargs):
    record={key:vars(kwargs[key]) for key in ['dataset','opt','pipe','fg']}
    record.update(argv=argv,physical_gpu=1,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],seed=0,
        eo_enabled=kwargs['eo'].enabled,testing_iterations=kwargs['testing_iterations'],saving_iterations=kwargs['saving_iterations'],
        checkpoint_iterations=kwargs['checkpoint_iterations'],source_repo=str(a.repo),
        source_sha256={name:hashlib.sha256((a.repo/name).read_bytes()).hexdigest() for name in ['train_foresplat.py','scene/gaussian_model.py','scene/cameras.py','utils/loss_utils.py','arguments/__init__.py','gaussian_renderer/__init__.py']})
    a.events.with_name('actual_training_parameters.json').write_text(json.dumps(record,indent=2))
    return actual_training(**kwargs)
m.training=recorded_training
a.events.parent.mkdir(parents=True,exist_ok=True)
if a.events.exists():raise FileExistsError(a.events)
begin=time.perf_counter();rc=m.main(argv)
a.events.with_suffix('.complete.json').write_text(json.dumps(dict(status='complete',wall_seconds=time.perf_counter()-begin,argv=argv,repo=str(a.repo),physical_gpu=1,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,observer_only=True,random_calls_added=0),indent=2))
raise SystemExit(rc)
