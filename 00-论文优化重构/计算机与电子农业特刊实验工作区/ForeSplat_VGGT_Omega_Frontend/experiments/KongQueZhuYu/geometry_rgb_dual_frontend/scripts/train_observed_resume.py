"""Recover an owned GGGS checkpoint using native restore; preserve interrupted records."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,hashlib,importlib.util,json,sys,time,re,shutil
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--events',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--checkpoint-sha256',required=True)
parser.add_argument('--previous-observed-seconds',type=float,required=True)
parser.add_argument('--previous-last-iteration',type=int,required=True)
parser.add_argument('argv',nargs=argparse.REMAINDER)
a=parser.parse_args();argv=a.argv[1:] if a.argv and a.argv[0]=='--' else a.argv
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
if '--smoke' in argv or '--start_checkpoint' in argv:raise ValueError('Native full training arguments only; recovery checkpoint supplied separately')
if not a.events.exists() or a.events.with_suffix('.complete.json').exists():raise ValueError('Recovery requires existing interrupted records without completed training')
if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Changed owned checkpoint')
match=re.fullmatch(r'chkpnt(\d+)\.pth',a.checkpoint.name)
if not match:raise ValueError('Expected native iteration checkpoint')
first=int(match[1])
if not 0<first<30000:raise ValueError('Resume only a partial checkpoint before the fixed 30K endpoint')
sys.path.insert(0,str(a.repo))
spec=importlib.util.spec_from_file_location('actual_training',a.repo/'train_foresplat.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch
if torch.cuda.device_count()!=1:raise RuntimeError('Physical GPU1 only')
torch.set_num_threads(8)
load=torch.load
def owned_load(path,*args,**kwargs):
    if Path(path).resolve()!=a.checkpoint.resolve():return load(path,*args,**kwargs)
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Recovery checkpoint changed during training setup')
    kwargs['weights_only']=False
    return load(path,*args,**kwargs)
# This locally generated native checkpoint contains the GGGS App_model enum
# and optimizer state. Only its exact verified path gets this loading mode.
torch.load=owned_load
checkpoint=owned_load(a.checkpoint,map_location='cpu')
if not isinstance(checkpoint,tuple) or len(checkpoint)!=2 or checkpoint[1]!=first:
    raise ValueError('Checkpoint iteration does not match filename')
if checkpoint[0][0]!=3 or checkpoint[0][1]!=0:raise ValueError('Requires the locked SH3/SG0 GGGS training configuration')
del checkpoint
archive=a.events.parent/'resume_history'/str(int(time.time()));archive.mkdir(parents=True)
for name in [a.events.name,'actual_training_parameters.json','train_metrics.csv']:
    path=a.events.parent/name
    if path.exists():shutil.copy2(path,archive/name)
a.events.unlink()
state={'event':(first-500)//100}
old=m.GaussianModel.densify_and_prune
def observed(self,max_grad,min_opacity,extent,max_screen_size):
    state['event']+=1
    normal=self.xyz_gradient_accum/self.denom;normal[normal.isnan()]=0
    absolute=self.xyz_gradient_accum_abs/self.denom;absolute[absolute.isnan()]=0
    selected=torch.norm(normal,dim=-1)>=max_grad;ratio=selected.float().mean()
    quantile=torch.quantile(absolute.reshape(-1),1-ratio)
    row=dict(event=state['event'],iteration=500+100*state['event'],before=len(self._xyz),ratio=float(ratio),Q=float(quantile),
        denom_zero=int((self.denom==0).sum()),absolute_zero=int((absolute==0).sum()),
        zero_selected_by_abs=int(((absolute.reshape(-1)==0)&(absolute.reshape(-1)>=quantile)).sum()))
    result=old(self,max_grad,min_opacity,extent,max_screen_size)
    row.update(after=len(self._xyz),clone_added=int(result[0]),split_net_added=int(result[1]),pruned=int(result[2]))
    with a.events.open('a') as file:file.write(json.dumps(row)+'\n')
    return result
m.GaussianModel.densify_and_prune=observed
actual=m.training
def recorded(**kwargs):
    record={key:vars(kwargs[key]) for key in ['dataset','opt','pipe','fg']}
    record.update(argv=argv,physical_gpu=1,CUDA_VISIBLE_DEVICES='1',seed=0,eo_enabled=kwargs['eo'].enabled,
        testing_iterations=kwargs['testing_iterations'],saving_iterations=kwargs['saving_iterations'],checkpoint_iterations=kwargs['checkpoint_iterations'],
        source_repo=str(a.repo),source_sha256={name:sha(a.repo/name) for name in ['train_foresplat.py','scene/gaussian_model.py','scene/cameras.py','utils/loss_utils.py','arguments/__init__.py','gaussian_renderer/__init__.py']},
        native_resume_checkpoint=str(a.checkpoint),native_resume_iteration=first,checkpoint_sha256=a.checkpoint_sha256,
        interrupted_records=str(archive),resume_limitations='Native capture does not persist RNG/viewpoint-stack/filter/absolute-gradient state; restore uses native GGGS behavior and recomputes the 3D filter. No claim of bit-identical uninterrupted trajectory.')
    a.events.with_name('actual_training_parameters.json').write_text(json.dumps(record,indent=2))
    return actual(**kwargs)
m.training=recorded
argv+=['--start_checkpoint',str(a.checkpoint)]
begin=time.perf_counter();code=m.main(argv);continuation=time.perf_counter()-begin
a.events.with_suffix('.complete.json').write_text(json.dumps(dict(status='complete',
    wall_seconds=a.previous_observed_seconds+continuation,continuation_wall_seconds=continuation,
    previous_observed_wall_seconds_lower_bound=a.previous_observed_seconds,
    training_cost_scope='Previous observed tqdm elapsed lower bound plus exact native continuation wall time; downtime excluded; repeated iterations included',
    restored_iteration=first,previous_last_observed_iteration=a.previous_last_iteration,
    repeated_iterations=max(0,a.previous_last_iteration-first),argv=argv,repo=str(a.repo),physical_gpu=1,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
    observer_only=True,random_calls_added=0,interrupted_records=str(archive),checkpoint_sha256=a.checkpoint_sha256),indent=2))
raise SystemExit(code)
