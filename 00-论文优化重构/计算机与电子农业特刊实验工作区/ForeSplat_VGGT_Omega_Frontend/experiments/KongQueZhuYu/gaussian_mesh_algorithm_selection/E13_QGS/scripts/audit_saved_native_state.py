"""Read-only structured audit for fixed and rolling native QGS checkpoints."""
import argparse,hashlib,json,time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);args=parser.parse_args()
 torch.set_num_threads(8);rolling=args.checkpoint is None
 checkpoint=(ROOT/'recovery/latest_native_state.pth' if rolling else args.checkpoint).resolve(strict=True)
 before=checkpoint.stat();meta_path=checkpoint.with_suffix('.provenance.json')
 if rolling:metadata=json.loads(meta_path.read_text())
 else:
  params=json.loads((ROOT/'actual_training_parameters.json').read_text())
  if params.get('resume_checkpoint')==str(checkpoint) and (ROOT/'recovery/attempt_01_actual_training_parameters.json').is_file():params=json.loads((ROOT/'recovery/attempt_01_actual_training_parameters.json').read_text())
  metadata=dict(iteration=int(checkpoint.stem.removeprefix('chkpnt')),run_id=params.get('run_id'),stage='Native Scene.save capture before densification and optimizer step')
 state,iteration=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert len(state)==12 and iteration==metadata['iteration']
 n=len(state[1]);assert n>0 and state[4].shape==(n,6)
 names=['xyz','f_dc','f_rest','scaling','rotation','opacity','max_radii2D','xyz_gradient_accum','denom']
 fields=[];invalid=torch.zeros(n,dtype=torch.bool)
 for name,value in zip(names,state[1:10]):
  assert value.shape[0]==n
  bad=(~torch.isfinite(value)).reshape(n,-1).any(1)
  if name in names[:6]:invalid|=bad
  fields.append(dict(field=name,shape=list(value.shape),nonfinite_primitives=int(bad.sum()),NaNs=int(torch.isnan(value).sum()),infinities=int(torch.isinf(value).sum()),first_bad_native_indices=bad.nonzero().flatten()[:20].tolist()))
 for group in state[10]['param_groups']:
  for pid in group['params']:
   for key,value in state[10]['state'].get(pid,{}).items():
    if torch.is_tensor(value) and value.ndim:
     assert value.shape[0]==n
     bad=(~torch.isfinite(value)).reshape(n,-1).any(1)
     fields.append(dict(field=group.get('name','parameter')+'.Adam.'+key,nonfinite_primitives=int(bad.sum()),first_bad_native_indices=bad.nonzero().flatten()[:20].tolist()))
 # Decode actual float32 forward parameters on CPU, without writing back.
 # This is distinct from the float64 curvature diagnostic export.
 assert state[4].dtype==state[5].dtype==torch.float32
 signed_scale=torch.exp(state[4][:,:3])*torch.tanh(state[4][:,3:6])
 unit_rotation=torch.nn.functional.normalize(state[5],p=2,dim=1,eps=1e-12)
 decoded_bad=(~torch.isfinite(signed_scale)).any(1)|(~torch.isfinite(unit_rotation)).any(1)
 forward_decode=dict(dtype='torch.float32 CPU; no guarantee of bitwise CUDA transcendental equality',
  scale_equation='exp(raw_scale0..2)*tanh(raw_scale3..5)',rotation_equation='raw_quaternion/max(L2norm,1e-12)',
  nonfinite_decoded_primitives=int(decoded_bad.sum()),additional_nonfinite_decode_from_finite_raw=int((decoded_bad&~invalid).sum()),
  signed_scale_exact_zero_primitives=int((signed_scale==0).any(1).sum()),
  zero_raw_quaternion_primitives=int((state[5]==0).all(1).sum()),
  zero_decoded_quaternion_primitives=int((unit_rotation==0).all(1).sum()),
  finite_signed_scale_min_abs_per_axis=[float(v) if torch.isfinite(v) else None for v in torch.where(torch.isfinite(signed_scale),signed_scale.abs(),torch.inf).amin(0)],
  finite_signed_scale_max_abs_per_axis=[float(v) if torch.isfinite(v) else None for v in torch.where(torch.isfinite(signed_scale),signed_scale.abs(),-torch.inf).amax(0)],
  original_parameters_unchanged=True,quadratic_scale_is_not_ordinary_Gaussian_thickness=True)
 digest=hashlib.sha256()
 with checkpoint.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 after=checkpoint.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Checkpoint replaced during audit; retry stable actual state'
 if rolling:assert json.loads(meta_path.read_text())==metadata,'Producer metadata changed during audit'
 count=int(invalid.sum())
 result=dict(status='actual_checkpoint_structure_verified',observed_iteration=iteration,checkpoint=str(checkpoint),file_size_bytes=after.st_size,observed_sha256=digest.hexdigest(),primitives=n,native_signed_scale_channels=6,active_sh_degree=state[0],fields=fields,
  nonfinite_native_primitives=count,nonfinite_native_fraction=count/n,all_native_parameters_finite=count==0,
  numeric_defects_retained_unchanged=True,CPU_only=True,phase=metadata['stage'],run_id=metadata.get('run_id'),not_terminal_30000=iteration!=30000,audit_time=time.time(),native_forward_parameter_decode=forward_decode,
  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  mutable_path_note='Rolling path: count and SHA describe the observed state only, not a separately retained intermediate model.' if rolling else 'Actual fixed native checkpoint; provenance success is distinct from numerical quality.')
 target=ROOT/f'recovery/{"rolling" if rolling else "fixed"}_checkpoint_validation_{iteration}.json'
 temporary=target.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(target)
 print('Structured native state audit',iteration,'primitives',n,'nonfinite',count,'sha256',digest.hexdigest(),flush=True)

if __name__=='__main__':main()
