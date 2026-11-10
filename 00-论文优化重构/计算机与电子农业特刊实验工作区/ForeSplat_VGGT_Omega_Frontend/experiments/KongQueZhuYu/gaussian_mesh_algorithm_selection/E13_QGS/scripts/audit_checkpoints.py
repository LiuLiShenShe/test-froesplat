"""Validate real native QGS checkpoints against PLY fields, including partial stages."""
from pathlib import Path
import json,hashlib,gc,argparse
import numpy as np
import torch
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--allow-partial',action='store_true')
 args=parser.parse_args()
 records=[]
 for iteration in [7000,15000,30000]:
  checkpoint=ROOT/f'checkpoints/chkpnt{iteration}.pth'
  ply=ROOT/f'checkpoints/point_cloud/iteration_{iteration}/point_cloud.ply'
  if not checkpoint.exists() or not ply.exists():continue
  state,actual=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert actual==iteration and len(state)==12
  v=PlyData.read(ply,mmap='r')['vertex'].data
  n=state[1].shape[0]
  assert len(v)==n and state[4].shape==(n,6)
  expected=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(6)]+[f'rot_{i}' for i in range(4)]
  assert list(v.dtype.names)==expected
  arrays=dict(x=state[1][:,0].detach().numpy(),y=state[1][:,1].detach().numpy(),z=state[1][:,2].detach().numpy(),opacity=state[6][:,0].detach().numpy())
  for i in range(3):arrays[f'f_dc_{i}']=state[2][:,0,i].detach().numpy()
  for i in range(45):arrays[f'f_rest_{i}']=state[3][:,i%15,i//15].detach().numpy()
  for i in range(6):arrays[f'scale_{i}']=state[4][:,i].detach().numpy()
  for i in range(4):arrays[f'rot_{i}']=state[5][:,i].detach().numpy()
  nonfinite_rows=np.zeros(n,dtype=bool);nonfinite_fields={}
  for name,a in arrays.items():
   finite=np.isfinite(a);nonfinite_rows|=~finite
   if not finite.all():
    nonfinite_fields[name]=dict(nan=int(np.isnan(a).sum()),infinity=int(np.isinf(a).sum()))
   # Native optimization can leave invalid primitives while rendering a finite
   # image. Preserve and report them. Exact IEEE-754 bits, including NaN payloads,
   # validate that the exported native PLY really came from this checkpoint.
   assert a.dtype==np.float32 and np.asarray(v[name]).dtype==np.float32
   assert np.array_equal(a.view(np.uint32),np.asarray(v[name]).view(np.uint32)),(iteration,name,'PLY/checkpoint bit mismatch')
  records.append(dict(iteration=actual,stage='terminal_30000' if actual==30000 else f'partial_{actual}',native_primitive='QGS signed six-scale quadratic',
    primitives=n,checkpoint_path=str(checkpoint.resolve()),checkpoint_size=checkpoint.stat().st_size,checkpoint_sha256=digest(checkpoint),
    native_ply_path=str(ply.resolve()),native_ply_size=ply.stat().st_size,native_ply_sha256=digest(ply),
    all_native_fields_bitwise_match_checkpoint=True,all_native_fields_finite=not bool(nonfinite_rows.any()),
    native_nonfinite_primitives=int(nonfinite_rows.sum()),native_nonfinite_fraction=float(nonfinite_rows.mean()),
    native_nonfinite_field_counts=nonfinite_fields,
    quality_warning='Native invalid primitives are retained unchanged, including original NaN payloads; provenance pass is not a numerical-stability pass.' if nonfinite_rows.any() else None,
    active_sh_degree=state[0],viewing='native QGS renderer; not ordinary SuperSplat Gaussian'))
  del state,v,arrays;gc.collect()
 has_terminal=any(r['iteration']==30000 for r in records)
 assert records and (has_terminal or args.allow_partial),'No actual terminal checkpoint'
 result=dict(status='passed' if has_terminal else 'passed_partial',terminal_iteration=30000 if has_terminal else None,
  validation_scope='Exact native checkpoint/PLY provenance; nonfinite primitive quality defects are explicitly reported and not silently filtered',
  all_retained_models_finite=all(r['all_native_fields_finite'] for r in records),
  partial_checkpoints_not_treated_as_terminal=True,records=records)
 target='partial_checkpoint_provenance_validation.json' if args.allow_partial else 'checkpoint_provenance_validation.json'
 (ROOT/target).write_text(json.dumps(result,indent=2)+'\n')
 print('QGS native partial/terminal checkpoint provenance passed',[r['iteration'] for r in records],flush=True)
if __name__=='__main__':main()

