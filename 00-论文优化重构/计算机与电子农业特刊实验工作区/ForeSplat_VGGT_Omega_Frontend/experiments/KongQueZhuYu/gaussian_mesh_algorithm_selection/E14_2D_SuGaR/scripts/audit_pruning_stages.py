"""Bitwise native before/removed/after checks on the actual DBSCAN artifacts."""
from pathlib import Path
import gc,json,hashlib
import numpy as np
import torch
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def verify_checkpoint(path,vertices,expected_iteration=7000):
 state,iteration=torch.load(path,map_location='cpu',weights_only=False)
 assert iteration==expected_iteration and len(state)==12 and state[4].shape==(len(vertices),2)
 arrays={k:state[1][:,i].detach().numpy() for i,k in enumerate(['x','y','z'])}
 arrays['opacity']=state[6][:,0].detach().numpy()
 for i in range(3):arrays[f'f_dc_{i}']=state[2][:,0,i].detach().numpy()
 for i in range(45):arrays[f'f_rest_{i}']=state[3][:,i%15,i//15].detach().numpy()
 for i in range(2):arrays[f'scale_{i}']=state[4][:,i].detach().numpy()
 for i in range(4):arrays[f'rot_{i}']=state[5][:,i].detach().numpy()
 for key,array in arrays.items():assert np.array_equal(vertices[key].view('u4'),array.view('u4')),(path.name,key,'PLY/checkpoint bit mismatch')
 del state,arrays;gc.collect()

def main():
 folder=ROOT/'cluster_pruning';info=json.loads((folder/'complete.json').read_text())
 assert info['status']=='complete'
 before=Path(info['before_path']);after=Path(info['after_path']);removed=Path(info['removed_path'])
 b=PlyData.read(before,mmap='r')['vertex'].data
 a=PlyData.read(after,mmap='r')['vertex'].data
 d=PlyData.read(removed,mmap='r')['vertex'].data
 with np.load(folder/'removed_indices.npz') as z:deleted=z['removed']
 assert deleted.ndim==1 and np.array_equal(deleted,np.unique(deleted))
 assert len(b)==info['before'] and len(a)==info['after'] and len(d)==info['removed']==len(deleted)
 assert len(a)+len(d)==len(b) and (not len(deleted) or 0<=deleted[0]<=deleted[-1]<len(b))
 assert b.dtype==a.dtype==d.dtype
 keep=np.ones(len(b),dtype=bool);keep[deleted]=False;retained=np.flatnonzero(keep)
 for index,ids in [(a,retained),(d,deleted)]:
  for start in range(0,len(ids),100000):
   assert np.array_equal(index[start:start+100000].view('u1'),b[ids[start:start+100000]].view('u1')),'DBSCAN before/after contains an extra numerical change'
 nonfinite_before=int((~np.isfinite(b.view('f4').reshape(len(b),-1)).all(1)).sum())
 nonfinite_after=int((~np.isfinite(a.view('f4').reshape(len(a),-1)).all(1)).sum())
 verify_checkpoint(Path(info['before_checkpoint']),b)
 verify_checkpoint(Path(info['after_checkpoint']),a)
 result=dict(status='complete',iteration=7000,before=len(b),after=len(a),removed=len(d),
  all_PLY_fields_bitwise_partition_source=True,before_and_after_checkpoint_fields_bitwise_match=True,
  nonfinite_native_primitives_before=nonfinite_before,nonfinite_native_primitives_after=nonfinite_after,
  numerical_quality_is_separate_from_bitwise_provenance=True,original_NaN_bits_retained=True,
  native_schedule='Scene.save immediately before DBSCAN; optimizer.step later. Native7K train checkpoint saved after optimizer is a separate state.',
  artifacts=[dict(path=str(p.resolve()),file_size_bytes=p.stat().st_size,sha256=digest(p)) for p in [before,after,removed,Path(info['before_checkpoint']),Path(info['after_checkpoint'])]],
  attribution='Before/after PLY and corresponding TSDF use an exact deletion-only intervention, not an additional optimization step. No claim of unique prior-vs-loss causality.')
 (folder/'stage_provenance_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('Actual native7K DBSCAN bitwise partition/checkpoints verified',len(b),len(a),len(d),flush=True)

if __name__=='__main__':main()
