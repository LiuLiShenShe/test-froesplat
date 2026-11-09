import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch,json
from pathlib import Path
from types import SimpleNamespace
from nksr.solver import SparseMatrix
p=Path(__file__).resolve().parents[1];records=[];torch.manual_seed(0)
for n in [2,10000,1000000]:
 index=torch.arange(n,device='cuda',dtype=torch.int32);values=torch.full((n,),2.,device='cuda');rhs=torch.linspace(.1,1.,n,device='cuda')
 matrix=SparseMatrix(1);matrix.add_block(0,0,n,n,index,index,values)
 result=matrix.solve({0:rhs},SimpleNamespace(tol=1e-5,max_iter=20,verbose=True))[0]
 err=float((result-rhs/2).abs().max());assert err<1e-5,(n,err)
 records.append(dict(N=n,max_error=err));print(records[-1],flush=True)
# A dense SPD system split into unequal blocks exercises actual CG iterations,
# off-diagonal transpose matvec and convergence, with an independent CPU reference.
g=torch.Generator().manual_seed(19)
mat=torch.randn(257,257,dtype=torch.float64,generator=g)
mat=mat@mat.T/257+torch.eye(257,dtype=torch.float64)*.5
truth=torch.linspace(-.5,.8,257,dtype=torch.float64)
rhs=mat@truth;reference=torch.linalg.solve(mat,rhs)
blocks=SparseMatrix(2);sizes=[128,129];offset=[0,128,257]
for i,j in [(0,0),(0,1),(1,1)]:
 vals=mat[offset[i]:offset[i+1],offset[j]:offset[j+1]].float().contiguous().cuda()
 rows=torch.arange(sizes[i],dtype=torch.int32,device='cuda').repeat_interleave(sizes[j])
 cols=torch.arange(sizes[j],dtype=torch.int32,device='cuda').repeat(sizes[i])
 blocks.add_block(i,j,sizes[i],sizes[j],rows,cols,vals.reshape(-1))
out=blocks.solve({0:rhs[:128].float().cuda(),1:rhs[128:].float().cuda()},SimpleNamespace(tol=1e-5,max_iter=2000,verbose=True))
result=torch.cat([out[0],out[1]]).cpu().double()
err=float((result-reference).abs().max());resid=float(torch.linalg.vector_norm(mat@result-rhs)/torch.linalg.vector_norm(rhs))
assert err<3e-4 and resid<2e-5,(err,resid)
records.append(dict(N=257,case='Unequal two-block dense SPD',max_error=err,relative_residual=resid));print(records[-1],flush=True)
(p/'camera_interface/nksr_pcg_validation.json').write_text(json.dumps(records,indent=2)+'\n')

import sys
sys.stdout.flush();sys.stderr.flush();os._exit(0)
