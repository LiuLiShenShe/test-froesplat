import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch,json
from pathlib import Path
x=torch.tensor([1.,2.],device='cuda');y=x.clone()
z=float(torch.sum(x*y));n=float(torch.linalg.vector_norm(x));assert z==5 and abs(n-5**.5)<1e-6
x.add_(y,alpha=.5);assert torch.equal(x,torch.tensor([1.5,3.],device='cuda'))
print({'dot':z,'norm':n,'add':x.tolist()},flush=True)
Path(__file__).resolve().parents[1].joinpath('camera_interface/aten_vector_ops_validation.json').write_text(json.dumps(dict(dot=z,norm=n,add=x.tolist()),indent=2)+'\n')
import sys
sys.stdout.flush();os._exit(0)
