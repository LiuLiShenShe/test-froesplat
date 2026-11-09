import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch, ctypes, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
libpath=next((root/'.venv-nksr/lib/python3.12/site-packages/nvidia/cublas/lib').glob('libcublas.so.12'))
blas=ctypes.CDLL(str(libpath));handle=ctypes.c_void_p()
for name,args in [('cublasCreate_v2',[ctypes.POINTER(ctypes.c_void_p)]),('cublasSetStream_v2',[ctypes.c_void_p,ctypes.c_void_p]),('cublasSetPointerMode_v2',[ctypes.c_void_p,ctypes.c_int]),('cublasSdot_v2',[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]),('cublasDestroy_v2',[ctypes.c_void_p])]:
 f=getattr(blas,name);f.argtypes=args;f.restype=ctypes.c_int
x=torch.tensor([1.,2.],device='cuda');y=x.clone()
assert blas.cublasCreate_v2(ctypes.byref(handle))==0
assert blas.cublasSetStream_v2(handle,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))==0
assert blas.cublasSetPointerMode_v2(handle,0)==0
z=ctypes.c_float();print('private cuBLAS handle ready',flush=True)
rc=blas.cublasSdot_v2(handle,2,ctypes.c_void_p(x.data_ptr()),1,ctypes.c_void_p(y.data_ptr()),1,ctypes.byref(z))
print('cublasSdot return',rc,'value',z.value,flush=True)
assert rc==0 and abs(z.value-5)<1e-6
blas.cublasDestroy_v2(handle)
(root/'camera_interface/cublas_private_context_validation.json').write_text(json.dumps(dict(status='passed',dot=z.value,library=str(libpath)),indent=2)+'\n')
import sys
sys.stdout.flush();os._exit(0)
