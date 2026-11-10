"""CPU-only actual Metric3Dv2 constructor and downloaded-state compatibility audit; no forward or CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'metric3d_official'))
torch.set_num_threads(8);started=time.time()
model=torch.hub.load(str(ROOT/'metric3d_official'),'metric3d_vit_giant2',source='local',pretrain=False)
state=torch.load(ROOT/'weights/metric_depth_vit_giant2_800k.pth',map_location='cpu',weights_only=False,mmap=True)['model_state_dict']
result=model.load_state_dict(state,strict=False)
unused_keys={'depth_model.encoder.mask_token','depth_model.encoder.norm.weight','depth_model.encoder.norm.bias'}
assert model.depth_model.encoder.multi_output is True
record=dict(status='passed' if set(result.missing_keys)<=unused_keys and not result.unexpected_keys else 'failed',missing_key_reason='Official multi_output=True bypasses final norm; masksNone bypasses mask_token. No inference weights missing.',missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,model_parameters=sum(p.numel() for p in model.parameters()),state_tensors=len(state),torch=torch.__version__,forward_called=False,gpu_called=False,elapsed_seconds=time.time()-started,weight=str((ROOT/'weights/metric_depth_vit_giant2_800k.pth').resolve()),source='Exact official local torch.hub constructor/pretrainFalse + SHA256-verified full checkpoint')
(ROOT/'metric3d_cpu_load_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
assert record['status']=='passed'

