"""Read the actual native30K capture and verify its PLY, retaining NaN bits."""
import json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from audit_pruning_stages import digest,verify_checkpoint

ROOT=Path(__file__).resolve().parents[1]

def main():
    info=json.loads((ROOT/'training_complete.json').read_text())
    assert info['status']=='complete' and info['iterations']==30000
    model=Path(info['native_2d_gaussian_ply']).resolve(strict=True)
    checkpoint=Path(info['native_checkpoint']).resolve(strict=True)
    vertices=PlyData.read(model,mmap='r')['vertex'].data
    verify_checkpoint(checkpoint,vertices,expected_iteration=30000)
    state,iteration=torch.load(checkpoint,map_location='cpu',weights_only=False)
    raw_nonfinite=int((~np.isfinite(vertices.view('f4').reshape(len(vertices),-1)).all(1)).sum())
    decoded_scale=torch.exp(state[4]);decoded_rotation=torch.nn.functional.normalize(state[5],dim=-1)
    forward_bad=(~torch.isfinite(decoded_scale).all(1))|(~torch.isfinite(decoded_rotation).all(1))
    result=dict(status='complete',iteration=iteration,primitives=len(vertices),native_scale_fields=2,
        active_sh_degree=int(state[0]),all_saved_PLY_parameters_bitwise_equal=True,
        raw_nonfinite_native_primitives=raw_nonfinite,nonfinite_forward_scale_or_rotation_primitives=int(forward_bad.sum()),
        all_raw_saved_parameters_finite=raw_nonfinite==0,numerical_quality_separate_from_provenance=True,
        original_parameters_and_NaN_bits_unchanged=True,CPU_only=True,
        phase='Native iteration30000 Scene.save and capture; no final optimizer step or densification follows, so their parameters must match.',
        artifacts=[dict(path=str(path),file_size_bytes=path.stat().st_size,SHA256=digest(path)) for path in [model,checkpoint]])
    (ROOT/'native_terminal_provenance_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Actual native2D terminal provenance',len(vertices),'raw nonfinite',raw_nonfinite,flush=True)

if __name__=='__main__':main()
