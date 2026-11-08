"""Validate a mapping correction with actual CPU Pillow/torch operations."""
import sys,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'controlled_fixes'))
from pixel_transform_adapter import square_model_affine,resize_affine,projection_from_intrinsics
from camera_numeric_audit import OUT,cuda_project,pinhole_project

def main():
    side=3840;load=1024;model=518
    # Float ramp makes the pixel-centre mapping directly observable. Evaluate
    # interior to exclude finite-support boundary changes in bicubic filtering.
    ramp=np.tile(np.arange(side,dtype=np.float32),(4,1))
    im=Image.fromarray(ramp)
    resized=np.array(im.resize((load,4),Image.Resampling.BICUBIC),copy=True)
    output=F.interpolate(torch.from_numpy(resized)[None,None],size=(4,model),mode='bilinear',align_corners=False)[0,0,1].numpy()
    x=np.arange(model);expected=(x+.5)*side/model-.5;old=x*side/model
    interior=(x>8)&(x<model-9)
    A=square_model_affine((2160,3840));A1024=square_model_affine((2160,3840),1024);B=resize_affine((1024,1024),(518,518))
    Kmodel=np.array([[765.,0,259.],[0,750.,259.],[0,0,1.]])
    Kcorrect=np.linalg.inv(A)@Kmodel;Ktrain=resize_affine((2160,3840),(1080,1920))@Kcorrect
    E=np.eye(4);xyz=np.array([[.01,.1,1.],[-.2,-.1,2.],[.1,.05,.6]])
    got=cuda_project(projection_from_intrinsics(Ktrain,1080,1920),E,xyz,1080,1920);want=pinhole_project(Ktrain,E,xyz)
    result={'experiment':'same resize operations, pixel transform only','GPU_used':False,'new_model_inference':False,'new_training':False,
            'float_ramp_median_original_px_error_old_affine':float(np.median(np.abs(output[interior]-old[interior]))),
            'float_ramp_median_original_px_error_pixel_centres':float(np.median(np.abs(output[interior]-expected[interior]))),
            'float_ramp_p95_original_px_error_pixel_centres':float(np.quantile(np.abs(output[interior]-expected[interior]),.95)),
            'composed_affine_max_error':float(np.max(np.abs(B@A1024-A))),
            'corrected_K_original':Kcorrect.tolist(),'corrected_K_train':Ktrain.tolist(),
            'P_to_K_projection_max_error_px':float(np.max(np.linalg.norm(got-want,axis=1))),
            'limitation':'Projection-only helper; GGGS CUDA ray-plane/pruning/normal paths must also consume cx/cy before declaring principal-point repair.'}
    assert result['composed_affine_max_error']<1e-12
    assert result['P_to_K_projection_max_error_px']<1e-9
    assert result['float_ramp_median_original_px_error_pixel_centres']<.02
    (OUT/'pixel_transform_control.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
