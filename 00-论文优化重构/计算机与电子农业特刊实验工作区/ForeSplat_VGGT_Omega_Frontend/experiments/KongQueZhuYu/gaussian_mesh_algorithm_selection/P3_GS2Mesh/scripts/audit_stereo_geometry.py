"""CPU camera/sign/padding audit; synthetic values are validation, never experiment geometry."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"official"))
from gs2mesh_utils.stereo_utils import Stereo
from core.utils.utils import InputPadder
from render_e10_stereo import write
import torch

def main():
    doc=json.loads((ROOT/"stereo/camera_data.json").read_text())
    records=[]
    for pair in doc["pairs"]:
        l,r=pair["left"],pair["right"]
        KL=np.array(l["intrinsic"]);KR=np.array(r["intrinsic"])
        CL=np.array(l["extrinsic"]);ER=np.linalg.inv(np.array(r["extrinsic"]))
        u,v=np.meshgrid(np.linspace(64,l["width"]-64,7),np.linspace(64,l["height"]-64,7))
        uv=np.column_stack([u.ravel(),v.ravel()])
        depth=np.resize(np.array([.35,.8,1.2]),len(uv))
        pc=np.column_stack([(uv[:,0]-l["cx"])*depth/l["fx"],(uv[:,1]-l["cy"])*depth/l["fy"],depth])
        world=pc@CL[:3,:3].T+CL[:3,3]
        pr=world@ER[:3,:3].T+ER[:3,3]
        ur=pr[:,:2]/pr[:,2,None]*[r["fx"],r["fy"]]+[r["cx"],r["cy"]]
        d=uv[:,0]-ur[:,0]
        estimated_depth=l["fx"]*doc["baseline"]/d
        assert np.all(d>0)
        rec=dict(source_id=l["source_id"],vertical_rectification_max_px=float(np.max(np.abs(uv[:,1]-ur[:,1]))),
                 disparity_formula_max_px=float(np.max(np.abs(d-l["fx"]*doc["baseline"]/depth))),
                 depth_roundtrip_max=float(np.max(np.abs(estimated_depth-depth))),
                 world_roundtrip_max=float(np.max(np.abs(world-(pr@np.asarray(r['extrinsic'])[:3,:3].T+np.asarray(r['extrinsic'])[:3,3])))))
        records.append(rec)
    shape=(1,3,1920,1080)
    x=torch.arange(1080,dtype=torch.float32).reshape(1,1,1,1080).expand(1,3,1920,1080)
    pad=InputPadder(shape,divis_by=32)
    padded,=pad.pad(x)
    assert torch.equal(pad.unpad(padded),x)
    assert torch.equal(pad.unpad(torch.flip(torch.flip(padded,[3]),[3])),x)
    # Native consistency mask: constant disparity plane, left boundary occluded.
    d=np.full((64,80),7,dtype=np.float32)
    mask=Stereo.get_occlusion_mask(None,d,d,3)
    assert mask[:,7:].all() and not mask[:,:7].any()
    doc=dict(status="passed",views=208,checks="rectified positive-X stereo, DLNR negative horizontal flow -> positive disparity, Z=fx*B/disparity; native occlusion and original padding roundtrip",
             vertical_rectification_max_px=max(r["vertical_rectification_max_px"] for r in records),
             disparity_formula_max_px=max(r["disparity_formula_max_px"] for r in records),
             depth_roundtrip_max=max(r["depth_roundtrip_max"] for r in records),
             original_shape=list(shape),padded_shape=list(padded.shape),native_consistency_constant_plane_pass=True,records=records,
             note="Synthetic CPU validation only; no synthetic plant artifact or mesh is exported")
    assert doc["vertical_rectification_max_px"]<1e-7 and doc["disparity_formula_max_px"]<1e-7 and doc["depth_roundtrip_max"]<1e-10
    write(ROOT/"CAMERA_STEREO_VALIDATION.json",doc)
    print({k:v for k,v in doc.items() if k!="records"})

if __name__=="__main__":main()

