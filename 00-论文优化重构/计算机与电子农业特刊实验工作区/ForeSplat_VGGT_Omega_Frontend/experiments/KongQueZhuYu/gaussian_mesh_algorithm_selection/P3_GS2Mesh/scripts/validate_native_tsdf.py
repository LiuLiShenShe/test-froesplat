"""Known-plane CPU validation of official GS2Mesh TSDF, without synthetic model exports."""
import os
os.environ["OMP_NUM_THREADS"]="8"
import json,tempfile,sys
from pathlib import Path
from argparse import Namespace
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"official"))
from gs2mesh_utils.tsdf_utils import TSDF
import open3d as o3d

def main():
    with tempfile.TemporaryDirectory(prefix="foresplat_p3_native_fusion_validation_") as tmp:
        folder=Path(tmp);model=folder/"out_DLNR_Middlebury";model.mkdir()
        h=w=128;z=np.full((h,w),.8,dtype=np.float32)
        np.save(model/"depth.npy",z);np.save(model/"occlusion_mask.npy",np.ones((h,w),bool))
        np.save(folder/"left_mask.npy",np.ones((h,w),bool))
        Image.fromarray(np.full((h,w,3),128,np.uint8)).save(folder/"left.png")
        camera=dict(width=w,height=h,fx=200.,fy=201.,cx=63.2,cy=62.1,extrinsic=np.eye(4))
        class Renderer:
            left_cameras=[camera];baseline=.04
            def __len__(self):return 1
            def render_folder_name(self,i):return str(folder)
        renderer=Renderer()
        args=Namespace(TSDF_valid=None,TSDF_skip=None,TSDF_voxel=2.,TSDF_sdf_trunc=.02,TSDF_dilate=1,
             TSDF_use_mask=True,TSDF_invert_mask=False,TSDF_erode_mask=False,TSDF_use_occlusion_mask=True,
             TSDF_min_depth_baselines=4,TSDF_max_depth_baselines=40,TSDF_scale=1.,TSDF_cleaning_threshold=50)
        tsdf=TSDF(renderer,Namespace(model_name="DLNR_Middlebury"),args,"validation")
        tsdf.run(visualize=False)
        xyz=np.asarray(tsdf.mesh.vertices);tri=np.asarray(tsdf.mesh.triangles)
        error=float(np.max(np.abs(xyz[:,2]-.8))) if len(xyz) else None
        passed=bool(len(xyz) and len(tri) and error<.01)
        doc=dict(status="passed" if passed else "failed",synthetic_validation_only=True,vertices=int(len(xyz)),faces=int(len(tri)),
          median_depth=float(np.median(xyz[:,2])) if len(xyz) else None,depth_error=error,
          open3d_version=o3d.__version__,numpy_version=np.__version__,official_TSDF_import_and_camera_integration=True,
          noncentral_principal_point=[camera["cx"],camera["cy"]],synthetic_mesh_exported=False)
        (ROOT/("NATIVE_FUSION_CPU_VALIDATION_"+o3d.__version__+".json")).write_text(json.dumps(doc,indent=2)+"\n")
        (ROOT/"NATIVE_FUSION_CPU_VALIDATION.json").write_text(json.dumps(doc,indent=2)+"\n")
        print(doc,flush=True)
        if not passed:raise RuntimeError("Native known-plane TSDF validation failed")

if __name__=="__main__":
    main()
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

