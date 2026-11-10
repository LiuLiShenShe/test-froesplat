"""GGGS renderer adapter for official GS2Mesh; E10 checkpoint is read-only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[3]
DUAL = ROOT.parents[1] / "geometry_rgb_dual_frontend"
HISTORY = ROOT.parents[1] / "geometry_mesh_final_selection"
sys.path.append(str(HISTORY/'scripts'))
from surface_protocol import save_npz_fast

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)

def prepare():
    refs = json.loads((HISTORY / "surface_inputs/e10_gggs_surface/camera_runtime.json").read_text())["records"]
    assert len(refs) == 208
    centers = np.asarray([r["camera_center"] for r in refs])
    radius = float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1)))
    baseline = radius * .07  # native GS2Mesh custom default, 7% camera radius
    pairs = []
    for c in refs:
        E = np.asarray(c["world_view_transform"], dtype=np.float64).T
        C = np.linalg.inv(E)
        R = C.copy()
        R[:3, 3] += baseline * C[:3, 0]
        K = np.array([[c["Fx"], 0, c["Cx"]], [0, c["Fy"], c["Cy"]], [0, 0, 1]])
        # Rectified stereo must have unchanged rotation and y/z in left camera coordinates.
        translation_left = (R[:3, 3] - C[:3, 3]) @ E[:3, :3].T
        assert np.max(np.abs(translation_left - [baseline, 0, 0])) < 1e-7
        assert np.max(np.abs(R[:3, :3] - C[:3, :3])) == 0
        d = dict(source_id=c["source_id"], name=c["name"], width=c["width"], height=c["height"],
                 fx=c["Fx"], fy=c["Fy"], cx=c["Cx"], cy=c["Cy"], intrinsic=K.tolist())
        pairs.append(dict(left=dict(d, extrinsic=C.tolist()), right=dict(d, extrinsic=R.tolist())))
    assert sorted(p["left"]["source_id"] for p in pairs) == [i for i in range(210) if i not in [44, 206]]
    write(ROOT / "stereo/camera_data.json", dict(pairs=pairs, baseline=baseline, camera_radius=radius,
          baseline_percentage=7, coordinate_system="Existing E10 standard VGGT world gauge; OpenCV c2w",
          stereo_model="DLNR_Middlebury", rendering_resolution=[1080, 1920], camera_source=str(HISTORY / "surface_inputs/e10_gggs_surface/camera_runtime.json")))
    return pairs, baseline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    a=parser.parse_args()
    pairs, baseline=prepare()
    if a.prepare_only:
        print("208 stereo camera pairs prepared; baseline", baseline, flush=True)
        return
    repo=DUAL / "camera_interface/gggs_corrected"
    sys.path.insert(0,str(repo))
    from utils.cuda_preload import preload_cuda_ext_deps
    preload_cuda_ext_deps()
    from scene import GaussianModel, Scene
    from scene.cameras import Camera
    import gaussian_renderer as native
    torch.set_num_threads(8)
    torch.manual_seed(0); np.random.seed(0)
    checkpoint=DUAL/"E10_Standard_Corrected_Interface/checkpoints"
    runtime=ROOT/"runtime_checkpoint"
    linked=runtime/"point_cloud/iteration_30000/point_cloud.ply"
    linked.parent.mkdir(parents=True,exist_ok=True)
    original=checkpoint/"point_cloud/iteration_30000/point_cloud.ply"
    if not linked.exists(): linked.symlink_to(original)
    ns=argparse.Namespace(sh_degree=3,sg_degree=0,source_path=str(DUAL/"Standard_VGGT_Original_RGB/COLMAP_Corrected"),
       model_path=str(runtime),images="images",dataset="",resolution=2,white_background=False,data_device="cpu",eval=False,
       use_decoupled_appearance=0,disable_filter3D=False,kernel_size=0.,multi_view_num=8,multi_view_max_angle=30,
       multi_view_min_dis=.01,multi_view_max_dis=1.5)
    pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    gaussians=GaussianModel(3,0)
    start=time.time()
    scene=Scene(ns,gaussians,load_iteration=30000,shuffle=False)
    cameras={c.image_name:c for c in scene.getTrainCameras()}
    assert len(cameras)==208 and len(gaussians.get_xyz)==576396
    background=torch.zeros(3,device="cuda")
    torch.cuda.reset_peak_memory_stats()
    records=[]
    for i,pair in enumerate(pairs):
        if a.limit is not None and i>=a.limit: break
        left=cameras[pair["left"]["name"]]
        dst=ROOT/"stereo"/f"{i:03}"
        dst.mkdir(parents=True,exist_ok=True)
        if (dst/"render_complete.json").exists():
            records.append(json.loads((dst/"render_complete.json").read_text()))
            continue
        E=np.linalg.inv(np.asarray(pair["right"]["extrinsic"]))
        right=Camera(left.colmap_id,E[:3,:3].T,E[:3,3],left.FoVx,left.FoVy,left.original_image,
          None,left.image_name+"_right",left.uid,data_device="cpu",intrinsics=(left.Fx,left.Fy,left.Cx,left.Cy))
        assert np.max(np.abs(right.world_view_transform.cpu().numpy().T - E))<1e-6
        for key in ["Fx","Fy","Cx","Cy"]:
            assert abs(getattr(left,key)-getattr(right,key))<1e-8
        assert np.max(np.abs(left.projection_matrix.cpu().numpy() - right.projection_matrix.cpu().numpy()))<1e-7
        with torch.no_grad():
            outputs={}
            parity={}
            for name,view in [("left",left),("right",right)]:
                pkg=native.render(view,gaussians,pipe,background,0.)
                rgb=pkg["render"].permute(1,2,0).cpu().numpy().astype(np.float32)
                if name=="left" and pair['left']['source_id'] in [0,26,53,79,105,131,157,183]:
                    cached=HISTORY/'surface_inputs/e10_gggs_surface/rendered'/(left.image_name+'.npz')
                    with np.load(cached) as old:
                        delta=np.abs(rgb-old['render_raw'])
                    parity=dict(reference_render_max_abs=float(delta.max()),reference_render_mean_abs=float(delta.mean()),reference_render_path=str(cached))
                    assert parity['reference_render_max_abs']<1e-4, 'E10 native rendered left parity failed'
                Image.fromarray(np.clip(rgb*255,0,255).astype(np.uint8)).save(dst/(name+".png"),compress_level=1)
                outputs[name+"_alpha"]=pkg["mask"].cpu().numpy().squeeze().astype(np.float32)
                outputs[name+"_median_depth"]=pkg["median_depth"].cpu().numpy().squeeze().astype(np.float32)
                del pkg
            save_npz_fast(dst/"gggs_aux.npz",**outputs)
        fg=np.asarray(Image.open(Path(ns.source_path)/"masks"/(left.image_name+".png")).convert("L")
                      .resize((left.image_width,left.image_height),Image.Resampling.NEAREST))>127
        np.save(dst/"left_mask.npy",fg)
        Image.fromarray(fg.astype(np.uint8)*255).save(dst/"left_mask.png")
        rec=dict(source_id=pair["left"]["source_id"],name=left.image_name,path=str(dst),baseline=baseline,
                 left_mask_sha256=hashlib.sha256(fg.tobytes()).hexdigest(),**parity)
        write(dst/"render_complete.json",rec)
        records.append(rec)
        print("E10 stereo rendered",i,rec["source_id"],flush=True)
    done=a.limit is None or a.limit>=208
    write(ROOT/("render_complete.json" if done else "render_smoke.json"),
         dict(status="complete" if done else "partial",views=len(records),model_count=len(gaussians.get_xyz),checkpoint=str(original.resolve()),
              wall_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,physical_gpu=1,
              renderer=str(repo),records=records))

if __name__=="__main__": main()

