"""Use the official GS2Mesh DLNR model and TSDF, with E10 GGGS stereo images.
Adapter fixes repeated padding of LR/RL inference and exposes consistency confidence.
No Gaussian training or replacement stereo estimator is performed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse, hashlib, json, shutil, sys, time, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=ROOT/"official"
HISTORY=ROOT.parents[1]/"geometry_mesh_final_selection"
sys.path.insert(0,str(OFFICIAL))
from render_e10_stereo import write

class CachedRenderer:
    def __init__(self):
        doc=json.loads((ROOT/"stereo/camera_data.json").read_text())
        self.cameras=doc["pairs"]
        for p in self.cameras:
            for c in p.values():
                for k in ["intrinsic","extrinsic"]: c[k]=np.asarray(c[k])
        self.left_cameras=[p["left"] for p in self.cameras]
        self.baseline=doc["baseline"]
        self.output_dir_root=str(ROOT/"meshes")
        Path(self.output_dir_root).mkdir(parents=True,exist_ok=True)
    def __len__(self): return len(self.cameras)
    def render_folder_name(self,i): return str(ROOT/"stereo"/f"{i:03}")
    def render_image_pair(self,i,visualize=False):
        out=Path(self.render_folder_name(i))
        assert (out/"render_complete.json").exists()
        assert (out/"left.png").exists() and (out/"right.png").exists()

def parent_free_view_tasks():
    """Parent-authorized16 GGGS audit views, separate from P3 algorithm timing."""
    selection=ROOT.parent
    base=ROOT.parents[3]
    script=selection/'scripts/render_gggs_free.py'
    if not script.exists():return
    tasks=[('E10',base/'experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'),
           ('P2',selection/'P2_GGGS_Cleaning/cleaned_gaussians.ply')]
    for experiment,ply in tasks:
        out=selection/'comparison/visualizations'/(experiment+'_free')
        if (out/'complete.json').exists():continue
        command=['uv','run','--no-project','--python',str(base/'.venv/bin/python'),'python',str(script),'--ply',str(ply),'--out',str(out)]
        record=dict(stage=experiment+'_shared_free_views',command=command,cwd=str(base),started_unix=time.time(),
                    source_experiment=experiment,separate_from_P3_algorithm=True,physical_gpu=1)
        log=selection/'logs'/(experiment+'_shared_free_views.log');log.parent.mkdir(parents=True,exist_ok=True)
        with (selection/'additional_visual_commands.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print('Parent audit free views before DLNR',experiment,flush=True)
        with log.open('a') as f:
            code=subprocess.call(command,cwd=base,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
        record.update(returncode=code,ended_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
        with (selection/'additional_visual_commands.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if code or not (out/'complete.json').exists():raise RuntimeError('Parent '+experiment+' free view audit failed; '+str(log))

def run_stereo(a,renderer):
    parent_free_view_tasks()
    import torch
    from gs2mesh_utils.stereo_utils import Stereo
    args=argparse.Namespace(stereo_model="DLNR_Middlebury",stereo_warm=False,stereo_occlusion_threshold=3,stereo_shading_eps=1e-4)
    torch.set_num_threads(8);torch.manual_seed(0);np.random.seed(0)
    start=time.time();stereo=Stereo(str(OFFICIAL),renderer,args,device="cuda")
    torch.cuda.reset_peak_memory_stats()
    records=[]
    for i,c in enumerate(renderer.left_cameras):
        if a.limit is not None and i>=a.limit: break
        renderer.render_image_pair(i)
        folder=Path(renderer.render_folder_name(i));out=folder/"out_DLNR_Middlebury"
        out.mkdir(parents=True,exist_ok=True)
        if (out/"complete.json").exists():
            records.append(json.loads((out/"complete.json").read_text()));continue
        t=time.time()
        image1=stereo.load_image(str(folder/"left.png"));image2=stereo.load_image(str(folder/"right.png"))
        padder=stereo.input_padder(image1.shape,divis_by=32)
        left,right=padder.pad(image1,image2)  # pad exactly once; both directions unpad identically
        disparities={}
        with torch.no_grad():
            for direction in ["LR","RL"]:
                x,y=(left,right) if direction=="LR" else (torch.flip(right,dims=[3]),torch.flip(left,dims=[3]))
                _,flow=stereo.model(x,y,iters=stereo.model_args.valid_iters,test_mode=True,flow_init=None)
                if direction=="RL":flow=torch.flip(flow,dims=[3])
                disparities[direction]=-padder.unpad(flow).cpu().numpy().squeeze().astype(np.float32)
                np.save(out/("disparity_"+direction+".npy"),disparities[direction])
        d=disparities["LR"];r=disparities["RL"]
        assert d.shape==r.shape==(c["height"],c["width"])
        valid=np.isfinite(d)&(d>0)
        dsafe=np.where(valid,d,0)
        rsafe=np.where(np.isfinite(r),r,0)
        occlusion=stereo.get_occlusion_mask(dsafe,rsafe,args.stereo_occlusion_threshold)&valid
        yy,xx=np.mgrid[:d.shape[0],:d.shape[1]]
        projected=(xx-dsafe).astype(np.int32)
        clipped=np.clip(projected,0,d.shape[1]-1)
        residual=np.abs(xx-np.clip(clipped+rsafe[yy,clipped],0,d.shape[1]-1))
        confidence=np.exp(-residual/args.stereo_occlusion_threshold).astype(np.float32)
        confidence[(projected<0)|(projected>=d.shape[1])|~valid]=0
        with np.errstate(invalid="ignore",divide="ignore"):
            raw_depth=c["fx"]*renderer.baseline/d
        depth=np.where(np.isfinite(raw_depth)&(raw_depth>0),raw_depth,0).astype(np.float32)
        np.save(out/"depth_raw.npy",raw_depth.astype(np.float32))
        np.save(out/"depth.npy",depth)
        np.save(out/"occlusion_mask.npy",occlusion)
        np.save(out/"confidence.npy",confidence)
        np.save(out/"consistency_residual_px.npy",residual.astype(np.float32))
        Image.fromarray(occlusion.astype(np.uint8)*255).save(out/"occlusion_mask.png")
        Image.fromarray((confidence*255).astype(np.uint8)).save(out/"confidence.png")
        mask=np.load(folder/"left_mask.npy").astype(bool)
        rec=dict(source_id=c["source_id"],name=c["name"],elapsed_seconds=time.time()-t,shape=list(d.shape),
                 fg_pixels=int(mask.sum()),fg_valid_depth_pixels=int((mask&(depth>0)).sum()),
                 fg_stereo_consistent_pixels=int((mask&occlusion).sum()),path=str(out),
                 confidence="exp(-native left-right reprojection residual/3px); consistency score, not calibrated network probability")
        write(out/"complete.json",rec);records.append(rec)
        print("native DLNR stereo",i,c["source_id"],rec["elapsed_seconds"],flush=True)
        del image1,image2,left,right,flow
        torch.cuda.empty_cache()
    done=a.limit is None or a.limit>=208
    write(ROOT/("stereo_complete.json" if done else "stereo_smoke.json"),
       dict(status="complete" if done else "partial",views=len(records),wall_seconds=time.time()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            native_model=str(OFFICIAL/"third_party/DLNR/pretrained/DLNR_Middlebury.pth"),
            weight_sha256=hashlib.sha256((OFFICIAL/"third_party/DLNR/pretrained/DLNR_Middlebury.pth").read_bytes()).hexdigest(),
            native_valid_iters=10,physical_gpu=1,records=records))

def run_fusion(a,renderer):
    from gs2mesh_utils.tsdf_utils import TSDF
    sys.path.insert(0,str(HISTORY/"scripts"))
    from mesh_artifacts import save_meshes
    scale=json.loads((HISTORY/"configs/standard_spatial_scale.json").read_text())
    assert (ROOT/"stereo_complete.json").exists()
    if a.shared_depth_range:
        renderer.output_dir_root=str(ROOT/"meshes_shared_depth_range")
        Path(renderer.output_dir_root).mkdir(parents=True,exist_ok=True)
    mesh_out=Path(renderer.output_dir_root)
    args=argparse.Namespace(TSDF_valid=None,TSDF_skip=None,TSDF_voxel=scale["diagonal"],TSDF_sdf_trunc=scale["tsdf_sdf_trunc"],
       TSDF_dilate=1,TSDF_use_mask=True,TSDF_invert_mask=False,TSDF_erode_mask=False,
       TSDF_use_occlusion_mask=True,TSDF_min_depth_baselines=4,TSDF_max_depth_baselines=scale["tsdf_depth_trunc"]/renderer.baseline if a.shared_depth_range else 20,
       TSDF_scale=1.,TSDF_cleaning_threshold=50)
    start=time.time()
    stereo=argparse.Namespace(model_name="DLNR_Middlebury")
    tsdf=TSDF(renderer,stereo,args,"gs2mesh_E10")
    tsdf.run(visualize=False)
    tsdf.save_mesh()
    stats=save_meshes(tsdf.mesh,mesh_out,scale)
    summary=dict(status="complete",source_experiment="E10",algorithm="Official GS2Mesh DLNR stereo -> native ScalableTSDF",
      wall_seconds=time.time()-start,parameters=vars(args),scale=scale,views=208,depth_trunc=args.TSDF_max_depth_baselines*renderer.baseline,
      mask="Existing RAP without erosion; native stereo consistency mask",
      official_commit="560f3e8a2349aaad24fe9444ef1ca2770aeec5ab",coordinate_system="Unchanged E10 standard VGGT world gauge",**stats)
    write(mesh_out/"complete.json",summary)
    write(ROOT/("fusion_shared_depth_range_complete.json" if a.shared_depth_range else "fusion_complete.json"),summary)
    print("P3 fusion complete",stats,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["stereo","fusion","all"],default="all")
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--shared-depth-range",action="store_true",help="Supplementary fusion using historical maximum depth; native main retains20 baselines")
    a=p.parse_args()
    renderer=CachedRenderer()
    if a.stage in ["stereo","all"]:run_stereo(a,renderer)
    if a.stage in ["fusion","all"]:run_fusion(a,renderer)

if __name__=="__main__":
    main()
    # Historical native mesh workers can spend minutes faulting pages during
    # interpreter teardown after publication. Exit only after successful main.
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

