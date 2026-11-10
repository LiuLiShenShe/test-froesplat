"""Fixed eight source cameras, actual stereo and mesh images; no generated illustrations."""
import os
os.environ["MPLBACKEND"]="Agg"
os.environ["OMP_NUM_THREADS"]="8"
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT.parents[1]/"geometry_mesh_final_selection"
REPS=[0,26,53,79,105,131,157,183]
sys.path.insert(0,str(HISTORY/"scripts"))
from surface_protocol import write

def image(path,array):
    array=np.clip(array,0,1)
    Image.fromarray((array*255).astype(np.uint8)).save(path)

def stereo_panels():
    doc=json.loads((ROOT/"stereo/camera_data.json").read_text())
    views=ROOT/"visualizations/stereo";views.mkdir(parents=True,exist_ok=True)
    for i,pair in enumerate(doc["pairs"]):
        sid=pair["left"]["source_id"]
        if sid not in REPS:continue
        folder=ROOT/"stereo"/f"{i:03}";model=folder/"out_DLNR_Middlebury"
        if not (model/"complete.json").exists():continue
        mask=np.load(folder/"left_mask.npy").astype(bool)
        depth=np.load(model/"depth.npy");confidence=np.load(model/"confidence.npy");consistent=np.load(model/"occlusion_mask.npy")
        fig,axes=plt.subplots(1,5,figsize=(15,6))
        for ax,path,title in zip(axes[:2],[folder/"left.png",folder/"right.png"],["GGGS left","GGGS novel right"]):
            ax.imshow(Image.open(path));ax.set_title(title);ax.axis("off")
        shown=np.ma.masked_where(~mask|(depth<=0)|~np.isfinite(depth),depth)
        axes[2].imshow(shown,cmap="viridis",vmin=.15,vmax=1.57);axes[2].set_title("Native DLNR depth");axes[2].axis("off")
        axes[3].imshow(confidence,cmap="gray",vmin=0,vmax=1);axes[3].set_title("LR consistency score");axes[3].axis("off")
        axes[4].imshow(mask&consistent,cmap="gray");axes[4].set_title("RAP & native consistency");axes[4].axis("off")
        fig.suptitle("P3 E10 GGGS -> GS2Mesh, source "+str(sid))
        fig.tight_layout();fig.savefig(views/f"source_{sid:04d}.png",dpi=120);plt.close(fig)

def mesh_panels():
    import open3d as o3d
    from evaluate_meshes import cast
    cameras=json.loads((HISTORY/"surface_inputs/e10_gggs_surface/camera_runtime.json").read_text())["records"]
    index={r["source_id"]:r for r in json.loads((HISTORY/"surface_inputs/e10_gggs_surface/render_index.json").read_text())}
    light=np.array([.2,-.3,-1.]);light/=np.linalg.norm(light)
    for route in ["meshes","meshes_shared_depth_range"]:
        directory=ROOT/route
        if not (directory/"complete.json").exists():continue
        for stage in ["raw","post"]:
            mesh=o3d.io.read_triangle_mesh(str(directory/(stage+".ply")))
            if not len(mesh.vertices):continue
            v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
            scene=o3d.t.geometry.RaycastingScene(nthreads=8)
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            for c in cameras:
                sid=c["source_id"]
                if sid not in REPS:continue
                out=ROOT/"visualizations"/route/stage/f"source_{sid:04d}";out.mkdir(parents=True,exist_ok=True)
                depth,normal,rgb=cast(scene,c,v,f,colors);hit=np.isfinite(depth)&(depth>0)
                E=np.asarray(c["world_view_transform"]).T
                ncam=normal@E[:3,:3].T
                intensity=.25+.65*np.abs(ncam@light)
                gray=np.repeat(intensity[...,None],3,-1);gray[~hit]=0
                normal_color=(ncam+1)/2;normal_color[~hit]=0;rgb[~hit]=0
                # Primitive ID transitions display actual visible triangle edges at pixel resolution.
                h,w=hit.shape;C=np.linalg.inv(E);ids=np.full((h,w),np.iinfo(np.uint32).max,dtype=np.uint32)
                for start in range(0,h,192):
                    yy,xx=np.mgrid[start:min(start+192,h),:w]
                    dirs=np.stack([(xx-c["Cx"])/c["Fx"],(yy-c["Cy"])/c["Fy"],np.ones_like(xx)],-1)@C[:3,:3].T
                    rays=np.concatenate([np.broadcast_to(C[:3,3],dirs.shape),dirs],-1).astype(np.float32)
                    ids[start:start+len(yy)]=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8)["primitive_ids"].numpy()
                edges=np.zeros_like(hit)
                edges[:,1:] |= ids[:,1:]!=ids[:,:-1]
                edges[1:,:] |= ids[1:,:]!=ids[:-1,:]
                wire=gray.copy();wire[edges&hit]=[1,1,1]
                with np.load(index[sid]["path"]) as data:fg=data["foreground_mask"];gt=data["gt_rgb"]
                missing=fg&~hit
                internal=missing&(ndimage.distance_transform_edt(fg)>5)
                holes=gray.copy();holes[internal]=[1,0,1]
                image(out/"gray.png",gray);image(out/"normal.png",normal_color);image(out/"vertex_color.png",rgb)
                image(out/"wireframe.png",wire);image(out/"projected_interior_missing.png",holes)
                fig,axes=plt.subplots(1,6,figsize=(18,6))
                for ax,array,title in zip(axes,[gt,gray,normal_color,rgb,wire,holes],
                    ["GT","Gray surface","Camera normal","Vertex color","Visible triangles","Interior missing >5px"]):
                    ax.imshow(np.clip(array,0,1));ax.set_title(title);ax.axis("off")
                fig.suptitle(f"P3 {route} {stage}, source {sid}")
                fig.tight_layout();fig.savefig(out/"comparison.png",dpi=100);plt.close(fig)
    write(ROOT/"visualizations/protocol.json",dict(source_ids=REPS,camera="Exact E10 full1080x1920 corrected K and W2C",
      display="Depth panel fixed .15–1.57 E10 world units; mesh same full-source views; no geometry rotation/rescale",
      normal="Camera coordinates",wireframe="Native visible triangle primitive ID transitions at pixel resolution",
      texture="TSDF vertex RGB, no UV texture OBJ is claimed",hole_overlay="Missing rays inside RAP with >5px distance to RAP boundary; proxy missing area, not confirmed3D holes"))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stereo-only",action="store_true");a=parser.parse_args()
    stereo_panels()
    if not a.stereo_only:mesh_panels()
if __name__=="__main__":
    main()
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

