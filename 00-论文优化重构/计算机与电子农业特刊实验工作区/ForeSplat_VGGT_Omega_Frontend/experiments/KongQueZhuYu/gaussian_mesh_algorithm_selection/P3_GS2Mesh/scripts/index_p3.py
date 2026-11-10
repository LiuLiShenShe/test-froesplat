"""Index only actually existing P3 outputs, with resolved paths and real sizes."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[3]

def main():
    rows=[]
    def add(path,kind,stage,number="",faces=""):
        p=Path(path)
        if not p.is_file():return
        real=p.resolve()
        rows.append(dict(experiment_id="P3",method="E10 GGGS -> GS2Mesh DLNR",artifact_type=kind,stage=stage,
          format=p.suffix.lstrip("."),absolute_path=str(real),relative_path=str(real.relative_to(BASE)),
          file_size_bytes=real.stat().st_size,vertices_or_gaussians=number,faces=faces,
          coordinate_system="Original E10 standard VGGT world gauge; scalar depths in camera-axis units",
          source_checkpoint=str((BASE/"experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply").resolve()),
          viewer="Blender / MeshLab / Open3D" if kind=="Mesh PLY" else "NumPy / image viewer",status="exists"))
    for route in ["meshes","meshes_shared_depth_range"]:
        directory=ROOT/route;complete=directory/"complete.json"
        if complete.exists():
            stats=json.loads(complete.read_text())
            for stage in ["raw","post"]:
                add(directory/(stage+".ply"),"Mesh PLY",route+"/"+stage,stats[stage+"_vertices"],stats[stage+"_faces"])
    for folder in sorted((ROOT/"stereo").glob("[0-9][0-9][0-9]")):
        sid=json.loads((folder/"render_complete.json").read_text())["source_id"] if (folder/"render_complete.json").exists() else None
        if sid is None:continue
        for n in ["left.png","right.png"]:add(folder/n,"Stereo RGB","source_"+str(sid)+"/"+n.removesuffix(".png"))
        add(folder/"gggs_aux.npz","Depth Map","source_"+str(sid)+"/GGGS auxiliary depth and alpha")
        model=folder/"out_DLNR_Middlebury"
        for n,kind in [("depth_raw.npy","Depth Map"),("depth.npy","Depth Map"),("confidence.npy","Stereo Confidence"),
                       ("occlusion_mask.npy","Stereo Consistency"),("disparity_LR.npy","Stereo Disparity"),("disparity_RL.npy","Stereo Disparity")]:
            add(model/n,kind,"source_"+str(sid)+"/"+n.removesuffix(".npy"))
    if rows:
        with (ROOT/"ARTIFACTS_P3.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("P3 real artifact index",len(rows),"rows")

if __name__=="__main__":main()

