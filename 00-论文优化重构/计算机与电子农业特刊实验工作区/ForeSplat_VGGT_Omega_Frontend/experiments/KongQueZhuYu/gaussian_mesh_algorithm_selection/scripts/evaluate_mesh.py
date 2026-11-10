"""Common208 mesh evaluator + enclosed missing-area proxy and normal roughness."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MPLBACKEND='Agg')
import sys,argparse,json,time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy import ndimage
from PIL import Image
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection'
sys.path.insert(0,str(OLD/'scripts'))
from evaluate_meshes import structure,cast,metrics,csvwrite
from surface_protocol import save_npz_fast,write
p=argparse.ArgumentParser();p.add_argument('--mesh-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--stages',nargs='+',default=['raw','post']);a=p.parse_args()
out=a.mesh_dir/'unified_evaluation';out.mkdir(parents=True,exist_ok=True)
if (out/'complete.json').exists():raise SystemExit(0)
scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text());diag=scale['diagonal']
index=json.loads((a.input/'render_index.json').read_text());assert len(index)==208
def roughness(v,f):
    n=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);length=np.linalg.norm(n,axis=1);n/=np.maximum(length[:,None],1e-12)
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);fid=np.tile(np.arange(len(f)),3);e=np.sort(edges,1);codes=e[:,0].astype(np.int64)*len(v)+e[:,1];order=np.argsort(codes);cc=codes[order];starts=np.flatnonzero(np.r_[True,cc[1:]!=cc[:-1]]);ends=np.r_[starts[1:],len(cc)];two=ends-starts==2;left=fid[order[starts[two]]];right=fid[order[starts[two]+1]];cos=np.clip(np.einsum('ij,ij->i',n[left],n[right]),-1,1);angle=np.degrees(np.arccos(cos))
    return dict(surface_roughness=float(np.mean(1-cos)),neighbor_normal_angle_median_deg=float(np.median(angle)),neighbor_normal_angle_p90_deg=float(np.quantile(angle,.9)),definition='mean1-cos normal across manifold interior edges; curvature as well as reconstruction roughness; not GT error')
def enclosed_missing(pred,fg):
    miss=fg&(~pred);label,count=ndimage.label(miss)
    rim=fg&(~ndimage.binary_erosion(fg,iterations=3));touch=set(np.unique(label[rim]));touch.add(0)
    sizes=np.bincount(label.ravel());valid=[i for i in range(1,count+1) if i not in touch and sizes[i]>=9]
    holes=np.isin(label,valid)
    return dict(hole_count=len(valid),hole_area_px=int(holes.sum()),hole_area_fg_ratio=float(holes.sum()/max(fg.sum(),1)),open_edge_missing_area_px=int(miss.sum()-holes.sum())),holes
start=time.time();allrows=[];struct=[];reps=[0,26,53,79,105,131,157,183]
for stage in a.stages:
    path=a.mesh_dir/f'{stage}.ply'
    if not path.exists():
        native={'raw':'recon_init.ply','binary':'recon.ply','post':'recon_post.ply'};path=a.mesh_dir/native.get(stage,stage+'.ply')
    mesh=o3d.io.read_triangle_mesh(str(path));assert len(mesh.vertices) and len(mesh.triangles)
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);col=np.asarray(mesh.vertex_colors)
    s=structure(mesh,a.method,stage,diag);s.update(roughness(v,f));struct.append(s);csvwrite(out/'structure.csv',struct)
    scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));rows=[]
    for item in index:
        c=item['camera'];sid=item['source_id']
        with np.load(item['path']) as d:fg=d['foreground_mask'];reference=d['depth_surface']
        depth,normal,rgb=cast(scene,c,v,f,col);pred=np.isfinite(depth)&(depth>0)
        hole,holes=enclosed_missing(pred,fg)
        row=dict(method=a.method,stage=stage,source_id=sid,reference='RAP predicted mask',**metrics(pred,fg,depth,reference,diag),**hole);rows.append(row);allrows.append(row)
        saved=dict(mesh_depth=np.where(pred,depth,0),mesh_silhouette=pred,internal_missing_proxy=holes)
        if sid in reps:
            saved.update(mesh_normal_world=normal,mesh_rgb=rgb,foreground_mask=fg,reference_depth=reference)
            d=out/'visualizations'/f'{stage}_{sid:04d}';d.mkdir(parents=True,exist_ok=True)
            E=np.asarray(c['world_view_transform']).T;nc=normal@E[:3,:3].T
            shade=.2+.8*np.abs(nc[:,:,2]);gray=np.repeat(shade[...,None],3,2);gray[~pred]=0
            normalimg=(normal*.5+.5);normalimg[~pred]=0
            for name,img in [('gray',gray),('normal',normalimg),('vertex_color',rgb)]:
                Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(d/f'{name}.png')
            Image.fromarray((holes*255).astype(np.uint8)).save(d/'internal_missing_proxy.png')
        dest=out/'views'/f'{stage}_{sid:04d}.npz';dest.parent.mkdir(exist_ok=True);save_npz_fast(dest,**saved)
        print(a.method,stage,sid,row['silhouette_iou'],row['hole_area_fg_ratio'],flush=True)
    csvwrite(out/f'{stage}_per_view.csv',rows)
    del scene,mesh,v,f,col
csvwrite(out/'all_references_per_view.csv',allrows)
write(out/'complete.json',dict(status='complete',method=a.method,views_per_stage=208,stages=a.stages,elapsed_seconds=time.time()-start,scale=scale,hole_definition='Enclosed connected foreground missing-projection components >=9pixels not touching3px RAP exterior boundary; GT holes excluded. Proxy, not a 3D physical hole-area certificate',roughness_definition='mean1-cos adjacent face normal across manifold edges, includes real leaf curvature',camera_and_metrics='same historical208 integer-centre raycast and boundary scores'))
sys.stdout.flush();os._exit(0)

