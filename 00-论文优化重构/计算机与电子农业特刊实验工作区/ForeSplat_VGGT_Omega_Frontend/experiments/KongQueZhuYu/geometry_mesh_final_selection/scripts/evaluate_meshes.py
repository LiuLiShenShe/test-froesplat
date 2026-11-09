"""Native-resolution, integer-centre mesh reprojection and structure audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import open3d as o3d
from surface_protocol import save_npz_fast
from PIL import Image
from scipy import ndimage
from surface_protocol import ROOT,DUAL,write

MANUAL=Path('/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split/KongQueZhuYu')
REPS=[0,26,53,79,105,131,157,183]
METHOD_INPUT={'M1_Omega_NKSR':'omega_direct','M2_Standard_NKSR':'standard_direct','M3_E10_GGGS_NKSR':'e10_gggs_surface',
    'M4_E10_GGGS_TSDF':'e10_gggs_surface','M5_E11_2DGS_TSDF':'e11_2dgs_surface','E8_GGGS_NKSR':'e8_gggs_surface','E11_2DGS_NKSR':'e11_2dgs_surface'}

def csvwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def structure(mesh,method,stage,diag):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);a=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);direction=np.where(edges[:,0]<edges[:,1],1.,-1.);keys=np.sort(edges,axis=1)
    code=keys[:,0].astype(np.int64)*len(v)+keys[:,1];_,inv,c=np.unique(code,return_inverse=True,return_counts=True);signsum=np.bincount(inv,weights=direction)
    labels,counts,areas=mesh.cluster_connected_triangles();counts=np.asarray(counts)
    out=dict(method=method,stage=stage,vertices=len(v),faces=len(f),components=len(counts),largest_component_face_fraction=float(counts.max()/len(f)),
        boundary_edges=int((c==1).sum()),nonmanifold_edges=int((c>2).sum()),degenerate_face_fraction=float((a<diag**2*1e-12).mean()),surface_area=float(a.sum()),
        bbox_min=json.dumps(v.min(0).tolist()),bbox_max=json.dumps(v.max(0).tolist()),normal_winding_consistency=float((signsum[c==2]==0).mean()),scale_diagonal=diag)
    return out

def boundary_scores(pred,gt):
    p=pred^ndimage.binary_erosion(pred);g=gt^ndimage.binary_erosion(gt)
    if not p.any() or not g.any():return dict(boundary_precision_2px=0.,boundary_recall_2px=0.,boundary_fscore_2px=0.,boundary_chamfer_px=float('nan'))
    dp=ndimage.distance_transform_edt(~p);dg=ndimage.distance_transform_edt(~g)
    precision=float((dg[p]<=2).mean());recall=float((dp[g]<=2).mean())
    return dict(boundary_precision_2px=precision,boundary_recall_2px=recall,boundary_fscore_2px=2*precision*recall/max(precision+recall,1e-12),boundary_chamfer_px=float((dg[p].mean()+dp[g].mean())/2))

def cast(scene,c,vertices,faces,colours):
    h,w=c['height'],c['width'];c2w=np.linalg.inv(np.asarray(c['world_view_transform']).T);origin=c2w[:3,3].astype(np.float32)
    depth=np.full((h,w),np.inf,dtype=np.float32);normal=np.zeros((h,w,3),dtype=np.float32);rgb=np.zeros_like(normal)
    for start in range(0,h,192):
        yy,xx=np.mgrid[start:min(start+192,h),:w]
        direction=np.stack([(xx-c['Cx'])/c['Fx'],(yy-c['Cy'])/c['Fy'],np.ones_like(xx)],-1)@c2w[:3,:3].T
        rays=np.concatenate([np.broadcast_to(origin,direction.shape),direction],axis=-1).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8);z=hit['t_hit'].numpy();ids=hit['primitive_ids'].numpy();uv=hit['primitive_uvs'].numpy();ok=np.isfinite(z)&(z>0)
        depth[start:start+len(yy)]=z;normal[start:start+len(yy)]=hit['primitive_normals'].numpy()
        if len(colours)==len(vertices):
            triangles=faces[ids[ok]];b=uv[ok];colour=colours[triangles[:,0]]*(1-b.sum(1))[:,None]+colours[triangles[:,1]]*b[:,0,None]+colours[triangles[:,2]]*b[:,1,None]
            block=rgb[start:start+len(yy)];block[ok]=colour
    return depth,normal,rgb

def metrics(pred,fg,depth,reference,diag):
    union=(pred|fg).sum();common=pred&fg
    result=dict(silhouette_iou=float(common.sum()/max(union,1)),foreground_coverage=float(common.sum()/max(fg.sum(),1)),foreground_precision=float(common.sum()/max(pred.sum(),1)),background_geometry_fraction=float((pred&~fg).sum()/max((~fg).sum(),1)))
    result.update(boundary_scores(pred,fg))
    if reference is not None:
        good=common&np.isfinite(reference)&(reference>0)
        err=np.abs(depth[good]-reference[good]);result.update(depth_common_pixels=int(good.sum()),depth_mae=float(err.mean()) if len(err) else float('nan'),depth_p95=float(np.quantile(err,.95)) if len(err) else float('nan'),depth_mae_normalized=float(err.mean()/diag) if len(err) else float('nan'))
    else:result.update(depth_common_pixels=0,depth_mae=float('nan'),depth_p95=float('nan'),depth_mae_normalized=float('nan'))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh-dir',type=Path,required=True);p.add_argument('--method',required=True);a=p.parse_args()
    out=a.mesh_dir/'evaluation';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    method=a.method;primary=method.split('/')[0];inputname=METHOD_INPUT[primary];inp=ROOT/'surface_inputs'/inputname;info=json.loads((a.mesh_dir/'complete.json').read_text());diag=info['scale']['diagonal']
    cameras=json.loads((inp/'camera_runtime.json').read_text())['records'];manifest={int(r['source_index']):r for r in csv.DictReader((DUAL/'shared_manifest.csv').open())};structures=[];allrows=[];start=time.time()
    indices={r['source_id']:r for r in json.loads((inp/'render_index.json').read_text())} if (inp/'render_index.json').exists() else None
    removed=np.load(a.mesh_dir/'components_and_removals.npz')['removed_vertex_positions'];removed_rows=[]
    for stage in ['raw','post']:
        mesh=o3d.io.read_triangle_mesh(str(a.mesh_dir/f'{stage}.ply'));assert len(mesh.vertices) and len(mesh.triangles)
        structures.append(structure(mesh,method,stage,diag));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);colours=np.asarray(mesh.vertex_colors)
        scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));rows=[]
        for c in cameras:
            sid=c.get('source_id',int(c['name'].split('_')[-1]));fg=np.asarray(Image.open(manifest[sid]['mask_path']).convert('L').resize((c['width'],c['height']),Image.Resampling.NEAREST))>127
            depth,normal,rgb=cast(scene,c,v,f,colours);pred=np.isfinite(depth)&(depth>0)
            if indices:
                with np.load(indices[sid]['path']) as d:reference=d['depth_surface'].copy()
            else:
                # Direct map resolution differs: its own exact model pixel transform maps depth to shared coordinates.
                front=DUAL/('VGGT_Omega_Original_RGB' if inputname=='omega_direct' else 'Standard_VGGT_Original_RGB');fc=next(r for r in json.loads((front/'cameras.json').read_text())['frames'] if r['source_index']==sid)
                modeldepth=np.load(front/'per_frame'/f'depth_{c["name"]}.npy');A=np.asarray(fc['pixel_transform_pixel_center']);yy,xx=np.mgrid[:c['height'],:c['width']]
                source=np.stack([xx*2+.5,yy*2+.5,np.ones_like(xx)],-1)@A.T;reference=ndimage.map_coordinates(modeldepth,[source[...,1],source[...,0]],order=1,mode='constant',cval=0)
            row=dict(method=method,stage=stage,source_id=sid,reference='RAP predicted mask',**metrics(pred,fg,depth,reference,diag));rows.append(row);allrows.append(row)
            if sid in [0,25,50,75,100]:
                for scope in ['plant_only','plant_plus_pot']:
                    path=MANUAL/f'mask_{scope}_{sid:04d}.png'
                    if path.exists():
                        manual=np.asarray(Image.open(path).convert('L').resize((c['width'],c['height']),Image.Resampling.NEAREST))>127
                        allrows.append(dict(method=method,stage=stage,source_id=sid,reference=f'manual polygon, historical inferred scope {scope}',**metrics(pred,manual,depth,reference,diag)))
            dest=out/'views'/f'{stage}_{sid:04d}.npz';dest.parent.mkdir(exist_ok=True)
            saved=dict(mesh_depth=np.where(pred,depth,0),mesh_silhouette=pred)
            if sid in REPS:saved.update(mesh_normal_world=normal,mesh_rgb=rgb,foreground_mask=fg,reference_depth=reference)
            save_npz_fast(dest,**saved)
            if stage=='raw' and len(removed):
                E=np.asarray(c['world_view_transform']).T;pc=removed@E[:3,:3].T+E[:3,3];uv=pc[:,:2]/pc[:,2,None]*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']];ix=np.rint(uv).astype(np.int64)
                valid=(pc[:,2]>0)&(ix[:,0]>=0)&(ix[:,0]<c['width'])&(ix[:,1]>=0)&(ix[:,1]<c['height'])
                removed_rows.append(dict(source_id=sid,deleted_vertices_in_image=int(valid.sum()),deleted_vertices_project_into_RAP=int(fg[ix[valid,1],ix[valid,0]].sum()),
                    note='Projection overlap only; visibility/true plant identity is not inferred from overlap'))
            print(method,stage,sid,'IoU',row['silhouette_iou'],flush=True)
        csvwrite(out/f'{stage}_per_view.csv',rows)
    csvwrite(out/'structure.csv',structures);csvwrite(out/'all_references_per_view.csv',allrows);csvwrite(out/'deleted_projection.csv',removed_rows)
    write(out/'complete.json',dict(status='complete',method=method,views_per_stage=208,elapsed_seconds=time.time()-start,integer_pixel_centres=True,
        rays='Unnormalized C2W([ (u-cx)/fx, (v-cy)/fy, 1 ]); t_hit equals camera-axis depth. Corresponding model camera gauge only',
        scan='No confirmed same-specimen, same-acquisition scan was found; no 3D scan distances invented',manual_reference=str(MANUAL)))

if __name__=='__main__':
    import fcntl,sys
    target=Path(sys.argv[sys.argv.index('--mesh-dir')+1])/'evaluation'
    target.mkdir(parents=True,exist_ok=True)
    with (target/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
