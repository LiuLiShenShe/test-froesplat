"""CPU raycast visualization of actual Mesh faces/UV textures; true triangle-edge Wireframe, no image-edge proxy."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MPLBACKEND='Agg')
import sys,json,time,argparse,hashlib,traceback
from pathlib import Path
import numpy as np,open3d as o3d
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];HIST=ROOT.parent.parent/'geometry_mesh_final_selection'
REPS=[0,26,53,79,105,131,157,183]
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def png(path,image):Image.fromarray((np.clip(image,0,1)*255).round().astype(np.uint8)).save(path)
def read_cameras(source):
    if source.is_dir():source=source/'render_index.json'
    data=json.loads(source.read_text())
    rows=data.get('records',data.get('views')) if isinstance(data,dict) else data
    cameras={}
    for item in rows:
        c=item.get('camera',item);sid=item.get('source_id',c.get('source_id'))
        if sid is None:sid=int(c.get('name',c.get('image_name','')).split('_')[-1])
        cameras[int(sid)]=c
    assert all(sid in cameras for sid in REPS),('Missing fixed cameras',list(cameras))
    return [(f'fixed_{sid:04d}',cameras[sid]) for sid in REPS],source
def texture_samples(image,uv):
    a=np.asarray(image)
    if a.ndim==2:a=np.repeat(a[...,None],3,axis=2)
    a=a[...,:3].astype(np.float32,copy=False)/(255. if np.issubdtype(a.dtype,np.integer) else 1.)
    h,w=a.shape[:2];u=np.clip(uv[:,0],0,1)*(w-1);v=(1-np.clip(uv[:,1],0,1))*(h-1)
    x0=np.floor(u).astype(np.int64);y0=np.floor(v).astype(np.int64)
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    dx=(u-x0)[:,None];dy=(v-y0)[:,None]
    return (1-dy)*((1-dx)*a[y0,x0]+dx*a[y0,x1])+dy*((1-dx)*a[y1,x0]+dx*a[y1,x1])
def face_edge_distances(pixel,triangle_camera,c,near=1e-5):
    # Clip each actual3D face edge to positive camera-Z, then measure distance to its projected SEGMENT.
    # Only frontmost ray-hit faces enter this function, giving hidden-edge suppression.
    distance=np.full(len(pixel),np.inf,dtype=np.float32)
    for ia,ib in [(0,1),(1,2),(2,0)]:
        a=triangle_camera[:,ia].copy();b=triangle_camera[:,ib].copy()
        valid=(a[:,2]>=near)|(b[:,2]>=near)
        change=(a[:,2]<near)&valid
        t=(near-a[change,2])/(b[change,2]-a[change,2]);a[change]+=t[:,None]*(b[change]-a[change])
        change=(b[:,2]<near)&valid
        t=(near-b[change,2])/(a[change,2]-b[change,2]);b[change]+=t[:,None]*(a[change]-b[change])
        auv=a[:,:2]/np.maximum(a[:,2,None],near)*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']]
        buv=b[:,:2]/np.maximum(b[:,2,None],near)*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']]
        edge=buv-auv;d=pixel-auv
        fraction=np.clip(np.einsum('ij,ij->i',d,edge)/np.maximum(np.einsum('ij,ij->i',edge,edge),1e-20),0,1)
        delta=d-fraction[:,None]*edge
        value=np.sqrt(np.einsum('ij,ij->i',delta,delta));value[~valid]=np.inf
        distance=np.minimum(distance,value)
    return distance
def actual_raycast(mesh,c,wire_width=.65,threads=8,scene=None):
    h,w=int(c['height']),int(c['width']);E=np.asarray(c['world_view_transform'],dtype=np.float64).T
    C=np.linalg.inv(E);origin=C[:3,3].astype(np.float32)
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
    camera_vertices=vertices@E[:3,:3].T+E[:3,3]
    has_colors=len(colors)==len(vertices) and len(vertices)>0
    uv=np.asarray(mesh.triangle_uvs);materials=np.asarray(mesh.triangle_material_ids);textures=mesh.textures
    has_texture=len(uv)==3*len(faces) and len(textures)>0
    if has_texture:
        uv=uv.reshape(-1,3,2)
        if len(materials)!=len(faces):materials=np.zeros(len(faces),dtype=np.int64)
        assert materials.min()>=0 and materials.max()<len(textures),('Invalid OBJ material texture index',materials.min(),materials.max(),len(textures))
        texture_arrays={}
        # OBJ loaders may include an empty, unreferenced default material at texture index0.
        # Read only material IDs actually referenced by faces; never substitute a missing referenced image.
        for material in np.unique(materials):
            a=np.asarray(textures[int(material)])
            assert a.ndim in [2,3] and a.size>0,('Referenced OBJ texture has no actual image',int(material))
            if a.ndim==2:a=np.repeat(a[...,None],3,axis=2)
            texture_arrays[int(material)]=a[...,:3].astype(np.float32)/(255. if np.issubdtype(a.dtype,np.integer) else 1.)
    if scene is None:
        scene=o3d.t.geometry.RaycastingScene(nthreads=threads)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    depth=np.zeros((h,w),dtype=np.float32);normal=np.zeros((h,w,3),dtype=np.float32)
    silhouette=np.zeros((h,w),dtype=bool);wire=np.zeros((h,w),dtype=bool)
    color=np.zeros((h,w,3),dtype=np.float32) if has_colors else None
    texture=np.zeros((h,w,3),dtype=np.float32) if has_texture else None
    for start in range(0,h,128):
        yy,xx=np.mgrid[start:min(h,start+128),:w]
        direction=np.stack([(xx-c['Cx'])/c['Fx'],(yy-c['Cy'])/c['Fy'],np.ones_like(xx)],-1)@C[:3,:3].T
        rays=np.concatenate([np.broadcast_to(origin,direction.shape),direction],-1).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=threads)
        z=hit['t_hit'].numpy();fid=hit['primitive_ids'].numpy();b=hit['primitive_uvs'].numpy()
        valid=np.isfinite(z)&(z>0);sl=slice(start,start+len(yy))
        silhouette[sl]=valid;depth[sl]=np.where(valid,z,0);normal[sl]=hit['primitive_normals'].numpy()
        if not valid.any():continue
        ids=fid[valid].astype(np.int64);bary=np.stack([1-b[valid].sum(1),b[valid,0],b[valid,1]],axis=1)
        pixel=np.stack([xx[valid],yy[valid]],axis=1)
        wire[sl][valid]=face_edge_distances(pixel,camera_vertices[faces[ids]],c)<=wire_width
        if has_colors:color[sl][valid]=(colors[faces[ids]]*bary[:,:,None]).sum(1)
        if has_texture:
            sampled_uv=(uv[ids]*bary[:,:,None]).sum(1);mat=materials[ids];block=texture[sl]
            sampled=np.zeros((len(ids),3),dtype=np.float32)
            for material in np.unique(mat):
                selected=mat==material;sampled[selected]=texture_samples(texture_arrays[int(material)],sampled_uv[selected])
            block[valid]=sampled
    normal[~silhouette]=0
    return dict(depth=depth,normal_world=normal,silhouette=silhouette,wireframe=wire,vertex_color=color,texture=texture,has_vertex_color=has_colors,has_native_uv_texture=has_texture)
def images_from_channels(channels,c,normal_space='world',background=.25):
    pred=channels['silhouette'];world=channels['normal_world'];E=np.asarray(c['world_view_transform']).T
    camera=world@E[:3,:3].T
    gray=np.repeat((.2+.8*np.abs(camera[...,2]))[...,None],3,axis=2);gray[~pred]=background
    world_image=world*.5+.5;world_image[~pred]=background
    camera_image=camera*.5+.5;camera_image[~pred]=background
    wire=gray.copy();wire[channels['wireframe']]=0
    images=dict(gray=gray,normal=world_image if normal_space=='world' else camera_image,normal_world=world_image,normal_camera=camera_image,wireframe=wire,wireframe_mask=channels['wireframe'])
    for key,name in [('vertex_color','vertex_color'),('texture','textured')]:
        if channels[key] is not None:
            image=channels[key].copy();image[~pred]=background;images[name]=image
    return images
def render_mesh(path,out,cameras,normal_space,background,wire_width,threads):
    path=path.resolve(strict=True)
    out.mkdir(parents=True,exist_ok=True);marker=out/'complete.json'
    if marker.exists() and json.loads(marker.read_text())['status']=='complete':
        old=json.loads(marker.read_text())
        assert old['source_mesh']==str(path) and old['normal_space']==normal_space and old['background']==background and old['wireframe_width_px']==wire_width,'Choose a different --out for changed visualization settings; existing outputs remain preserved'
        return old
    mesh=o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.vertices) or not len(mesh.triangles):raise ValueError('Actual Mesh has no triangle surface')
    started=time.time();records=[]
    # A large15M-face Mesh builds its BVH once; all cameras use identical first-hit geometry.
    scene=o3d.t.geometry.RaycastingScene(nthreads=threads)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for name,c in cameras:
        folder=out/name;folder.mkdir(exist_ok=True)
        data=actual_raycast(mesh,c,wire_width,threads,scene=scene)
        if path.suffix.lower()=='.obj' and not data['has_native_uv_texture']:raise ValueError('Source OBJ UV/material texture failed to load; actual textured view unavailable, not replaced by vertex color/gray')
        images=images_from_channels(data,c,normal_space,background)
        for label,image in images.items():png(folder/(label+'.png'),image)
        arrays={key:value for key,value in data.items() if isinstance(value,np.ndarray)}
        np.savez_compressed(folder/'geometry_channels.npz',**arrays)
        write(folder/'camera_and_geometry.json',dict(source_mesh=str(path),camera=c,normal_space=normal_space,world_normal='Actual Open3D unit frontmost triangle normal in source mesh world coordinates',camera_normal='world normal @ W2C_rotation.T',gray='Same two-sided headlight shading0.2+0.8*abs(camera_normal.z), independent of all colors/textures',wireframe='Exact actual triangle vertex/face edge segments, clipped in camera Z and projected with corrected K; retain pixels on frontmost ray-hit face within the recorded pixel width of its actual edge; not RGB/normal/silhouette edge detection',wireframe_width_px=wire_width,vertex_color_available=data['has_vertex_color'],native_uv_texture_available=data['has_native_uv_texture'],color_semantics='Vertex color only if actually present; UV texture only if actual source OBJ has per-corner UV and decoded texture material image. Missing color/texture is unavailable, never replaced by gray.',background=background))
        records.append(dict(view=name,directory=str(folder.resolve()),vertex_color_available=data['has_vertex_color'],native_uv_texture_available=data['has_native_uv_texture']))
        print('actual Mesh geometry visual',path.name,name,flush=True)
    result=dict(status='complete',source_mesh=str(path),vertices=len(mesh.vertices),faces=len(mesh.triangles),output=str(out.resolve()),actual_views=records,elapsed_seconds=time.time()-started,normal_space=normal_space,BVH_builds=1,scene_shared_across_views=True,gray_headlight=[.2,.8],wireframe_width_px=wire_width,background=background,GPU=False)
    write(marker,result)
    del scene,mesh,data,images
    __import__('gc').collect()
    return result
def e14_mesh_list():
    entries=[]
    for phase,folder in [('cluster_before',ROOT/'cluster_pruning/cluster_before_mesh'),('cluster_after',ROOT/'cluster_pruning/cluster_after_mesh'),('initial',ROOT/'initial_mesh'),('refined',ROOT/'refinement/mesh_post')]:
        for version in ['raw','post']:entries.append((phase+'_'+version,folder/(version+'.ply')))
    entries.append(('initial_native_largest_cluster',ROOT/'initial_mesh/native_largest_cluster.ply'))
    marker=ROOT/'refinement_complete.json'
    if marker.exists():
        info=json.loads(marker.read_text())
        if info.get('status')=='complete':entries.append(('refined_native_textured_OBJ',Path(info['textured_obj'])))
        else:entries.append(('refined_native_textured_OBJ',None))
    else:entries.append(('refined_native_textured_OBJ',None))
    return entries
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mesh',type=Path,default=None,help='Actual Mesh PLY/OBJ; omit for all E14 stages')
    parser.add_argument('--input',type=Path,default=HIST/'surface_inputs/e10_gggs_surface',help='208-view render_index.json or camera_runtime.json, or its parent directory')
    parser.add_argument('--out',type=Path,default=ROOT/'visual_views/mesh')
    parser.add_argument('--normal',choices=['world','camera'],default='world')
    parser.add_argument('--background',type=float,default=.25)
    parser.add_argument('--wire-width-px',type=float,default=.65)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--protocol',type=Path,default=None,help='Optional explicit shared inspection/free cameras; otherwise use fixed eight from --input')
    parser.add_argument('--check-inputs-only',action='store_true')
    args=parser.parse_args();cameras,input_file=read_cameras(args.input)
    if args.protocol is not None:
        protocol=json.loads(args.protocol.read_text());assert len(protocol['views'])>0
        cameras=[(r['name'],r) for r in protocol['views']];input_file=args.protocol
    entries=[(args.mesh.stem,args.mesh)] if args.mesh else e14_mesh_list()
    ready=dict(status='ready',mesh_manifest=[dict(stage=tag,actual_path=str(path.resolve()) if path and path.is_file() else None,status='exists' if path and path.is_file() else 'awaiting actual output') for tag,path in entries],input=str(input_file.resolve()),input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),fixed_views=REPS if args.protocol is None else [name for name,c in cameras],normal=args.normal,gray_headlight=[.2,.8],wireframe_width_px=args.wire_width_px,background=args.background,GPU=False)
    if args.check_inputs_only:
        write(ROOT/('mesh_visuals_ready.json' if args.protocol is None else 'mesh_visuals_'+args.protocol.stem+'_ready.json'),ready);print(json.dumps(ready,ensure_ascii=False,indent=2));return
    args.out.mkdir(parents=True,exist_ok=True);results=[];started=time.time()
    for tag,path in entries:
        target=args.out/tag if args.mesh is None else args.out
        try:
            if path is None or not path.is_file():raise FileNotFoundError('No actual Mesh output exists for this stage')
            result=render_mesh(path,target,cameras,args.normal,args.background,args.wire_width_px,args.threads)
            result['stage']=tag
        except Exception as exc:
            result=dict(status='unavailable',stage=tag,reason=str(exc),traceback=traceback.format_exc(),actual_path=str(path.resolve()) if path and path.is_file() else None)
            print('Mesh visual unavailable',tag,str(exc),flush=True)
        results.append(result)
    done=dict(status='complete' if all(r['status']=='complete' for r in results) else 'partial',methods=results,input=str(input_file.resolve()),fixed_views=REPS if args.protocol is None else [name for name,c in cameras],elapsed_seconds=time.time()-started,protocol=ready)
    write(args.out/'mesh_visuals_manifest.json',done)
    if args.mesh is None:write(ROOT/'mesh_visuals_complete.json',done)
    if done['status']!='complete':raise SystemExit(2)
if __name__=='__main__':
    import fcntl
    output=Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else ROOT/'visual_views/mesh'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        main()

