"""Measure real near-front depth layers at immutable RGB correspondences.
Counts are morphology proxies, not certified erroneous double leaves.
"""
import csv,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import open3d as o3d
OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/same_ray_mesh_layers'

def read(path):return list(csv.DictReader(path.open()))
def writecsv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def summarize(counts,near,pairs,choose):
    hit=counts[choose]>0;n=int(choose.sum());nh=int(hit.sum())
    return dict(queries=n,mesh_hit_queries=nh,mesh_hit_fraction=nh/max(n,1),
        positive_distinct_layers_mean=float(counts[choose].mean()),
        near_front_4vox_layers_mean=float(near[choose].mean()),
        near_front_ge3_fraction_all_queries=float((near[choose]>=3).mean()),
        near_front_ge3_fraction_hit_queries=float((near[choose][hit]>=3).mean()) if nh else '',
        close_same_facing_pair_fraction_all_queries=float((pairs[choose]>0).mean()),
        close_same_facing_pair_fraction_hit_queries=float((pairs[choose][hit]>0).mean()) if nh else '')

def api_calibration(voxel):
    vertices=[];triangles=[]
    for k,z in enumerate([1.,1.+voxel,1.+2*voxel]):
        vertices.extend([[-2,-2,z],[2,-2,z],[2,2,z],[-2,2,z]])
        offset=4*k
        triangles.extend([[offset,offset+2,offset+1],[offset,offset+3,offset+2]] if k!=2 else [[offset,offset+1,offset+2],[offset,offset+2,offset+3]])
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,np.float32)),o3d.core.Tensor(np.asarray(triangles,np.uint32)))
    ray=o3d.core.Tensor(np.asarray([[0,0,0,.11,.07,1]],np.float32))
    hits=scene.list_intersections(ray,nthreads=4);z=np.sort(hits['t_hit'].numpy())
    assert len(z)==3 and np.max(np.abs(z-[1,1+voxel,1+2*voxel]))<1e-6
    first=scene.cast_rays(ray,nthreads=4)['t_hit'].numpy()[0]
    assert abs(first-z[0])<1e-6
    return dict(status='passed_actual_multiple_layer_API',Open3D=o3d.__version__,nonunit_ray=[.11,.07,1],
        expected_camera_Z=[1,1+voxel,1+2*voxel],actual_t_hit=z.tolist(),
        output_dtypes={k:str(v.dtype) for k,v in hits.items()},
        scope='Actual CPU intersection API and non-unit ray camera-Z parameterization; calibration geometry is explicitly synthetic, not a plant result')

def main():
    DEST.mkdir(exist_ok=True)
    manifest=OUT/'comparison/reflection_analysis/RGB_QUERY_MANIFEST.csv'
    queries=read(manifest);ids=np.array([int(r['query_id']) for r in queries]);assert np.array_equal(ids,np.arange(len(queries)))
    source_ids=np.array([int(r['source_id']) for r in queries]);assert not ({44,206}&set(source_ids))
    raw={int(r['query_id']):r['raw_context_candidate'].lower()=='true' for r in read(OUT/'comparison/reflection_analysis/ORIGINAL_RGB_QUERY_VALIDATION.csv')}
    bright=np.array([raw[i] for i in ids]);assert bright.sum()==134
    cameras={int(r['source_id']):r for r in json.loads((OLD/'surface_inputs/e10_gggs_surface/camera_runtime.json').read_text())['records']}
    rays=np.empty((len(queries),6),np.float32)
    for sid in sorted(set(source_ids)):
        c=cameras[sid];C=np.linalg.inv(np.asarray(c['world_view_transform'],np.float64).T);choose=source_ids==sid
        uv=np.array([[float(queries[i]['x']),float(queries[i]['y'])] for i in np.flatnonzero(choose)])
        direction=np.c_[(uv[:,0]-c['Cx'])/c['Fx'],(uv[:,1]-c['Cy'])/c['Fy'],np.ones(len(uv))]@C[:3,:3].T
        rays[choose,:3]=C[:3,3];rays[choose,3:]=direction
    lengths=np.linalg.norm(rays[:,3:].astype(np.float64),axis=1)
    scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text());voxel=scale['tsdf_voxel'];merge=min(voxel*.001,scale['diagonal']*1e-6)
    protocol=dict(status='actual_measurement_protocol',query_manifest=str(manifest.resolve()),query_SHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        queries=len(queries),source_views=len(set(source_ids)),original_RGB_brightness_queries=int(bright.sum()),
        camera_source=str((OLD/'surface_inputs/e10_gggs_surface/camera_runtime.json').resolve()),
        ray_parameter='Non-unit world ray from exact inverse W2C and corrected Fx/Fy/Cx/Cy: t_hit is camera Z; world spacing = delta_t * norm(direction)',
        merge_world_epsilon=merge,near_front_window_world=4*voxel,close_pair_world_maximum=2*voxel,same_facing_normal_dot_minimum=.9,
        maximum_camera_Z=scale['tsdf_depth_trunc'],normal='Exact triangle winding normal, signed unit world vector; not interpolated normal color',
        scope='Supplementary fixed image-only leaf-region RGB matches from69 source cameras;195 source/target photographs, alongside separate full208 mesh evaluation. Query-weighted means; missing rays and denominators retained. Legitimate front/back leaf surfaces, curls, touching or occluded leaves can create several layers; counts do not certify erroneous double surfaces, physical thickness or scan accuracy.',
        calibration=api_calibration(voxel))
    (DEST/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summaries=[];perviews=[];routes=read(OUT/'comparison/ALL_MESH_EXPERIMENTS.csv');seen=set()
    for route in routes:
        if route['mesh_id'].startswith('M1_'):continue # Different unregistered Omega gauge.
        path=Path(route['evaluated_mesh_path']).resolve()
        if str(path) in seen:continue
        seen.add(str(path));alias=route['mesh_id']+'_'+route['mesh_stage']+'_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
        folder=DEST/alias;folder.mkdir(exist_ok=True);done=folder/'complete.json';initial=path.stat()
        expected=dict(mesh=str(path),mesh_size=initial.st_size,mesh_mtime_ns=initial.st_mtime_ns,query_SHA256=protocol['query_SHA256'])
        if done.is_file() and all(json.loads(done.read_text()).get(k)==v for k,v in expected.items()):
            records=read(folder/'summary.csv');views=read(folder/'per_view.csv')
        else:
            started=time.time();mesh=o3d.io.read_triangle_mesh(str(path));assert len(mesh.triangles)>0
            scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            tensor=o3d.core.Tensor(rays);result=scene.list_intersections(tensor,nthreads=4)
            arrays={k:v.numpy() for k,v in result.items()};splits=arrays['ray_splits'];t=arrays['t_hit'];face_ids=arrays['primitive_ids'].astype(np.int64)
            assert len(splits)==len(queries)+1 and splits[0]==0 and splits[-1]==len(t)
            vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)[face_ids]
            normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
            norm=np.linalg.norm(normals,axis=1);normals=np.divide(normals,norm[:,None],out=np.zeros_like(normals),where=norm[:,None]>1e-20)
            first=scene.cast_rays(tensor,nthreads=4)['t_hit'].numpy()
            counts=np.zeros(len(queries),np.int32);near=np.zeros_like(counts);pairs=np.zeros_like(counts);merged_first=np.full(len(queries),np.inf,np.float64)
            for i,(lo,hi) in enumerate(zip(splits[:-1],splits[1:])):
                lo,hi=int(lo),int(hi);assert np.all(arrays['ray_ids'][lo:hi]==i)
                valid=np.flatnonzero(np.isfinite(t[lo:hi])&(t[lo:hi]>0)&(t[lo:hi]<=scale['tsdf_depth_trunc']))+lo
                order=valid[np.argsort(t[valid],kind='stable')];keep=[]
                for hit in order:
                    if not keep or (float(t[hit])-float(t[keep[-1]]))*lengths[i]>merge:keep.append(int(hit))
                counts[i]=len(keep)
                if not keep:continue
                merged_first[i]=t[keep[0]]
                local=[k for k in keep if (float(t[k])-float(t[keep[0]]))*lengths[i]<=4*voxel];near[i]=len(local)
                for left,right in zip(local,local[1:]):
                    if (float(t[right])-float(t[left]))*lengths[i]<=2*voxel and np.dot(normals[left],normals[right])>.9:pairs[i]+=1
            within=np.isfinite(first)&(first>0)&(first<=scale['tsdf_depth_trunc'])
            assert np.array_equal(within,np.isfinite(merged_first))
            assert np.max(np.abs(first[within]-merged_first[within]),initial=0)<1e-6
            np.savez_compressed(folder/'actual_intersections.npz',**arrays,triangle_normals_world=normals,
                query_ids=ids,source_ids=source_ids,rays_world=rays,distinct_layers=counts,near_front_layers=near,close_same_facing_pairs=pairs)
            metrics=[]
            for group,choose in [('all_fixed_queries',np.ones(len(queries),bool)),('original_RGB_brightness',bright)]:
                metrics.append(dict(mesh_id=route['mesh_id'],mesh_stage=route['mesh_stage'],evaluated_mesh_path=str(path),group=group,source_views=len(set(source_ids[choose])),**summarize(counts,near,pairs,choose)))
            views=[]
            for sid in sorted(set(source_ids)):
                views.append(dict(mesh_id=route['mesh_id'],mesh_stage=route['mesh_stage'],evaluated_mesh_path=str(path),source_id=int(sid),**summarize(counts,near,pairs,source_ids==sid)))
            writecsv(folder/'summary.csv',metrics);writecsv(folder/'per_view.csv',views)
            assert (initial.st_size,initial.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
            done.write_text(json.dumps(dict(status='complete',**expected,original_triangle_count=len(mesh.triangles),actual_intersections=len(t),elapsed_seconds=time.time()-started,CPU_only=True,first_hit_consistency_with_actual_cast_rays=True),indent=2)+'\n')
            records=metrics
            del scene,mesh,tensor,result,arrays,vertices,faces,normals;gc.collect()
        summaries.extend(records);perviews.extend(views)
        writecsv(DEST/'ALL_MESH_LAYER_METRICS.csv',summaries);writecsv(DEST/'ALL_MESH_LAYER_PER_VIEW.csv',perviews)
        print('Actual same-ray layers',route['mesh_id'],route['mesh_stage'],flush=True)
    (DEST/'complete.json').write_text(json.dumps(dict(status='complete_available_standard_routes',actual_mesh_files=len(seen),queries=len(queries),source_views=len(set(source_ids)),scope=protocol['scope']),indent=2)+'\n')

if __name__=='__main__':main()
