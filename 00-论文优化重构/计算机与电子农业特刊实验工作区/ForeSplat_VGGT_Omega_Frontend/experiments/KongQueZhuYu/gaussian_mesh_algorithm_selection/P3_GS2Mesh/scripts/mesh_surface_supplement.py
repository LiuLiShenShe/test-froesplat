"""CPU supplements: fixed-radius surface-plane residuals and true boundary graphs.
All distances are reconstruction-gauge proxy measurements, never scan/GT error.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import argparse,csv,hashlib,json,math,resource,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT.parents[1]/"geometry_mesh_final_selection"
REPS=[0,26,53,79,105,131,157,183]

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n");tmp.replace(path)

def csvwrite(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_mesh(path):
    import open3d as o3d
    mesh=o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.vertices) or not len(mesh.triangles):raise ValueError("Empty actual mesh: "+str(path))
    # Drop color/normal buffers held by Open3D after making these small necessary copies.
    v=np.array(mesh.vertices,dtype=np.float64)
    f=np.array(mesh.triangles,dtype=np.int32)
    del mesh
    if not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError("Invalid vertex/face array")
    return v,f,o3d.__version__

def face_geometry(v,f,chunk=100000):
    area=np.zeros(len(f),dtype=np.float64)
    normal=np.zeros((len(f),3),dtype=np.float32)
    for start in range(0,len(f),chunk):
        tri=f[start:start+chunk]
        n=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
        length=np.linalg.norm(n,axis=1)
        area[start:start+len(tri)]=length*.5
        normal[start:start+len(tri)]=(n/np.maximum(length[:,None],1e-300)).astype(np.float32)
    return area,normal

def sample_surface(v,f,area,normal,n,seed,stream):
    cdf=np.cumsum(area);total=float(cdf[-1])
    if not total>0:raise ValueError("No positive-area faces")
    rng=np.random.default_rng(np.random.SeedSequence([seed,stream]))
    ids=np.searchsorted(cdf,rng.random(n)*total,side="right")
    a=rng.random(n);b=rng.random(n);root=np.sqrt(a)
    bary=np.column_stack([1-root,root*(1-b),root*b])
    xyz=np.sum(v[f[ids]]*bary[...,None],axis=1)
    return xyz,normal[ids],ids

def summary_values(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if not len(values):return dict(mean=None,median=None,p90=None,p95=None)
    return dict(mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)))

def fixed_scale_roughness(v,f,diagonal,out,seed=0,spacing_divisor=512,
                          radius_divisors=(64,32),support_cap=200000,queries=3000,
                          min_neighbors=8,normal_angle_deg=60.,workers=4):
    start=time.time();area,face_normal=face_geometry(v,f);total=float(area.sum())
    target_spacing=diagonal/spacing_divisor
    requested=max(1000,int(math.ceil(total/target_spacing**2)))
    n=min(requested,support_cap)
    support,snormal,sface=sample_surface(v,f,area,face_normal,n,seed,0)
    query,qnormal,qface=sample_surface(v,f,area,face_normal,queries,seed,1)
    degenerate=int((area<=diagonal**2*1e-16).sum())
    del area,face_normal
    actual_spacing=math.sqrt(total/n)
    tree=cKDTree(support)
    rows=[];archive=dict(query_xyz=query,query_face_id=qface,query_face_normal=qnormal,
                       support_xyz=support,support_face_id=sface,support_face_normal=snormal)
    threshold=math.cos(math.radians(normal_angle_deg))
    for divisor in radius_divisors:
        radius=diagonal/divisor
        neighbors=tree.query_ball_point(query,radius,workers=workers,return_sorted=True)
        for mode in ["all_spatial_neighbors","same_oriented_normal"]:
            count=np.zeros(queries,dtype=np.int32)
            distance=np.full(queries,np.nan);rms=np.full(queries,np.nan);variation=np.full(queries,np.nan)
            plane_normal=np.full((queries,3),np.nan)
            for i,idx in enumerate(neighbors):
                idx=np.asarray(idx,dtype=np.int64)
                if mode=="same_oriented_normal":
                    idx=idx[(snormal[idx]@qnormal[i])>=threshold]
                count[i]=len(idx)
                if len(idx)<min_neighbors:continue
                p=support[idx];mean=p.mean(0);center=p-mean
                eigen,vectors=np.linalg.eigh(center.T@center/len(p))
                eigen=np.maximum(eigen,0)
                if eigen[1]<radius**2*1e-12:continue  # Degenerate line neighborhood is not a plane estimate.
                pn=vectors[:,0]
                distance[i]=abs(float((query[i]-mean)@pn))
                rms[i]=math.sqrt(float(eigen[0]))
                variation[i]=float(eigen[0]/max(eigen.sum(),1e-300))
                plane_normal[i]=pn
            valid=np.isfinite(rms)
            row=dict(radius_divisor=divisor,radius=radius,radius_normalized=1/divisor,neighborhood=mode,
                queries=queries,valid_queries=int(valid.sum()),valid_fraction=float(valid.mean()),
                neighbor_count_median=float(np.median(count)),neighbor_count_p10=float(np.quantile(count,.1)),
                support_samples=n,target_spacing=target_spacing,effective_spacing=actual_spacing,
                spacing_capped=n<requested,normal_angle_deg=normal_angle_deg if mode=="same_oriented_normal" else None)
            for name,values in [("query_plane_distance",distance),("local_plane_rms",rms),("surface_variation",variation)]:
                row.update({name+"_"+k:x for k,x in summary_values(values).items()})
                if name!="surface_variation":
                    row.update({name+"_normalized_"+k:x for k,x in summary_values(values/diagonal).items()})
            rows.append(row)
            tag="r"+str(divisor)+"_"+mode
            archive[tag+"_neighbor_count"]=count;archive[tag+"_query_plane_distance"]=distance
            archive[tag+"_local_plane_rms"]=rms;archive[tag+"_plane_normal"]=plane_normal
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez(out/"surface_samples_and_residuals.npz",**archive)
    csvwrite(out/"fixed_scale_roughness.csv",rows)
    result=dict(status="complete",seed=seed,query_stream=1,support_stream=0,surface_area=total,
       vertices=len(v),faces=len(f),degenerate_faces=degenerate,diagonal=diagonal,
       target_spacing=target_spacing,effective_spacing=actual_spacing,support_requested=requested,support_actual=n,
       support_capped=n<requested,query_count=queries,radius_divisors=list(radius_divisors),
       elapsed_seconds=time.time()-start,min_neighbors=min_neighbors,normal_angle_deg=normal_angle_deg,
       sampling="Independent area-uniform barycentric surface points, not mesh vertices; count targets area/spacing^2, capped explicitly",
       plane="Least-squares PCA plane over support neighbors excluding independent query; query distance and sqrt smallest covariance eigenvalue",
       interpretation="Fixed spatial scale lowers tessellation bias. Includes real curvature, nearby sheets and noise; not true3D distance error. Same-normal filter excludes opposite sides/steep folds but may exclude valid fold detail.",
       comparability="Radius/diagonal fixed; sampling density changes when capped; report actual spacing/valid fraction. Different mesh areas are not identical anatomical regions.",
       reference="https://github.com/CloudCompare/CCCoreLib/blob/master/src/Neighbourhood.cpp")
    write(out/"roughness_complete.json",result)
    return result

def boundary_graph(v,f):
    """Boundary incidence count1; in-place scalar edge sort limits15m-face memory."""
    nv=len(v);n=len(f);codes=np.empty(n*3,dtype=np.int64)
    for i,(a,b) in enumerate([(0,1),(1,2),(2,0)]):
        lo=np.minimum(f[:,a],f[:,b]).astype(np.int64)
        target=codes[i*n:(i+1)*n];np.multiply(lo,nv,out=target)
        target+=np.maximum(f[:,a],f[:,b]);del lo
    codes.sort()
    one=np.ones(len(codes),dtype=bool)
    one[1:] &= codes[1:]!=codes[:-1]
    one[:-1] &= codes[:-1]!=codes[1:]
    boundary_codes=codes[one];del one
    unique_edges=int(np.count_nonzero(codes[1:]!=codes[:-1])+1) if len(codes) else 0
    if len(codes)>=3:
        first_three=np.ones(len(codes)-2,dtype=bool)
        first_three[1:] &= codes[1:-2]!=codes[:-3]
        first_three &= codes[:-2]==codes[1:-1]
        first_three &= codes[:-2]==codes[2:]
        nonmanifold=int(first_three.sum());del first_three
    else:nonmanifold=0
    del codes
    edges=np.column_stack([boundary_codes//nv,boundary_codes%nv]).astype(np.int64)
    self_edges=int((edges[:,0]==edges[:,1]).sum())
    edges=edges[edges[:,0]!=edges[:,1]]
    graph={}
    for eid,(a,b) in enumerate(edges):
        a=int(a);b=int(b);graph.setdefault(a,[]).append((b,eid));graph.setdefault(b,[]).append((a,eid))
    visited=set();components=[]
    for first in sorted(graph):
        if first in visited:continue
        stack=[first];visited.add(first);nodes=[];eids=set()
        while stack:
            a=stack.pop();nodes.append(a)
            for b,eid in graph[a]:
                eids.add(eid)
                if b not in visited:visited.add(b);stack.append(b)
        degree=np.array([len(graph[x]) for x in nodes])
        kind="closed_loop" if len(nodes)>=3 and np.all(degree==2) else (
             "open_chain" if np.all(degree<=2) and int((degree==1).sum())==2 else "branched_boundary_network")
        ordered=[]
        if kind in ["closed_loop","open_chain"]:
            current=min(nodes) if kind=="closed_loop" else min(x for x in nodes if len(graph[x])==1)
            previous=None;ordered=[current]
            while True:
                options=[x for x,_ in graph[current] if x!=previous]
                if not options:break
                nextnode=min(options)
                if nextnode==ordered[0]:break
                ordered.append(nextnode);previous,current=current,nextnode
                if len(ordered)>len(nodes):raise RuntimeError("Invalid boundary walk")
            if len(ordered)!=len(nodes):raise RuntimeError("Boundary walk lost vertices")
        eids=np.asarray(sorted(eids),dtype=np.int64)
        segments=edges[eids];lengths=np.linalg.norm(v[segments[:,1]]-v[segments[:,0]],axis=1)
        length=float(lengths.sum())
        row=dict(boundary_id=len(components),kind=kind,vertices=len(nodes),edges=len(eids),
             cycle_rank=len(eids)-len(nodes)+1,endpoints=int((degree==1).sum()),branch_vertices=int((degree>2).sum()),
             length=length,bbox_diagonal=float(np.linalg.norm(v[nodes].max(0)-v[nodes].min(0))),
             projected_best_fit_polygon_area=None,plane_max_deviation=None)
        if kind=="closed_loop":
            xyz=v[ordered];center=xyz.mean(0);eigen,axes=np.linalg.eigh((xyz-center).T@(xyz-center)/len(xyz))
            uv=(xyz-center)@axes[:,1:]
            row["projected_best_fit_polygon_area"]=float(abs(np.sum(uv[:,0]*np.roll(uv[:,1],-1)-uv[:,1]*np.roll(uv[:,0],-1)))*.5)
            row["plane_max_deviation"]=float(np.max(np.abs((xyz-center)@axes[:,0])))
        components.append(dict(row=row,ordered=np.asarray(ordered,dtype=np.int64),edge_ids=eids))
    summary=dict(boundary_edges=len(edges),boundary_self_edges_excluded=self_edges,total_unique_edges=unique_edges,
       nonmanifold_edges=nonmanifold,boundary_components=len(components),
       closed_boundary_loops=sum(x["row"]["kind"]=="closed_loop" for x in components),
       open_boundary_chains=sum(x["row"]["kind"]=="open_chain" for x in components),
       branched_boundary_networks=sum(x["row"]["kind"]=="branched_boundary_network" for x in components),
       boundary_graph_cycle_rank=sum(x["row"]["cycle_rank"] for x in components),
       definition="Only all-degree2 simple-cycle components count as unambiguous boundary loops. Branched networks retain cycle-rank separately; loops are not automatically holes.")
    return edges,components,summary

def sample_boundary_component(v,edges,component,spacing,cap=2048):
    ids=component["edge_ids"];segment=edges[ids]
    length=np.linalg.norm(v[segment[:,1]]-v[segment[:,0]],axis=1)
    total=float(length.sum())
    if not total>0:return np.empty((0,3))
    n=min(cap,max(3,int(math.ceil(total/spacing))))
    distance=(np.arange(n)+.5)/n*total
    cumulative=np.cumsum(length);index=np.searchsorted(cumulative,distance)
    start=np.r_[0,cumulative[:-1]][index]
    fraction=(distance-start)/np.maximum(length[index],1e-300)
    selected=segment[index]
    return v[selected[:,0]]*(1-fraction[:,None])+v[selected[:,1]]*fraction[:,None]

def enclosed_missing(pred,fg,rim_pixels=3,min_pixels=9):
    missing=fg&~pred;labels,n=ndimage.label(missing)
    rim=fg&~ndimage.binary_erosion(fg,iterations=rim_pixels)
    touches=set(np.unique(labels[rim]));touches.add(0)
    size=np.bincount(labels.ravel())
    chosen=[i for i in range(1,n+1) if i not in touches and size[i]>=min_pixels]
    holes=np.isin(labels,chosen)
    return holes,dict(enclosed_missing_components=len(chosen),enclosed_missing_area_px=int(holes.sum()),
                      open_boundary_missing_area_px=int(missing.sum()-holes.sum()))

def read_cameras(path):
    obj=json.loads(Path(path).read_text())
    if isinstance(obj,dict):obj=obj.get("records",obj.get("frames",obj.get("views")))
    if not isinstance(obj,list):raise ValueError("Camera index must be a list or contain records/frames/views")
    out=[]
    for row in obj:
        c=row.get("camera",row)
        sid=row.get("source_id",c.get("source_id"))
        if sid is None:sid=int(c.get("name",c.get("image_name","")).split("_")[-1])
        out.append(dict(c,source_id=int(sid)))
    return out

def boundary_projection_proxy(v,edges,components,diagonal,camera_index,evaluation_dir,stage,source_ids=REPS):
    cameras=read_cameras(camera_index);directory=Path(evaluation_dir)
    samples=[sample_boundary_component(v,edges,x,diagonal/1024) for x in components]
    count=len(samples);total=np.zeros((count,5),dtype=np.int64);rows=[];projection_rows=[]
    for c in cameras:
        sid=c["source_id"]
        if sid not in source_ids:continue
        path=directory/"views"/f"{stage}_{sid:04d}.npz"
        if not path.exists():continue
        with np.load(path) as d:
            z=d["mesh_depth"].copy();fg=d["foreground_mask"].astype(bool);pred=d["mesh_silhouette"].astype(bool)
            holes=d["internal_missing_proxy"].astype(bool) if "internal_missing_proxy" in d else enclosed_missing(pred,fg)[0]
        _,missing_stats=enclosed_missing(pred,fg)
        gt_edge=fg^ndimage.binary_erosion(fg)
        edge_distance=ndimage.distance_transform_edt(~gt_edge)
        hole_distance=ndimage.distance_transform_edt(~holes) if holes.any() else None
        E=np.asarray(c["world_view_transform"]).T;h,w=z.shape
        view_count=np.zeros(5,dtype=np.int64)
        for i,xyz in enumerate(samples):
            if not len(xyz):continue
            p=xyz@E[:3,:3].T+E[:3,3]
            with np.errstate(divide="ignore",invalid="ignore"):
                u=p[:,0]/p[:,2]*c["Fx"]+c["Cx"];vv=p[:,1]/p[:,2]*c["Fy"]+c["Cy"]
            good=np.isfinite(u)&np.isfinite(vv)&(p[:,2]>0)&(u>=0)&(u<w)&(vv>=0)&(vv<h)
            if not good.any():continue
            pc=p[good];ix=np.rint(u[good]).astype(np.int64);iy=np.rint(vv[good]).astype(np.int64)
            ix=np.clip(ix,0,w-1);iy=np.clip(iy,0,h-1)
            best=np.full(len(ix),np.inf)
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    reference=z[np.clip(iy+dy,0,h-1),np.clip(ix+dx,0,w-1)]
                    delta=np.where(np.isfinite(reference)&(reference>0),np.abs(reference-pc[:,2]),np.inf)
                    best=np.minimum(best,delta)
            visible=best<=np.maximum(diagonal/1024,.002*np.abs(pc[:,2]))
            mask_edge=visible&(edge_distance[iy,ix]<=3)
            internal=visible&~mask_edge&(fg[iy,ix])&(hole_distance[iy,ix]<=3) if hole_distance is not None else np.zeros_like(visible)
            outside=visible&~fg[iy,ix]&~mask_edge
            counts=np.array([len(ix),int(visible.sum()),int(mask_edge.sum()),int(internal.sum()),int(outside.sum())])
            total[i]+=counts;view_count+=counts
        projection_rows.append(dict(source_id=sid,inframe_boundary_samples=int(view_count[0]),visible_boundary_samples=int(view_count[1]),
              foreground_mask_edge_supported_samples=int(view_count[2]),internal_missing_edge_proxy_samples=int(view_count[3]),
              outside_mask_visible_samples=int(view_count[4]),**missing_stats))
    for i,x in enumerate(components):
        inf,visible,edge,internal,outside=map(int,total[i])
        kind="unobserved_or_occluded" if visible==0 else ("foreground_mask_boundary_supported" if edge/visible>=.5 else (
             "internal_missing_boundary_candidate" if internal/visible>=.5 else "mixed_or_ambiguous"))
        rows.append(dict(boundary_id=x["row"]["boundary_id"],projection_class=kind,inframe_samples=inf,visible_samples=visible,
               visible_fraction=visible/max(inf,1),foreground_mask_edge_fraction=edge/max(visible,1),
               internal_missing_edge_fraction=internal/max(visible,1),outside_mask_fraction=outside/max(visible,1)))
    return rows,projection_rows

def analyze_boundary(v,f,diagonal,out,camera_index=None,evaluation_dir=None,stage="post"):
    start=time.time();edges,components,summary=boundary_graph(v,f);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[dict(x["row"],length_normalized=x["row"]["length"]/diagonal,
           projected_best_fit_polygon_area_normalized=None if x["row"]["projected_best_fit_polygon_area"] is None else x["row"]["projected_best_fit_polygon_area"]/diagonal**2) for x in components]
    ordered=np.concatenate([x["ordered"] for x in components]) if components else np.empty(0,dtype=np.int64)
    offsets=np.r_[0,np.cumsum([len(x["ordered"]) for x in components])]
    all_edge_ids=np.concatenate([x["edge_ids"] for x in components]) if components else np.empty(0,dtype=np.int64)
    edge_offsets=np.r_[0,np.cumsum([len(x["edge_ids"]) for x in components])]
    np.savez(out/"boundary_topology.npz",boundary_edges=edges,ordered_vertex_ids=ordered,component_offsets=offsets,
             component_edge_ids=all_edge_ids,component_edge_offsets=edge_offsets)
    csvwrite(out/"boundary_components.csv",rows)
    summary["projection_status"]="not_requested"
    if camera_index and evaluation_dir:
        boundary_rows,view_rows=boundary_projection_proxy(v,edges,components,diagonal,camera_index,evaluation_dir,stage)
        csvwrite(out/"boundary_projection_classes.csv",boundary_rows);csvwrite(out/"boundary_projection_views.csv",view_rows)
        classes={}
        for row in boundary_rows:classes[row["projection_class"]]=classes.get(row["projection_class"],0)+1
        summary.update(projection_evaluated_views=len(view_rows),projection_requested_source_ids=REPS,
            projection_actual_source_ids=[row["source_id"] for row in view_rows],
            projection_status="complete_fixed8" if len(view_rows)==len(REPS) else ("partial_views" if view_rows else "caches_unavailable"),
            projection_component_classes=classes,projection_boundary_spacing=diagonal/1024,
            projection_boundary_sample_cap_per_component=2048,
            projection_evidence="Existing fixed8 full-resolution first-hit depth and RAP mask caches; no new GPU or full208 evaluation")
    summary.update(status="complete",diagonal=diagonal,elapsed_seconds=time.time()-start,
       leaf_edge_policy="Visible foreground-mask boundary support is a leaf/foreground-edge candidate; RAP also includes pot/marker. Internal missing projection is a separate proxy. Neither labels physical3D holes.",
       area_policy="Boundary polygon best-fit projected area is not a hole area; outer leaf perimeters also enclose polygons.",
       visibility_policy="Boundary point camera Z agrees within max(diagonal/1024,0.2% depth) with finite first-hit depth in3x3 image neighborhood; occluded points are excluded")
    write(out/"boundary_complete.json",summary)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument("--mesh",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--frontend",choices=["standard","omega"],default="standard");p.add_argument("--scale-json",type=Path)
    p.add_argument("--mode",choices=["roughness","boundary","all"],default="all")
    p.add_argument("--support-cap",type=int,default=200000);p.add_argument("--queries",type=int,default=3000)
    p.add_argument("--radius-divisors",type=float,nargs="+",default=[64,32]);p.add_argument("--spacing-divisor",type=float,default=512)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--workers",type=int,default=4);p.add_argument("--normal-angle",type=float,default=60)
    p.add_argument("--camera-index",type=Path);p.add_argument("--evaluation-dir",type=Path);p.add_argument("--stage",default="post")
    a=p.parse_args();start=time.time();a.out.mkdir(parents=True,exist_ok=True)
    scale_file=a.scale_json or HISTORY/"configs"/(a.frontend+"_spatial_scale.json")
    scale=json.loads(scale_file.read_text());diagonal=float(scale["diagonal"])
    if diagonal<=0 or min(a.radius_divisors)<=0 or a.spacing_divisor<=0 or a.support_cap<8 or a.queries<1 or not 0<=a.normal_angle<=180:raise ValueError("Invalid metric parameters")
    load_start=time.time();v,f,version=load_mesh(a.mesh);load_time=time.time()-load_start;results={}
    if a.mode in ["roughness","all"]:
        results["roughness"]=fixed_scale_roughness(v,f,diagonal,a.out,a.seed,a.spacing_divisor,a.radius_divisors,a.support_cap,a.queries,normal_angle_deg=a.normal_angle,workers=a.workers)
    if a.mode in ["boundary","all"]:
        results["boundary"]=analyze_boundary(v,f,diagonal,a.out,a.camera_index,a.evaluation_dir,a.stage)
    metadata=dict(status="complete",mesh=str(a.mesh.resolve()),mesh_bytes=a.mesh.stat().st_size,scale_file=str(scale_file.resolve()),
       frontend=a.frontend,coordinate_system="Existing frontend world gauge; no3D reference/metric calibration",
       mode=a.mode,seed=a.seed,vertices=len(v),faces=len(f),mesh_load_seconds=load_time,total_wall_seconds=time.time()-start,
       peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,open3d_version=version,numpy_version=np.__version__,
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results)
    write(a.out/"supplement_complete.json",metadata)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
    # Avoid known Open3D interpreter teardown faults only after all results succeeded.
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

