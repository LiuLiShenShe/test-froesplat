"""Independent CPU metric checks; synthetic fixtures are not delivered as models."""
import json,sys,tempfile,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import mesh_surface_supplement as m

def grid(n=10,height=None):
    yy,xx=np.mgrid[0:n+1,0:n+1]/n
    z=np.zeros_like(xx) if height is None else height(xx,yy)
    v=np.column_stack([xx.ravel(),yy.ravel(),z.ravel()])
    f=[]
    for y in range(n):
        for x in range(n):
            a=y*(n+1)+x;b=a+1;c=a+n+1;d=c+1
            f.extend([[a,b,d],[a,d,c]])
    return v,np.asarray(f,dtype=np.int32)

def ring():
    v=np.asarray([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2],[-.3,-.3,2],[.3,-.3,2],[.3,.3,2],[-.3,.3,2]],float)
    f=[]
    for a in range(4):
        b=(a+1)%4;f.extend([[a,b,b+4],[a,b+4,a+4]])
    return v,np.asarray(f,dtype=np.int32)

def main():
    start=time.time();records={};out=m.ROOT/'validation/surface_supplement';out.mkdir(parents=True,exist_ok=True)
    for name,n in [('planar_coarse',1),('planar_dense',25)]:
        v,f=grid(n);edges,components,stats=m.boundary_graph(v,f)
        assert stats['closed_boundary_loops']==1 and stats['open_boundary_chains']==0 and stats['nonmanifold_edges']==0
        assert abs(components[0]['row']['length']-4)<1e-12
        r=m.fixed_scale_roughness(v,f,1,out/name,seed=0,radius_divisors=(8,4),support_cap=5000,queries=200)
        rows=list(__import__('csv').DictReader((out/name/'fixed_scale_roughness.csv').open()))
        assert all(float(x['valid_fraction'])>.99 for x in rows)
        assert all(float(x['query_plane_distance_mean'])<1e-12 and float(x['local_plane_rms_mean'])<1e-12 for x in rows)
        records[name]=dict(topology=stats,roughness_max=max(float(x['local_plane_rms_mean']) for x in rows),support_capped=r['support_capped'])
    v,f=grid(40,lambda x,y:.2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))
    r=m.fixed_scale_roughness(v,f,1,out/'curved',seed=0,radius_divisors=(8,4),support_cap=5000,queries=200)
    rows=list(__import__('csv').DictReader((out/'curved/fixed_scale_roughness.csv').open()))
    assert all(float(x['local_plane_rms_mean'])>.001 for x in rows)
    records['curved_surface']=dict(local_plane_rms=[float(x['local_plane_rms_mean']) for x in rows],meaning='Real surface curvature raises proxy; not an error classification')
    v,f=ring();edges,components,stats=m.boundary_graph(v,f)
    assert stats['closed_boundary_loops']==2 and stats['boundary_edges']==8
    assert np.allclose(sorted(x['row']['projected_best_fit_polygon_area'] for x in components),[.36,4])
    records['annulus']=stats
    v2=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    tetra=np.asarray([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],dtype=np.int32)
    _,_,s=m.boundary_graph(v2,tetra);assert s['boundary_edges']==0 and s['closed_boundary_loops']==0
    records['closed_tetrahedron']=s
    v3=np.vstack([v2,[.5,-.5,0]])
    _,_,s=m.boundary_graph(v3,np.vstack([tetra,[0,1,4]]));assert s['open_boundary_chains']==1 and s['nonmanifold_edges']==1
    records['nonmanifold_open_chain']=s
    v4=np.asarray([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],float)
    _,_,s=m.boundary_graph(v4,np.asarray([[0,1,2],[0,3,4]],dtype=np.int32))
    assert s['closed_boundary_loops']==0 and s['branched_boundary_networks']==1 and s['boundary_graph_cycle_rank']==2
    records['two_cycles_shared_vertex']=s
    # Three views use the same physical boundary points and cameras. Different RAP
    # evidence/first-hit depth must alter only proxy attribution, never topology.
    camera=dict(source_id=0,name='crop_0000',width=64,height=64,Fx=32.,Fy=32.,Cx=32.,Cy=32.,world_view_transform=np.eye(4).tolist())
    m.write(out/'cameras.json',dict(records=[camera]));projection={}
    for case in ['missing_inside_foreground','actual_mask_void','occluded']:
        directory=out/case/'views';directory.mkdir(parents=True,exist_ok=True)
        fg=np.zeros((64,64),bool);fg[16:49,16:49]=True
        pred=fg.copy();pred[28:37,28:37]=False
        if case=='actual_mask_void':fg=pred.copy()
        z=np.where(pred,1. if case=='occluded' else 2.,np.inf)
        np.savez(directory/'post_0000.npz',mesh_depth=z,foreground_mask=fg,mesh_silhouette=pred)
        result,views=m.boundary_projection_proxy(v,edges,components,2,out/'cameras.json',directory.parent,'post')
        types=sorted(x['projection_class'] for x in result)
        if case=='missing_inside_foreground':assert types==['foreground_mask_boundary_supported','internal_missing_boundary_candidate'],result
        if case=='actual_mask_void':assert types==['foreground_mask_boundary_supported']*2,result
        if case=='occluded':assert types==['unobserved_or_occluded']*2,result
        projection[case]=dict(component_results=result,view_results=views)
    records['projection_attribution']=projection
    # Explicit opposite-oriented nearby sheets illustrate the contamination filter.
    a,b=grid(1);v5=np.vstack([a,a+[0,0,.02]]);f5=np.vstack([b,b[:,::-1]+len(a)])
    m.fixed_scale_roughness(v5,f5,1,out/'opposed_parallel_sheets',radius_divisors=(8,),support_cap=5000,queries=200)
    rows=list(__import__('csv').DictReader((out/'opposed_parallel_sheets/fixed_scale_roughness.csv').open()))
    assert float(rows[0]['local_plane_rms_mean'])>.005 and float(rows[1]['local_plane_rms_mean'])<1e-10
    records['opposed_parallel_sheets']=dict(all_spatial_rms=float(rows[0]['local_plane_rms_mean']),same_oriented_rms=float(rows[1]['local_plane_rms_mean']),meaning='Filtering shows nearby sheet mixing; may exclude real steep folds')
    m.write(out/'VALIDATION.json',dict(status='all_checks_passed',elapsed_seconds=time.time()-start,seed=0,records=records,
       policy='Synthetic in-memory fixtures and small cached depth/mask arrays only; no synthetic mesh is exported or indexed as an experiment model'))
    print(json.dumps(dict(status='all_checks_passed',path=str(out/'VALIDATION.json'),seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
