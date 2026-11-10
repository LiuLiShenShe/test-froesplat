from common import *
import open3d as o3d, csv

def main():
    root=ROOT/'task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh';src=root/'meshes_shared_depth_range/post.ply';out=root/'local_repair';out.mkdir(exist_ok=True)
    m=o3d.io.read_triangle_mesh(str(src));v=np.asarray(m.vertices);f=np.asarray(m.triangles);co=np.asarray(m.cluster_connected_triangles()[0]);cent=v[f].mean(1)
    original=json.loads((ROOT/'task1_p3_local_repair/pot_marker/marker_region.json').read_text());lo=np.array(original['bbox_min'])-.02;hi=np.array(original['bbox_max'])+.02;inside=((cent>=lo)&(cent<=hi)).all(1)
    selected=[];records=[]
    for k in np.unique(co[inside]):
        ix=np.flatnonzero(co==k);p=v[np.unique(f[ix])];fraction=float(inside[ix].mean())
        # Below native pot base and spatially corresponding to the known separate reference marker.
        take=p[:,1].min()>.20 and fraction>.65
        records.append(dict(new_component_id=int(k),faces=len(ix),bbox_min=p.min(0).tolist(),bbox_max=p.max(0).tolist(),fraction_in_expanded_original_marker_aabb=fraction,removed=bool(take)))
        if take:selected.extend(ix.tolist())
    removed=np.unique(selected);keep=np.setdiff1d(np.arange(len(f)),removed)
    for name,ii in [('G2_marker_removed_components',removed),('G2_targeted_cleaned_mesh',keep)]:
        result=o3d.geometry.TriangleMesh(m);remove=np.ones(len(f),bool);remove[ii]=False;result.remove_triangles_by_mask(remove);result.remove_unreferenced_vertices();o3d.io.write_triangle_mesh(str(out/(name+'.ply')),result,write_ascii=False);np.save(out/(name+'_original_g2_triangle_ids.npy'),ii)
    with (out/'removed_triangles.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['original_G2_triangle_id','G2_component_id','reason'])
        for tid in removed:w.writerow([int(tid),int(co[tid]),'Separate non-target marker at matched original world location below pot; freshly generated G2 component IDs'])
    (out/'post.ply').symlink_to('G2_targeted_cleaned_mesh.ply') if not (out/'post.ply').exists() else None
    write(out/'REPAIR_PROVENANCE.json',dict(source=str(src),source_sha256=sha(src),marker_reference_world_region=original,records=records,removed_triangles=len(removed),kept_triangles=len(keep),selection='World correspondence and Y>.20 below target pot; independent disconnected components. Original P3 component97 NOT reused as G2 ID.',status='Geometry edited; 208 metrics and visual review separate'))
    print('G2 marker repair',len(removed),[r['new_component_id'] for r in records if r['removed']],flush=True)
if __name__=='__main__':main()
