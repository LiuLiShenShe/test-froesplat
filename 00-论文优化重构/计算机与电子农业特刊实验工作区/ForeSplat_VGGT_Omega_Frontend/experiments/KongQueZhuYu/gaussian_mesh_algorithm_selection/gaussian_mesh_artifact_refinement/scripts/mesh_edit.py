from common import *
import open3d as o3d,csv,shutil
def subset(mesh,ids,path):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);col=np.asarray(mesh.vertex_colors);used=np.unique(f[ids]);mapping=np.full(len(v),-1,np.int64);mapping[used]=np.arange(len(used));m=o3d.geometry.TriangleMesh();m.vertices=o3d.utility.Vector3dVector(v[used]);m.triangles=o3d.utility.Vector3iVector(mapping[f[ids]]);m.vertex_colors=o3d.utility.Vector3dVector(col[used]);m.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(path),m)
def main():
    d=ROOT/'task1_p3_local_repair';versions=d/'mesh_versions';path=SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply';mesh=o3d.io.read_triangle_mesh(str(path));comp=np.asarray(mesh.cluster_connected_triangles()[0]);allids=np.arange(len(mesh.triangles));probe=json.loads((d/'component_multiview_evidence.json').read_text());marker=probe['marker_component']
    for stage in ['raw','post']:
        target=versions/f'P3_original_{stage}.ply'
        if not target.exists():target.symlink_to(SEL/f'P3_GS2Mesh/meshes_shared_depth_range/{stage}.ply')
    removed1=np.isin(comp,[28,29]);removed2=removed1|(comp==marker)
    subset(mesh,allids[removed1],versions/'P3_targeted_removed_components.ply')
    subset(mesh,allids[comp==marker],d/'pot_marker/marker_component.ply')
    for name,removed in [('P3_local_cleanup_v1',removed1),('P3_local_cleanup_v2',removed2)]:
        subset(mesh,allids[~removed],versions/f'{name}.ply');np.save(versions/f'{name}_original_triangle_ids.npy',allids[~removed])
        with (versions/f'{name}_deleted_triangles.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['original_triangle_id','original_component_id','reason'])
            for i in allids[removed]:w.writerow([i,comp[i],'source157 independent ring: no GGGS depth support in208 views' if comp[i] in [28,29] else 'Independent blue reference marker, semantic exclusion from Plant+Pot'])
    marker_rec=next(r for r in probe['summary'] if r['component_id']==marker)
    write(d/'pot_marker/marker_region.json',dict(**marker_rec,semantic='Blue independent reference marker identified at actual source0 RGB pixel650,1390; preserved in original baseline, excluded from Plant+Pot output. Actual marker has stereo/depth support and is not labeled reconstruction error.'))
    ev=np.load(d/'face_multiview_evidence.npz');candidate=(ev['component']==int(np.bincount(comp).argmax()))&(ev['visible']>=15)&(ev['fg']>=5)&(ev['gggs_depth_support']==0)&(ev['stereo_depth_support']==0)&(ev['slender']>20)
    ids=ev['original_triangle_ids'][candidate]
    if len(ids):subset(mesh,ids,d/'wrong_connections/unsupported_slender_candidates.ply')
    np.save(d/'wrong_connections/candidate_original_triangle_ids.npy',ids)
    write(d/'edits_complete.json',dict(source=str(path),source_sha256=sha(path),v1_deleted=int(removed1.sum()),v2_deleted=int(removed2.sum()),marker_component=marker,wrong_connection_candidates=len(ids),wrong_connections_edited=False,global_smoothing=False,source157_original_component_ids=[28,29]))
    print('mesh edits',int(removed1.sum()),int(removed2.sum()),'unconfirmed bridges',len(ids),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
