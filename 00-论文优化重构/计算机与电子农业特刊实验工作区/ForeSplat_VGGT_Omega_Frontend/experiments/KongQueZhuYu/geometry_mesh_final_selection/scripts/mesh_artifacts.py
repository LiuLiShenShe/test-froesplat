"""Preserve raw mesh, every component and exactly which geometry is removed."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from surface_protocol import write

def save_meshes(mesh,out,scale):
    out.mkdir(parents=True,exist_ok=True)
    assert len(mesh.vertices)>0 and len(mesh.triangles)>0
    mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(out/'raw.ply'),mesh,write_ascii=False)
    labels,counts,areas=mesh.cluster_connected_triangles();labels=np.asarray(labels);counts=np.asarray(counts);areas=np.asarray(areas)
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles);remove=counts[labels]<50
    removed_faces=np.flatnonzero(remove);retained_vertex=np.unique(faces[~remove]);removed_vertex=np.setdiff1d(np.arange(len(vertices)),retained_vertex)
    np.savez_compressed(out/'components_and_removals.npz',triangle_component=labels,component_faces=counts,component_areas=areas,
        removed_face_indices=removed_faces,removed_face_centroids=vertices[faces[remove]].mean(axis=1),removed_vertex_indices=removed_vertex,removed_vertex_positions=vertices[removed_vertex])
    post=o3d.geometry.TriangleMesh(mesh);post.remove_triangles_by_mask(remove);post.remove_unreferenced_vertices();post.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out/'post.ply'),post,write_ascii=False)
    # Nearest original observation colours do not alter geometry.
    write(out/'postprocess.json',dict(rule='Remove components with <50 triangles only; all larger components retained',raw_vertices=len(vertices),raw_faces=len(faces),raw_components=len(counts),
        deleted_vertices=len(removed_vertex),deleted_faces=len(removed_faces),kept_components=int((counts>=50).sum()),deleted_component_area=float(areas[counts<50].sum()),scale=scale,
        raw_is_preserved=True,removed_geometry_archive=str(out/'components_and_removals.npz')))
    return dict(raw_vertices=len(vertices),raw_faces=len(faces),post_vertices=len(post.vertices),post_faces=len(post.triangles),raw_components=len(counts),post_components=int((counts>=50).sum()))

def colour_from_observations(mesh,xyz,rgb):
    _,idx=cKDTree(xyz).query(np.asarray(mesh.vertices),workers=8)
    mesh.vertex_colors=o3d.utility.Vector3dVector(np.asarray(rgb[idx],dtype=float)/255.)
    return mesh
