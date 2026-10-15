"""Descriptive geometry only. No independent geometry ground truth exists."""
import numpy as np

def structure(mesh):
    v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    assert len(v)>0 and len(t)>0,'EMPTY_MESH'
    finite=np.isfinite(v).all(1)
    tri=v[t];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);twice_area=np.linalg.norm(cross,axis=1)
    diagonal=float(np.linalg.norm(np.ptp(v,axis=0)))
    degenerate=(twice_area<=max(diagonal**2*1e-12,1e-20))
    edges=np.sort(np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1)
    unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    face_ids=np.tile(np.arange(len(t)),3)
    order=np.argsort(inverse,kind='stable');offset=np.r_[0,np.cumsum(counts)]
    paired=np.flatnonzero(counts==2)
    a=face_ids[order[offset[paired]]];b=face_ids[order[offset[paired]+1]]
    normals=cross/np.maximum(twice_area[:,None],1e-30)
    valid=(~degenerate[a])&(~degenerate[b])
    cosine=np.sum(normals[a[valid]]*normals[b[valid]],axis=1).clip(-1,1)
    labels,ntri,area=mesh.cluster_connected_triangles();ntri=np.asarray(ntri)
    result={'vertices':len(v),'triangles':len(t),'connected_components':len(ntri),
            'largest_component_triangle_ratio':float(ntri.max()/len(t)),
            'components_ge50_triangles':int((ntri>=50).sum()),'boundary_edge_count':int((counts==1).sum()),
            'non_manifold_edge_count':int((counts>2).sum()),'degenerate_face_ratio':float(degenerate.mean()),
            'finite_vertex_ratio':float(finite.mean()),'bbox_min':v.min(0).tolist(),'bbox_max':v.max(0).tolist(),
            'surface_area_COLMAP_units_squared':float(twice_area.sum()/2),
            'adjacent_normal_cosine_mean':float(cosine.mean()) if len(cosine) else None,
            'adjacent_abs_normal_cosine_mean':float(np.abs(cosine).mean()) if len(cosine) else None,
            'independent_3D_GT':'NONE','physical_units':'arbitrary inherited COLMAP units',
            'normal_valid_fraction':float((~degenerate).mean())}
    return result,unique[counts==1]

def raycast(mesh,camera):
    import open3d as o3d
    import math
    h,w=camera.image_height,camera.image_width
    # Match this fork's to_cam_open3d pixel-center convention exactly.
    K=np.array([[w/(2*math.tan(camera.FoVx/2)),0,(w-1)/2],[0,h/(2*math.tan(camera.FoVy/2)),(h-1)/2],[0,0,1]],dtype=np.float64)
    extrinsic=camera.world_view_transform.T.detach().cpu().numpy().astype(np.float64)
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays=scene.create_rays_pinhole(o3d.core.Tensor(K),o3d.core.Tensor(extrinsic),w,h)
    result=scene.cast_rays(rays,nthreads=4)
    return result['t_hit'].numpy(),result['primitive_normals'].numpy()

def reprojection(mesh_depth,gaussian_depth,alpha,roi):
    visible=np.isfinite(mesh_depth)&(mesh_depth>0)
    supported=np.isfinite(gaussian_depth)&(gaussian_depth>0)&(alpha>0.5)
    shared=visible&supported&roi
    difference=np.abs(mesh_depth[shared]-gaussian_depth[shared])
    union=visible|roi
    return {'mesh_ROI_coverage_proxy':float(visible[roi].mean()),
            'mesh_proxy_silhouette_IoU':float((visible&roi).sum()/union.sum()),
            'mesh_Gaussian_shared_ROI_pixels':int(shared.sum()),
            'mesh_Gaussian_depth_MAE':float(difference.mean()) if len(difference) else None,
            'mesh_Gaussian_depth_median_absolute_difference':float(np.median(difference)) if len(difference) else None,
            'mesh_Gaussian_depth_p95_absolute_difference':float(np.quantile(difference,.95)) if len(difference) else None,
            'mesh_Gaussian_relative_depth_MAE':float((difference/gaussian_depth[shared]).mean()) if len(difference) else None,
            'geometry_accuracy_claim':False,'silhouette_is_independent_GT':False}
