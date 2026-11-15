"""Protect historical surface semantics while changing memory and I/O layout."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nksr_pipeline.common import (Meter, StreamingObservations, read_mesh,
    unique_observations, unproject, write_mesh)


def test_streaming_selection_matches_historical_merge():
    rng = np.random.default_rng(15)
    scale = dict(origin=[-.12,.03,-.08], sampling_voxel=.05)
    parts = []
    for frame in range(5):
        xyz = rng.integers(-6,7,(400,3)).astype(np.float32)*.05
        normal = np.tile([0.,0.,1.],(400,1)).astype(np.float32)
        normal[::3] *= -1
        part = dict(xyz=xyz,normal=normal,sensor=xyz+normal,
            rgb=np.full((400,3),frame,np.uint8))
        if parts:
            for key in part:
                part[key][:30] = parts[0][key][:30]
            part['rgb'][:30] = frame
        parts.append(part)
    historical = unique_observations({k:np.concatenate([p[k] for p in parts]) for k in parts[0]},scale)
    streaming = StreamingObservations(scale)
    for part in parts: streaming.add(part)
    result = streaming.finish()
    for key in historical:
        np.testing.assert_array_equal(result[key],historical[key])


def test_backprojection_preserves_existing_world_gauge():
    rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    extrinsic = np.eye(4); extrinsic[:3,:3] = rotation; extrinsic[:3,3] = [.2,-.1,.7]
    depth = np.array([[1.,1.3],[.8,2.]],np.float32)
    camera = dict(Fx=13.2,Fy=12.4,Cx=.3,Cy=.6,world_view_transform=extrinsic.T)
    points,sensor = unproject(depth,camera)
    projected = points@rotation.T+extrinsic[:3,3]
    yy,xx = np.mgrid[:2,:2]
    np.testing.assert_allclose(projected[...,2],depth,atol=1e-7)
    np.testing.assert_allclose(projected[...,0]/projected[...,2]*camera['Fx']+camera['Cx'],xx,atol=2e-6)
    np.testing.assert_allclose(projected[...,1]/projected[...,2]*camera['Fy']+camera['Cy'],yy,atol=2e-6)
    np.testing.assert_allclose(rotation@sensor+extrinsic[:3,3],0,atol=1e-7)


def test_mesh_io_preserves_raw_geometry_and_open3d_compatibility(tmp_path):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals();mesh.paint_uniform_color([.2,.4,.6])
    meter = Meter(0)
    path = tmp_path/'raw.ply'
    write_mesh(path,mesh,meter)
    restored = read_mesh(path,meter)
    native = o3d.io.read_triangle_mesh(str(path))
    for actual in [restored,native]:
        np.testing.assert_array_equal(actual.vertices,mesh.vertices)
        np.testing.assert_array_equal(actual.triangles,mesh.triangles)
        np.testing.assert_allclose(actual.vertex_colors,mesh.vertex_colors,atol=1/255)
    assert meter.read_bytes == meter.write_bytes == path.stat().st_size
