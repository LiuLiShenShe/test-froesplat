"""Trace the fixed RGB brightness queries onto actual mesh triangles.

Every query retains its original pixel, native surface point and ray-hit mesh
point. Missing hits are reported separately from finite geometric statistics.
These are cross-representation consistency measurements, not scan errors.
"""
import os
os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import open3d as o3d

OUT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    dest = OUT / 'comparison/reflection_mesh_correspondence'
    dest.mkdir(exist_ok=True)
    marker = dest / (args.method + '_complete.json')
    if marker.exists():
        return
    source = OUT / 'comparison/reflection_analysis' / (args.source + '_spatial_correspondence.csv')
    rows = list(csv.DictReader(source.open()))
    index = {int(r['source_id']): r for r in json.loads((args.input / 'render_index.json').read_text())}
    assert len(index) == 208
    mesh_path = args.mesh.resolve(strict=True)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(mesh.triangles) > 0
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['source_id'])].append(row)
    records = []
    for sid, group in sorted(groups.items()):
        c = index[sid]['camera']
        E = np.asarray(c['world_view_transform'], dtype=np.float64).T
        C = np.linalg.inv(E)
        uv = np.array([[float(r['x']), float(r['y'])] for r in group])
        shifts = np.array([[0, 0], [3, 0], [-3, 0], [0, 3], [0, -3]])
        samples = (uv[:, None, :] + shifts).reshape(-1, 2)
        rays_camera = np.column_stack(((samples[:, 0] - c['Cx']) / c['Fx'],
                                       (samples[:, 1] - c['Cy']) / c['Fy'],
                                       np.ones(len(samples))))
        directions = rays_camera @ C[:3, :3].T
        origins = np.broadcast_to(C[:3, 3], directions.shape)
        hit = scene.cast_rays(o3d.core.Tensor(np.c_[origins, directions].astype(np.float32)), nthreads=8)
        depth = hit['t_hit'].numpy().reshape(-1, 5)
        normals = hit['primitive_normals'].numpy().reshape(-1, 5, 3)
        face_ids = hit['primitive_ids'].numpy().reshape(-1, 5)
        valid = np.isfinite(depth) & (depth > 0)
        positions = C[:3, 3] + directions.reshape(-1, 5, 3) * depth[..., None]
        jumps = np.degrees(np.arccos(np.clip(np.abs((normals[:, 1:] * normals[:, :1]).sum(-1)), 0, 1)))
        jumps[~(valid[:, :1] & valid[:, 1:])] = np.nan
        for i, r in enumerate(group):
            native_valid = r['valid'].lower() == 'true'
            native_depth = float(r['source_depth'])
            finite_hit = bool(valid[i, 0])
            item = dict(r, mesh_method=args.method, mesh_path=str(mesh_path), mesh_hit=finite_hit,
                        mesh_face_id=int(face_ids[i, 0]) if finite_hit else '',
                        mesh_depth=float(depth[i, 0]) if finite_hit else '',
                        native_depth_valid=native_valid,
                        mesh_vs_native_relative_depth=abs(float(depth[i, 0])-native_depth)/max(native_depth, 1e-8)
                        if finite_hit and native_valid else '',
                        mesh_world_x=float(positions[i, 0, 0]) if finite_hit else '',
                        mesh_world_y=float(positions[i, 0, 1]) if finite_hit else '',
                        mesh_world_z=float(positions[i, 0, 2]) if finite_hit else '',
                        mesh_normal_world_x=float(normals[i, 0, 0]) if finite_hit else '',
                        mesh_normal_world_y=float(normals[i, 0, 1]) if finite_hit else '',
                        mesh_normal_world_z=float(normals[i, 0, 2]) if finite_hit else '',
                        mesh_local_normal_jump3px_deg=float(np.nanmean(jumps[i]))
                        if np.isfinite(jumps[i]).any() else '',
                        mesh_neighbor_hit_fraction3px=float(valid[i, 1:].mean()))
            records.append(item)
    target = dest / (args.method + '_spatial_correspondence.csv')
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for candidate in [True, False]:
        group = [r for r in records if r['brightness_candidate'].lower() == str(candidate).lower()]
        stats = {}
        for key in ['mesh_vs_native_relative_depth', 'mesh_local_normal_jump3px_deg']:
            values = [float(r[key]) for r in group if r[key] != '' and np.isfinite(float(r[key]))]
            stats[key + '_median'] = float(np.median(values)) if values else None
            stats[key + '_p90'] = float(np.quantile(values, .9)) if values else None
            stats[key + '_finite_queries'] = len(values)
        summary.append(dict(brightness_candidate=candidate, queries=len(group),
                            mesh_hit_fraction=sum(r['mesh_hit'] for r in group)/max(len(group), 1), **stats))
    marker.write_text(json.dumps(dict(status='complete', method=args.method, source=args.source,
        mesh=str(mesh_path), vertices=len(mesh.vertices), faces=len(mesh.triangles), rows=len(records),
        CSV=str(target.resolve()), summary=summary,
        interpretation='Fixed image-only RGB matches, native rendered source points, and actual ray-hit triangle positions in the same saved W2C gauge. Brightness change is only a reflection candidate. A missing ray hit may be an open edge, incomplete surface or occlusion: it is not automatically an internal 3D hole. Local triangle-normal changes include true curvature. Relative depth is mesh-to-native consistency, not scan accuracy.'), indent=2))
    print(args.method, 'actual spatial mesh correspondence', len(records), flush=True)


if __name__ == '__main__':
    main()
