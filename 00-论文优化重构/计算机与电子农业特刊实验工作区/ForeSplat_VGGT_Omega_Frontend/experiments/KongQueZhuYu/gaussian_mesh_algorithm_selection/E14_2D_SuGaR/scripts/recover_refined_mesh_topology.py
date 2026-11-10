"""Restore coincident OBJ UV-seam vertices without changing any triangle geometry."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'refinement'
MARKER = ROOT / 'refinement_complete.json'
info = json.loads(MARKER.read_text())
if info.get('exact_vertex_weld_verified'):
    raise SystemExit(0)
started = time.time()
original = Path(info['refined_raw_ply'])
mesh = o3d.io.read_triangle_mesh(str(original))
v = np.asarray(mesh.vertices)
f = np.asarray(mesh.triangles)
assert np.isfinite(v).all() and len(f) > 0
before = hashlib.sha256(v[f].tobytes()).hexdigest()
# The native OBJ has836610 geometric vertices. Its per-face UV coordinates
# cause the Open3D OBJ importer to duplicate them into4764186 vertices.
# Unique exactXYZ changes connectivity only; no distance tolerance is used.
unique, inverse = np.unique(v, axis=0, return_inverse=True)
remapped = inverse[f]
assert np.array_equal(unique[remapped], v[f])
after = hashlib.sha256(unique[remapped].tobytes()).hexdigest()
assert before == after
welded = o3d.geometry.TriangleMesh()
welded.vertices = o3d.utility.Vector3dVector(unique)
welded.triangles = o3d.utility.Vector3iVector(remapped)
original_post = OUT / 'mesh_post'
archive = OUT / 'uv_seam_unwelded_post_attempt'
assert not archive.exists(), 'Original post attempt archive already exists'
original_post.rename(archive)
sys.path.append(str(ROOT.parent.parent / 'geometry_mesh_final_selection/scripts'))
from mesh_artifacts import save_meshes
scale = json.loads((ROOT.parent.parent / 'geometry_mesh_final_selection/configs/standard_spatial_scale.json').read_text())
stats = save_meshes(welded, original_post, scale)
assert stats['post_faces'] > 0
proof = dict(status='complete', native_textured_obj=info['textured_obj'], original_geometry_only_ply=str(original.resolve()), original_import_vertices=len(v), exact_unique_vertices=len(unique), faces_before=len(f), faces_after=len(remapped), triangle_position_sha256_before=before, triangle_position_sha256_after=after, all_triangle_positions_bitwise_unchanged=True, tolerance=0, vertex_averaging=False, smoothing=False, original_unwelded_post_attempt=str(archive.resolve()), welded_raw=str((original_post / 'raw.ply').resolve()), post=str((original_post / 'post.ply').resolve()), elapsed_seconds=time.time()-started)
proof_path = ROOT / 'refined_mesh_exact_weld_validation.json'
proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
info.update(stats)
info.update(exact_vertex_weld_verified=True, exact_vertex_weld_evidence=str(proof_path.resolve()), refined_welded_raw_ply=proof['welded_raw'], refined_post_ply=proof['post'], original_unwelded_raw_ply=str(original.resolve()), original_unwelded_post_attempt=proof['original_unwelded_post_attempt'], topology_adaptation='Open3D UV-seam duplicated vertices merged by exactXYZ only; every original triangle coordinate bitwise unchanged; native OBJ/MTL/texture preserved', elapsed_seconds=info['elapsed_seconds']+proof['elapsed_seconds'])
MARKER.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
print('Exact weld verified', len(v), '->', len(unique), 'vertices;', stats['post_faces'], 'post faces', flush=True)
