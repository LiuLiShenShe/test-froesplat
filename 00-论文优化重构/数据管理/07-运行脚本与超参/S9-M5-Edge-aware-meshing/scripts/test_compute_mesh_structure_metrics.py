#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_mesh_structure_metrics import (  # noqa: E402
    compute_displacement_metrics,
    find_standard_post_mesh,
    compute_topology_metrics,
)


def make_two_triangle_components() -> o3d.geometry.TriangleMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


class MeshStructureMetricsTest(unittest.TestCase):
    def test_topology_metrics_count_components_and_isolated_vertices(self) -> None:
        metrics = compute_topology_metrics(make_two_triangle_components(), small_component_triangle_threshold=2)

        self.assertEqual(metrics["connected_components"], 2)
        self.assertEqual(metrics["largest_component_vertices"], 3)
        self.assertAlmostEqual(metrics["largest_component_ratio"], 3 / 7)
        self.assertEqual(metrics["small_component_count"], 2)
        self.assertEqual(metrics["isolated_vertices"], 1)
        self.assertEqual(metrics["boundary_edge_count"], 6)
        self.assertEqual(metrics["non_manifold_edge_count"], 0)

    def test_displacement_metrics_include_p95_for_matching_meshes(self) -> None:
        source = make_two_triangle_components()
        moved = o3d.geometry.TriangleMesh(source)
        moved_vertices = np.asarray(moved.vertices).copy()
        moved_vertices[:, 0] += np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10])
        moved.vertices = o3d.utility.Vector3dVector(moved_vertices)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.ply"
            moved_path = Path(tmpdir) / "moved.ply"
            o3d.io.write_triangle_mesh(str(source_path), source)
            o3d.io.write_triangle_mesh(str(moved_path), moved)

            metrics = compute_displacement_metrics(source_path, moved_path)

        self.assertAlmostEqual(metrics["mean_displacement"], float(np.mean([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10])))
        self.assertAlmostEqual(metrics["p95_displacement"], float(np.percentile([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10], 95)))
        self.assertAlmostEqual(metrics["max_displacement"], 0.10)

    def test_find_standard_post_mesh_uses_sample_variant_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            standard_dir = root / "Sample_A6_M1_soft_M4_standard"
            standard_dir.mkdir(parents=True)
            expected = standard_dir / "fuse_post.ply"
            o3d.io.write_triangle_mesh(str(expected), make_two_triangle_components())

            self.assertEqual(find_standard_post_mesh(root, "Sample"), expected)


if __name__ == "__main__":
    unittest.main()
