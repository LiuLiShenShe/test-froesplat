import unittest

import numpy as np

from scripts.phenotype_core import (
    MeasurementBlocked,
    canopy_spans,
    height_along_axis,
    leaf_length,
    leaf_width,
    metric_length,
    normalize_axis,
    pca_extents,
    post_vertices_subset_of_raw,
)


class PhenotypeCoreTests(unittest.TestCase):
    def test_known_box_extents_and_height(self):
        points = np.array([[x, y, z] for x in (0, 2) for y in (0, 3) for z in (0, 4)])
        center, _, extents = pca_extents(points)
        self.assertTrue(np.allclose(center, [1, 1.5, 2]))
        self.assertTrue(np.allclose(extents, [4, 3, 2]))
        self.assertEqual(height_along_axis(points, [0, 0, 0], [0, 0, 2]), 4.0)

    def test_rotation_invariant_pca_extents(self):
        points = np.array([[x, y, z] for x in (0, 2) for y in (0, 3) for z in (0, 4)])
        angle = 0.73
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        _, _, before = pca_extents(points)
        _, _, after = pca_extents(points @ rotation.T)
        self.assertTrue(np.allclose(before, after))

    def test_height_and_canopy_are_rotation_equivariant(self):
        points = np.array([
            [-1.0, -2.0, 0.0], [2.0, 1.0, 0.5], [0.5, -0.5, 4.0],
        ])
        base = np.array([0.0, 0.0, 0.0])
        vertical, axis_a, axis_b = np.eye(3)
        angle = 0.41
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        height = height_along_axis(points, base, vertical)
        spans = canopy_spans(points, vertical, axis_a, axis_b)
        rotated_height = height_along_axis(
            points @ rotation.T, base @ rotation.T, vertical @ rotation.T
        )
        rotated_spans = canopy_spans(
            points @ rotation.T,
            vertical @ rotation.T,
            axis_a @ rotation.T,
            axis_b @ rotation.T,
        )
        self.assertAlmostEqual(height, rotated_height)
        self.assertTrue(np.allclose(spans, rotated_spans))

    def test_scale_linear_and_missing_scale_blocks_cm(self):
        self.assertEqual(metric_length(2.5, 10.0, "calibration.csv"), 25.0)
        self.assertEqual(
            metric_length(2.5, 10.0, "calibration.csv"),
            metric_length(7.5, 10.0 / 3.0, "calibration.csv"),
        )
        with self.assertRaisesRegex(MeasurementBlocked, "PHYSICAL_SCALE_NOT_AVAILABLE"):
            metric_length(2.5, None, None)

    def test_invalid_vertical_axis_blocks_formal_height(self):
        with self.assertRaisesRegex(MeasurementBlocked, "VERTICAL_AXIS_MUST_BE_NONZERO"):
            height_along_axis(np.eye(3), [0, 0, 0], [0, 0, 0])
        with self.assertRaisesRegex(MeasurementBlocked, "VERTICAL_AXIS_MUST_BE_A_FINITE_3_VECTOR"):
            normalize_axis([0, float("nan"), 1], "vertical_axis")

    def test_canopy_axes_must_be_orthogonal(self):
        points = np.array([[0, 0, 0], [2, 3, 4]])
        spans = canopy_spans(points, [0, 0, 1], [1, 0, 0], [0, 1, 0])
        self.assertEqual(spans, (2.0, 3.0, 3.0))
        with self.assertRaisesRegex(MeasurementBlocked, "CANOPY_AXES_MUST_BE_MUTUALLY_ORTHOGONAL"):
            canopy_spans(points, [0, 0, 1], [1, 0, 0], [1, 1, 0])

    def test_leaf_identity_is_required(self):
        with self.assertRaisesRegex(MeasurementBlocked, "LEAF_ID_NOT_AVAILABLE"):
            leaf_length(None, [0, 0, 0], [0, 0, 2])
        with self.assertRaisesRegex(MeasurementBlocked, "LEAF_ID_NOT_AVAILABLE"):
            leaf_width("LEAF_ID_NOT_AVAILABLE", [[0, 0, 0], [0, 2, 0]], [0, 1, 0])
        self.assertEqual(leaf_length("L01", [0, 0, 0], [0, 0, 2]), 2.0)
        self.assertEqual(leaf_width("L01", [[0, 0, 0], [0, 2, 0]], [0, 1, 0]), 2.0)

    def test_post_mesh_identity_is_a_coordinate_subset(self):
        raw = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        post = raw[[0, 2]]
        self.assertTrue(post_vertices_subset_of_raw(raw, post))
        self.assertFalse(post_vertices_subset_of_raw(raw, np.array([[2, 2, 2]], dtype=np.float64)))


if __name__ == "__main__":
    unittest.main()
