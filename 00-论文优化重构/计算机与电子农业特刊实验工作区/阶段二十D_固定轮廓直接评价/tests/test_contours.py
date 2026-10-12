from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rasterize_contours import rasterize, round_vertices


def annotation(shapes):
    return {"imageWidth": 20, "imageHeight": 20, "shapes": shapes}


def shape(points, label="potted_plant"):
    return {"label": label, "shape_type": "linestrip", "points": points}


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_closure_preserves_original(rasterizer):
    source = annotation([shape([[2.5, 2.5], [14.4, 2.5], [14.4, 14.4], [2.5, 14.4]])])
    original = copy.deepcopy(source)
    target, mapping, counts = rasterize(source, rasterizer, 20, 20)
    assert source == original
    assert target[10, 10] and not target[1, 1]
    assert mapping[0]["original_points"] == original["shapes"][0]["points"]
    assert mapping[0]["rounded_points"][0] == [3, 3]
    assert counts["potted_plant"] == 1


def test_half_away_from_zero():
    points = [[-1.5, -2.5], [1.5, 0.5], [0.49, -0.49]]
    assert round_vertices(points).tolist() == [[-2, -3], [2, 1], [0, 0]]


@pytest.mark.parametrize("points", [[], [[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [float("nan"), 2], [2, 0]], [[0, 0], [float("inf"), 2], [2, 0]],
    [[0, 0], [0.1, 0], [0, 0.1]], [[0, 0], [1e20, 2], [2, 0]], [[0, 0], ["bad", 2], [2, 0]]])
def test_invalid_points_fail(points):
    with pytest.raises(ValueError):
        round_vertices(points)


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_bow_tie_not_rejected_for_zero_signed_area(rasterizer):
    source = annotation([shape([[2, 2], [16, 16], [2, 16], [16, 2]])])
    target, mapping, counts = rasterize(source, rasterizer, 20, 20)
    assert target.any()
    assert len(mapping[0]["rounded_points"]) == 4


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_overlapping_shapes_are_union_not_xor(rasterizer):
    source = annotation([shape([[2, 2], [12, 2], [12, 12], [2, 12]]),
                         shape([[6, 6], [16, 6], [16, 16], [6, 16]])])
    target, mapping, counts = rasterize(source, rasterizer, 20, 20)
    assert target[8, 8] and target[3, 3] and target[15, 15]
    assert counts["potted_plant"] == 2


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_cube_subtraction_and_component_ignorance(rasterizer):
    source = annotation([shape([[2, 2], [16, 2], [16, 16], [2, 16]]),
                         shape([[6, 6], [10, 6], [10, 10], [6, 10]], "blue_cube"),
                         shape([[0, 0]], "plant"), shape([[0, 0]], "pot")])
    target, mapping, counts = rasterize(source, rasterizer, 20, 20)
    assert target[3, 3] and not target[8, 8]
    assert len(mapping) == 2
    assert counts["blue_cube"] == 1


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_missing_or_off_canvas_gt_never_becomes_placeholder(rasterizer):
    with pytest.raises(ValueError):
        rasterize(annotation([]), rasterizer, 20, 20)
    with pytest.raises(ValueError):
        rasterize(annotation([shape([[100, 100], [110, 100], [100, 110]])]), rasterizer, 20, 20)


@pytest.mark.parametrize("rasterizer", ["opencv", "pillow"])
def test_grid_mismatch_fails(rasterizer):
    with pytest.raises(ValueError):
        rasterize(annotation([shape([[2, 2], [10, 2], [2, 10]])]), rasterizer)
