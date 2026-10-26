#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contracts for the mesh + exploratory phenotype step (manuscript §2.8).

These guard the two things that could silently falsify the numbers:

  M1  the meshing protocol stays the frozen S19 standard TSDF (voxel 0.02 /
      depth_trunc 6.0 / sdf_trunc 0.08 / mesh_res 256), so results stay
      comparable to the archived S19 mesh table;
  M2  the "up" axis really comes from the camera poses and the derived
      horizontal basis is orthonormal — otherwise the height/span columns
      would be measured along an arbitrary direction;
  M3  the physical-scale gate is closed: the module exposes no metric scale
      for KongQueZhuYu, so nothing downstream can report cm.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/"
               "ForeSplat_Original_Method_Recovery/scripts/run_mesh_phenotype.py"))
MOD_NAME = "mesh_phenotype_contracts"


@pytest.fixture(scope="module")
def mod():
    if MOD_NAME in sys.modules:
        return sys.modules[MOD_NAME]
    spec = importlib.util.spec_from_file_location(MOD_NAME, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MOD_NAME] = module
    spec.loader.exec_module(module)
    return module


def test_meshing_protocol_is_frozen(mod):
    args = mod.MESH_ARGS
    assert "--skip_train" in args and "--skip_test" in args
    assert "--meshing_mode" in args and args[args.index("--meshing_mode") + 1] == "standard"
    pairs = dict(zip(args[::2], args[1::2]))
    assert float(pairs["--voxel_size"]) == 0.02
    assert float(pairs["--depth_trunc"]) == 6.0
    assert float(pairs["--sdf_trunc"]) == 0.08
    assert int(pairs["--mesh_res"]) == 256
    assert int(pairs["--num_cluster"]) == 20
    assert mod.ITERATION == 30000


def test_camera_up_and_horizontal_basis(mod):
    identity = {"rotation": np.eye(3).tolist()}
    up = mod.up_axis_from_cameras([identity])
    # world->camera rows; image y points down, so camera up is world -y
    assert np.allclose(up, [0.0, -1.0, 0.0], atol=1e-9)

    # a camera pitched 90 deg about x: its up now points along world +z
    pitch = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    up2 = mod.up_axis_from_cameras([{"rotation": pitch.tolist()}])
    assert np.allclose(up2, [0.0, 0.0, 1.0], atol=1e-9)

    for vertical in (up, up2, np.array([0.3, 0.5, -0.8]) / np.linalg.norm([0.3, 0.5, -0.8])):
        a, b = mod.horizontal_basis(vertical)
        basis = np.stack([vertical, a, b])
        assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)


def test_physical_scale_gate_is_closed(mod):
    assert mod.__dict__.get("PHYSICAL_SCALE_CM_PER_UNIT") is None
    source = SCRIPT.read_text(encoding="utf-8")
    assert "PHYSICAL_SCALE_NOT_AVAILABLE" in source
    # no hard-coded metric factor may sneak in
    assert "cm_per_unit\":" not in source.replace("physical_scale_cm_per_unit", "")
