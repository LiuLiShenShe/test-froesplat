"""Multi-view fusion for the ForeSplat VGGT-Omega frontend.

Sub-modules:

* :mod:`normal_estimation` - per-view point-map normals, surface quality, edge flag
* :mod:`adaptive_voxel`    - scene-adaptive voxel size selection + hashing
* :mod:`multiview_fusion`  - confidence-weighted, normal-aware, edge-preserving fusion
* :mod:`observation_mapping` - fused point -> source observation provenance
"""

from .adaptive_voxel import choose_voxel_size, estimate_nn_scale, voxel_keys
from .multiview_fusion import FusionConfig, fused_keys, fuse_views, normal_bin_ids
from .normal_estimation import depth_edge_mask, estimate_view_normals
from .observation_mapping import (
    build_observation_table,
    export_observations_csv,
    load_observations_npz,
    observations_for_point,
    pixel_index_to_xy,
    save_observations_npz,
)

__all__ = [
    "FusionConfig",
    "fuse_views",
    "fused_keys",
    "normal_bin_ids",
    "estimate_view_normals",
    "depth_edge_mask",
    "choose_voxel_size",
    "estimate_nn_scale",
    "voxel_keys",
    "build_observation_table",
    "save_observations_npz",
    "load_observations_npz",
    "observations_for_point",
    "pixel_index_to_xy",
    "export_observations_csv",
]