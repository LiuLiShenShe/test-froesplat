"""ForeSplat VGGT-Omega frontend adapter.

Turns RAP-FSAM3 foreground images + binary masks into a GGGS-ready COLMAP
dataset via VGGT-Omega (camera + depth prediction), with foreground-mask
constrained geometry.
"""

from .dataset import FramePaths, collect_frames, foreground_ratios
from .inference import load_model, run_inference, unproject_depth_map_to_point_map
from .export import export_gggs_dataset

__all__ = [
    "FramePaths",
    "collect_frames",
    "foreground_ratios",
    "load_model",
    "run_inference",
    "unproject_depth_map_to_point_map",
    "export_gggs_dataset",
]
