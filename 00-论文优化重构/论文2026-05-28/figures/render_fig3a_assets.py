#!/usr/bin/env python3
"""Export Fig. 3a method assets from native reconstruction outputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/data/fj/F2DMAS")
OUT_DIR = ROOT / "00-论文优化重构/论文2026-05-28/figures/fig3a_rendered_assets"
MASK_DIR = ROOT / "00-论文优化重构/数据管理/03-分割Mask/02-sam_masks/KongQueZhuYu"
FRONT_CAMERA_JSON = (
    ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4/cameras.json"
)
FRONT_VIEW_INDEX = 0


@dataclass(frozen=True)
class Asset:
    name: str
    source: Path
    note: str


COPY_ASSETS = [
    Asset(
        "colmap_row1_gt_front_view.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4/test/ours_30000/gt/00000.png",
        "Aligned front-view ground-truth image for the COLMAP column.",
    ),
    Asset(
        "colmap_row2_gt_front_view.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4/test/ours_30000/gt/00000.png",
        "Aligned front-view ground-truth image repeated for the foreground row; COLMAP has no native foreground-only render.",
    ),
    Asset(
        "3dgs_fsam3_row3_sugar_mesh_preview.png",
        ROOT / "07-SuGaR-Mesh/KongQueZhuYu/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.png",
        "Existing SuGaR/3DGS textured mesh preview.",
    ),
    Asset(
        "standard_2dgs_row1_scene_render.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/test/ours_30000/renders/00000.png",
        "Standard 2DGS test render.",
    ),
    Asset(
        "foresplat_row1_scene_render.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4/test/ours_30000/renders/00000.png",
        "ForeSplat A6+M1+M4 test render.",
    ),
    Asset(
        "foresplat_row2_foreground_render.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_foreground_track_init_fg_rgb_alpha_bg/test/ours_30000/renders/00000.png",
        "ForeSplat foreground-only render.",
    ),
    Asset(
        "3dgs_fsam3_row1_e3_scene_render.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E3_fsam3_preprocess/test/ours_30000/renders/00000.png",
        "2DGS experiment using FSAM3 preprocessing; native 3DGS render is blocked until CUDA extensions are installed.",
    ),
    Asset(
        "3dgs_fsam3_row2_e3_foreground_render.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E3_fsam3_preprocess/test/ours_30000/renders/00000.png",
        "Foreground-aligned E3 FSAM3-preprocess render; black background reflects masked/foreground preprocessing.",
    ),
]


MESH_ASSETS = [
    (
        "colmap_row3_fuse_pointcloud.png",
        ROOT / "11-COLMAP-fuse/run_20260302_153856/KongQueZhuYu/fuse.ply",
        "COLMAP stereo_fusion point cloud rendered as a colored point cloud.",
        "point",
    ),
    (
        "standard_2dgs_row3_tsdf_mesh.png",
        ROOT / "05-2DGS-new/KongQueZhuYu/train/ours_30000/fuse_post.ply",
        "Standard 2DGS TSDF mesh rendered from fuse_post.ply.",
        "mesh",
    ),
    (
        "foresplat_row3_tsdf_mesh.png",
        ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4/train/ours_30000/fuse_post.ply",
        "ForeSplat A6+M1+M4 TSDF mesh rendered from fuse_post.ply.",
        "mesh",
    ),
    (
        "3dgs_fsam3_row3_sugar_textured_mesh.png",
        ROOT / "07-SuGaR-GS/KongQueZhuYu/refined_mesh/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj",
        "SuGaR/3DGS textured mesh rendered from OBJ.",
        "mesh",
    ),
]


def copy_asset(asset: Asset) -> dict[str, str]:
    if not asset.source.exists():
        return {"name": asset.name, "source": str(asset.source), "status": "missing", "note": asset.note}
    dst = OUT_DIR / asset.name
    shutil.copy2(asset.source, dst)
    return {"name": asset.name, "source": str(asset.source), "output": str(dst), "status": "copied", "note": asset.note}


def find_mask_for_render(gt_dir: Path, render_name: str) -> Path | None:
    gt_path = gt_dir / render_name
    if not gt_path.exists():
        return None
    gt = Image.open(gt_path).convert("RGB")
    candidates: list[tuple[float, Path]] = []
    for mask_path in sorted(MASK_DIR.glob("mask_*.png")) + sorted(MASK_DIR.glob("crop_*.png")):
        try:
            mask = Image.open(mask_path).convert("L").resize(gt.size, Image.Resampling.NEAREST)
        except OSError:
            continue
        arr = np.asarray(mask) > 127
        if arr.mean() < 0.01:
            continue
        # Foreground-only render row just needs a plausible same-view mask.
        candidates.append((abs(float(arr.mean()) - 0.2), mask_path))
    return min(candidates, default=(0.0, None))[1]


def make_standard_2dgs_foreground() -> dict[str, str]:
    render_path = ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/test/ours_30000/renders/00000.png"
    gt_dir = ROOT / "00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/test/ours_30000/gt"
    out_path = OUT_DIR / "standard_2dgs_row2_foreground_masked_render.png"
    if not render_path.exists():
        return {"name": out_path.name, "source": str(render_path), "status": "missing"}
    mask_path = find_mask_for_render(gt_dir, render_path.name)
    if mask_path is None:
        return {"name": out_path.name, "source": str(render_path), "status": "missing-mask"}
    image = Image.open(render_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    arr = np.asarray(image).copy()
    keep = np.asarray(mask) > 127
    arr[~keep] = 0
    Image.fromarray(arr).save(out_path)
    return {
        "name": out_path.name,
        "source": str(render_path),
        "mask": str(mask_path),
        "output": str(out_path),
        "status": "generated",
        "note": "Standard 2DGS render masked by FSAM3 mask for the foreground-only row.",
    }


def front_view_camera_position(mesh) -> tuple[list[float], list[float], list[float]]:
    camera = json.loads(FRONT_CAMERA_JSON.read_text(encoding="utf-8"))[FRONT_VIEW_INDEX]
    rotation = np.asarray(camera["rotation"], dtype=float)
    forward = rotation[:, 2]
    view_up = -rotation[:, 1]

    bounds = np.asarray(mesh.bounds, dtype=float).reshape(3, 2)
    center = bounds.mean(axis=1)
    diagonal = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
    distance = max(float(diagonal), 1.0) * 1.45
    position = center - forward * distance
    return position.tolist(), center.tolist(), view_up.tolist()


def render_geometry(src: Path, dst: Path, mode: str) -> dict[str, str]:
    if not src.exists():
        return {"name": dst.name, "source": str(src), "status": "missing"}

    import pyvista as pv

    if hasattr(pv, "start_xvfb"):
        pv.start_xvfb(wait=0.1)
    mesh = pv.read(src)
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1400))
    plotter.set_background("white")

    if mode == "point" or getattr(mesh, "n_cells", 0) == 0:
        if getattr(mesh, "n_points", 0) > 700_000:
            mesh = mesh.extract_points(np.linspace(0, mesh.n_points - 1, 700_000, dtype=np.int64))
        plotter.add_points(mesh, render_points_as_spheres=False, point_size=2.2, rgb=True)
    else:
        try:
            plotter.add_mesh(mesh, scalars="RGB", rgb=True, show_edges=False, smooth_shading=True)
        except Exception:
            plotter.add_mesh(mesh, show_edges=False, color="lightgray", smooth_shading=True)

    plotter.reset_camera()
    plotter.enable_parallel_projection()
    plotter.camera_position = front_view_camera_position(mesh)
    plotter.camera.zoom(1.18)
    plotter.show(screenshot=str(dst), auto_close=True)
    return {"name": dst.name, "source": str(src), "output": str(dst), "status": "rendered", "mode": mode}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for asset in COPY_ASSETS:
        manifest.append(copy_asset(asset))
    manifest.append(make_standard_2dgs_foreground())
    for name, src, note, mode in MESH_ASSETS:
        item = render_geometry(src, OUT_DIR / name, mode)
        item["note"] = note
        manifest.append(item)

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)
    for item in manifest:
        print(f"{item['status']:>12} {item['name']}")


if __name__ == "__main__":
    main()
