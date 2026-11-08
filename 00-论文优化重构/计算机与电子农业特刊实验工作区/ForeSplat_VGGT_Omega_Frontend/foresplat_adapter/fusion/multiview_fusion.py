"""Confidence-weighted, normal-aware, edge-preserving multi-view fusion.

Candidate points are the per-view foreground pixels straight out of the
VGGT-Omega point maps - **no pre-downsampling happens before fusion**.  Every
candidate carries (XYZ, RGB, confidence, normal, surface quality, edge flag,
source view, source pixel), so the fused cloud can always be traced back to
the observations it came from.

Fusion is performed in two passes over a voxel hash:

* **core pass** - regular pixels, fused with the adaptive voxel size chosen to
  hit the requested point budget;
* **edge pass** - depth-discontinuity pixels, fused with a *finer* voxel
  (``voxel / edge_voxel_divisor``) and a larger weight, which keeps leaf
  silhouettes dense instead of averaging them into the neighbouring surface.

The voxel key is extended with a **spherical normal bin**.  Two surfaces that
share a voxel but not an orientation (the two sides of a thin leaf, two
overlapping leaves) therefore land in different fused points and are never
merged away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .adaptive_voxel import estimate_nn_scale, voxel_keys
from .observation_mapping import build_observation_table

__all__ = ["FusionConfig", "fuse_views", "normal_bin_ids", "fused_keys"]


@dataclass
class FusionConfig:
    """Parameters of the multi-view fusion."""

    target_points: int = 1_500_000
    min_points: int = 1_000_000
    max_points: int = 2_000_000
    core_fraction: float = 0.8           # share of the budget given to the core pass
    azimuth_bins: int = 24               # normal quantisation (spherical grid)
    elevation_bins: int = 12
    edge_voxel_divisor: float = 2.0      # edge pass uses voxel / divisor
    edge_weight_boost: float = 2.0
    quality_weight_floor: float = 0.25   # w = conf * (floor + quality)
    nn_alpha: float = 1.0                # voxel >= alpha * median 1-NN distance
    seed: int = 0
    extra: dict = field(default_factory=dict)


def normal_bin_ids(normals: np.ndarray, azimuth_bins: int = 24, elevation_bins: int = 12) -> np.ndarray:
    """Spherical quantisation of unit normals into deterministic bin ids."""
    n = np.asarray(normals, dtype=np.float32)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    unit = n / np.maximum(norm, 1e-8)
    az = np.arctan2(unit[:, 1], unit[:, 0])              # [-pi, pi]
    el = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))       # [-pi/2, pi/2]
    az_bin = np.floor((az + np.pi) / (2 * np.pi) * azimuth_bins).astype(np.int64)
    el_bin = np.floor((el + np.pi / 2) / np.pi * elevation_bins).astype(np.int64)
    az_bin = np.clip(az_bin, 0, azimuth_bins - 1)
    el_bin = np.clip(el_bin, 0, elevation_bins - 1)
    return az_bin * elevation_bins + el_bin


def fused_keys(points: np.ndarray, normals: np.ndarray, voxel_size: float, cfg: FusionConfig) -> np.ndarray:
    """Voxel key extended with the normal bin (orientation-aware grouping)."""
    return voxel_keys(points, voxel_size) * (cfg.azimuth_bins * cfg.elevation_bins) + normal_bin_ids(
        normals, cfg.azimuth_bins, cfg.elevation_bins
    )


def _choose_voxel(points: np.ndarray, normals: np.ndarray, target: int, nn_scale: float, cfg: FusionConfig) -> dict:
    """Binary-search the voxel size whose orientation-aware fused count hits ``target``."""
    extent = float(np.ptp(points, axis=0).max())
    seed_voxel = extent / max(target, 1) ** (1.0 / 3.0)
    voxel = max(seed_voxel, cfg.nn_alpha * nn_scale)

    def count(v: float) -> int:
        return int(np.unique(fused_keys(points, normals, v, cfg)).size)

    low, high = voxel / 3.0, voxel * 3.0
    floor = max(cfg.nn_alpha * nn_scale, 1e-6)
    expected = count(voxel)
    for _ in range(10):
        if abs(expected - target) <= 0.06 * target:
            break
        # Count decreases as the voxel grows: too many points -> coarser voxel.
        if expected > target:
            low = voxel
        else:
            high = voxel
        voxel = max(float(np.sqrt(low * high)), floor)
        expected = count(voxel)
    return {
        "voxel_size": float(voxel),
        "nn_scale": float(nn_scale),
        "expected_fused_points": int(expected),
        "scene_extent": extent,
    }


def _collect_candidates(views: list[dict]) -> dict:
    """Flatten every view's valid pixels into candidate arrays (no pre-downsampling)."""
    parts: dict[str, list[np.ndarray]] = {
        "points": [], "colors": [], "conf": [], "normals": [], "quality": [],
        "edge": [], "view": [], "pixel": [],
    }
    for view_index, view in enumerate(views):
        valid = np.asarray(view["valid"], dtype=bool)
        if not valid.any():
            continue
        h, w = valid.shape
        flat_idx = np.flatnonzero(valid.ravel()).astype(np.int32)
        parts["points"].append(np.asarray(view["points"], dtype=np.float32).reshape(-1, 3)[flat_idx])
        colors = np.asarray(view["colors"], dtype=np.float32).reshape(-1, 3)[flat_idx]
        parts["colors"].append(np.clip(colors, 0.0, 1.0))
        parts["conf"].append(np.asarray(view["conf"], dtype=np.float32).ravel()[flat_idx])
        parts["normals"].append(np.asarray(view["normals"], dtype=np.float32).reshape(-1, 3)[flat_idx])
        parts["quality"].append(np.asarray(view["quality"], dtype=np.float32).ravel()[flat_idx])
        parts["edge"].append(np.asarray(view["edge"], dtype=bool).ravel()[flat_idx])
        parts["view"].append(np.full(flat_idx.size, view_index, dtype=np.int32))
        parts["pixel"].append(flat_idx)
        del h, w

    if not parts["points"]:
        raise ValueError("No valid foreground candidates to fuse")
    return {key: np.concatenate(value) for key, value in parts.items()}


def _aggregate(
    candidates: dict,
    voxel_size: float,
    cfg: FusionConfig,
    view_ids: np.ndarray,
    offset: int,
    weights: np.ndarray,
) -> dict:
    """Fuse one pass of candidates into weighted per-voxel representatives."""
    points = candidates["points"]
    normals = candidates["normals"]
    keys = fused_keys(points, normals, voxel_size, cfg)
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    uniq, first, inverse = np.unique(keys_sorted, return_index=True, return_inverse=True)
    # ``inverse`` is indexed by *sorted* position; scatter it back so ``group``
    # lines up with the candidate arrays (weights, colors, pixels, ...).
    group = np.empty(inverse.shape, dtype=inverse.dtype)
    group[order] = inverse
    group = group.astype(np.int64)
    del first

    quality = candidates["quality"]
    edge = candidates["edge"]
    weight = np.asarray(weights, dtype=np.float64)

    n_groups = uniq.size
    w_sum = np.bincount(group, weights=weight, minlength=n_groups).astype(np.float64)
    safe_w = np.maximum(w_sum, 1e-12)

    pos = np.stack(
        [np.bincount(group, weights=weight * points[:, i], minlength=n_groups) / safe_w for i in range(3)],
        axis=1,
    )
    rgb = np.stack(
        [np.bincount(group, weights=weight * candidates["colors"][:, i], minlength=n_groups) / safe_w for i in range(3)],
        axis=1,
    )
    nrm = np.stack(
        [np.bincount(group, weights=weight * normals[:, i], minlength=n_groups) / safe_w for i in range(3)],
        axis=1,
    )
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)

    conf_mean = np.bincount(group, weights=weight * candidates["conf"], minlength=n_groups) / safe_w
    quality_mean = np.bincount(group, weights=weight * quality, minlength=n_groups) / safe_w
    count = np.bincount(group, minlength=n_groups).astype(np.int32)
    edge_any = np.zeros(n_groups, dtype=bool)
    edge_any[group[edge]] = True

    # Representative observation = largest weight inside the group.
    tiebreak = weight - 1e-9 * candidates["view"]
    order2 = np.lexsort((tiebreak, group))
    last = np.zeros(n_groups, dtype=np.int64)
    last[group[order2]] = order2
    rep_view_local = candidates["view"][last]
    rep_pixel = candidates["pixel"][last]

    return {
        "xyz": pos.astype(np.float32),
        "colors": (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8),
        "normals": nrm.astype(np.float32),
        "confidence": conf_mean.astype(np.float32),
        "quality": quality_mean.astype(np.float32),
        "weight": w_sum.astype(np.float32),
        "observation_count": count,
        "is_edge": edge_any,
        "voxel_size": float(voxel_size),
        "rep_view": view_ids[rep_view_local],
        "rep_pixel": rep_pixel,
        "group": group,
        "keys": uniq,
        "offset": offset,
    }


def fuse_views(views: list[dict], cfg: FusionConfig | None = None) -> dict:
    """Fuse the per-view candidate clouds into one provenance-tracked cloud.

    ``views`` entries must provide ``points`` (H,W,3), ``valid`` (H,W),
    ``colors`` (H,W,3 float 0..1), ``conf`` (H,W), ``normals`` (H,W,3),
    ``quality`` (H,W), ``edge`` (H,W) and ``image_id`` (int).
    """
    cfg = cfg or FusionConfig()
    t_start = time.time()
    candidates = _collect_candidates(views)
    n_candidates = int(candidates["points"].shape[0])
    view_ids = np.array([int(v["image_id"]) for v in views], dtype=np.int32)

    nn_scale = estimate_nn_scale(candidates["points"], seed=cfg.seed)
    weight_base = candidates["conf"] * (cfg.quality_weight_floor + candidates["quality"])
    target = int(cfg.target_points)
    core_target = max(1, int(round(target * cfg.core_fraction)))
    edge_target = max(1, target - core_target)

    # Degenerate case: fewer candidates than the budget.  Shrink the voxel to
    # the local point spacing so candidates are *not* merged - never pad.
    if n_candidates <= target:
        base_voxel = max(nn_scale * 0.5, 1e-6)
        core_choice = {"voxel_size": base_voxel, "nn_scale": nn_scale,
                       "expected_fused_points": n_candidates, "scene_extent": float(np.ptp(candidates["points"], axis=0).max())}
    else:
        core_mask = ~candidates["edge"]
        if core_mask.sum() < 1000:
            core_mask = np.ones_like(core_mask, dtype=bool)
            candidates = {**candidates, "edge": np.zeros_like(candidates["edge"], dtype=bool)}
        core_candidates = {k: v[core_mask] for k, v in candidates.items()}
        core_choice = _choose_voxel(core_candidates["points"], core_candidates["normals"], core_target, nn_scale, cfg)

    core_mask = ~candidates["edge"]
    if core_mask.sum() == 0:
        core_mask = np.ones_like(core_mask, dtype=bool)
    core = _aggregate(
        {k: v[core_mask] for k, v in candidates.items()}, core_choice["voxel_size"], cfg, view_ids,
        offset=0, weights=weight_base[core_mask],
    )

    edge_mask = candidates["edge"]
    if edge_mask.any() and core_choice["expected_fused_points"] > 0:
        edge_voxel = core_choice["voxel_size"] / max(cfg.edge_voxel_divisor, 1.0)
        edge = _aggregate(
            {k: v[edge_mask] for k, v in candidates.items()}, edge_voxel, cfg, view_ids,
            offset=int(core["xyz"].shape[0]), weights=weight_base[edge_mask] * cfg.edge_weight_boost,
        )
    else:
        empty = np.zeros((0, 3), dtype=np.float32)
        edge = {
            "xyz": empty, "colors": np.zeros((0, 3), dtype=np.uint8), "normals": empty,
            "confidence": np.zeros(0, dtype=np.float32), "quality": np.zeros(0, dtype=np.float32),
            "weight": np.zeros(0, dtype=np.float32), "observation_count": np.zeros(0, dtype=np.int32),
            "is_edge": np.zeros(0, dtype=bool), "voxel_size": 0.0,
            "rep_view": np.zeros(0, dtype=np.int32), "rep_pixel": np.zeros(0, dtype=np.int64),
            "group": np.zeros(0, dtype=np.int64), "keys": np.zeros(0, dtype=np.int64), "offset": 0,
        }

    fused = {k: np.concatenate([core[k], edge[k]]) for k in
             ("xyz", "colors", "normals", "confidence", "quality", "weight",
              "observation_count", "is_edge", "rep_view", "rep_pixel")}

    # Hard cap: never exceed max_points.  Only the *edge* representatives (the
    # surplus density) are thinned - the core pass always survives in full.
    n_core = int(core["xyz"].shape[0])
    n_total = int(fused["xyz"].shape[0])
    thinned = 0
    if n_total > cfg.max_points and n_total > n_core:
        rng = np.random.default_rng(cfg.seed)
        keep_extra = max(0, cfg.max_points - n_core)
        n_extra = n_total - n_core
        if keep_extra < n_extra:
            keep_idx = rng.choice(n_extra, size=keep_extra, replace=False)
            keep_idx = np.sort(keep_idx)
            for key in ("xyz", "colors", "normals", "confidence", "quality", "weight",
                        "observation_count", "is_edge", "rep_view", "rep_pixel"):
                fused[key] = np.concatenate([fused[key][:n_core], fused[key][n_core:][keep_idx]])
            thinned = int(n_extra - keep_extra)

    # Candidate -> final fused id.  The core pass is never thinned, so its
    # group ids map directly.  When the budget cap thins the edge pass, the
    # surviving edge groups are re-indexed and the observations of *removed*
    # edge groups are dropped from the provenance table (they belong to no
    # surviving fused point and must not be mis-attributed).
    fused_gid = np.empty(n_candidates, dtype=np.int64)
    fused_gid[core_mask] = core["group"]
    obs_keep = np.ones(n_candidates, dtype=bool)
    if edge["xyz"].shape[0] > 0:
        edge_group = edge["group"]
        if thinned > 0:
            # keep_idx (sorted) selected edge groups; rebuild their final ids.
            n_extra = int(edge_group.max()) + 1
            edge_new = np.full(n_extra, -1, dtype=np.int64)
            edge_new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
            mapped = edge_new[edge_group]
            obs_keep[edge_mask] = mapped >= 0
            fused_gid[edge_mask] = n_core + np.maximum(mapped, 0)
        else:
            fused_gid[edge_mask] = n_core + edge_group

    observations = build_observation_table(
        fused_point_id=fused_gid[obs_keep],
        source_image_id=view_ids[candidates["view"]][obs_keep],
        view_index=candidates["view"][obs_keep],
        source_pixel=candidates["pixel"][obs_keep],
        source_confidence=candidates["conf"][obs_keep],
        source_normal=candidates["normals"][obs_keep],
        source_edge=candidates["edge"][obs_keep],
        source_weight=(weight_base * np.where(candidates["edge"], cfg.edge_weight_boost, 1.0))[obs_keep],
        num_fused_points=int(fused["xyz"].shape[0]),
    )

    stats = {
        "num_candidates": n_candidates,
        "num_fused_points": int(fused["xyz"].shape[0]),
        "core_points": n_core,
        "edge_points": int(n_total - n_core),
        "thinned_edge_points": thinned,
        "observations_kept": int(obs_keep.sum()),
        "observations_dropped": int((~obs_keep).sum()),
        "core_voxel_size": float(core_choice["voxel_size"]),
        "edge_voxel_size": float(edge["voxel_size"]),
        "nn_scale": float(nn_scale),
        "scene_extent": float(core_choice["scene_extent"]),
        "mean_observations_per_point": float(observations["fused_point_id"].size / max(fused["xyz"].shape[0], 1)),
        "target_points": int(cfg.target_points),
        "min_points": int(cfg.min_points),
        "max_points": int(cfg.max_points),
        "below_target": bool(fused["xyz"].shape[0] < cfg.target_points),
        "below_min_points": bool(fused["xyz"].shape[0] < cfg.min_points),
        "elapsed_seconds": round(time.time() - t_start, 3),
    }
    return {"fused": fused, "observations": observations, "stats": stats}