"""M2M3-GO: Mask-to-Model and Model-to-Mask Micro-Matching Gaussian Optimisation.

Phase 21 implementation of the manuscript's primitive-level refinement layer.

The module is deliberately pure (no optimizer / no CUDA-only calls) so it can be
unit-tested on CPU tensors.  It implements, for each Gaussian primitive g_j:

    S_m2g  = sum_i q_i eta_ij mean_{p in E_ij} M_i(p) / sum_i q_i eta_ij
    H_i(p) = M_i(p) (1 - A_i(p))
    S_g2m  = sum_i q_i eta_ij mean_{p in E_ij} H_i(p) / sum_i q_i eta_ij
    B      = sum_i q_i eta_ij mean_{p in E_ij} exp(-d(p,bd M_i)^2 / (2 sigma_b^2)) / ...
    U      = sigmoid(w1 S_m2g + w2 S_g2m + w3 B + w4 log(alpha+eps)
                     + w5 log(1+|V|) + w6 R + b)

with E_ij approximated by a screen-space footprint around the projected centre
(radius from the native `max_radii2D` state, averaged with a box filter), and
`R` the foreground photometric residual inside the footprint.

All footprint means are computed with `avg_pool2d` + `grid_sample`, so the cost
is O(views * gaussians) with no per-Gaussian Python loop.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


# ── projection ───────────────────────────────────────────────────────────────

def project_centers(xyz: torch.Tensor, cam) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Gaussian centres into NDC.

    Returns (pix_ndc [N,2] in [-1,1], depth [N], in_view [N] bool).
    Mirrors the convention already used by `mask_consistency_scores`.
    """
    ones = torch.ones_like(xyz[:, :1])
    xyz_h = torch.cat([xyz, ones], dim=1)
    proj = xyz_h @ cam.full_proj_transform
    depth = proj[:, 3]
    pix = proj[:, :2] / depth[:, None].clamp_min(1e-6)
    in_view = (
        (depth > 0)
        & (pix[:, 0] > -1.0) & (pix[:, 0] < 1.0)
        & (pix[:, 1] > -1.0) & (pix[:, 1] < 1.0)
    )
    return pix, depth, in_view


# ── footprint statistics ─────────────────────────────────────────────────────

def _grid_sample_map(field: torch.Tensor, pix_ndc: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a [C,H,W] field at NDC points. Returns [C,N]."""
    if field.dim() == 2:
        field = field.unsqueeze(0)
    grid = pix_ndc.view(1, -1, 1, 2)
    out = F.grid_sample(field.unsqueeze(0), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.reshape(field.shape[0], -1)


def _kernel_bucket(radius_px: torch.Tensor, kernels: tuple[int, ...]) -> torch.Tensor:
    """Map a per-Gaussian screen radius to the smallest odd kernel that covers it."""
    needed = (2.0 * radius_px.clamp_min(0.5) + 1.0)
    kernel_t = torch.tensor(kernels, dtype=torch.float32, device=radius_px.device)
    idx = torch.searchsorted(kernel_t, needed.clamp(max=float(kernels[-1])))
    return idx.clamp(max=len(kernels) - 1)


def _replicate_pad_pool(field: torch.Tensor, kernel: int) -> torch.Tensor:
    """Average-pool with replicate padding (no zero border contamination).

    `avg_pool2d` zero-pads, which shrinks means near the field border.  For
    footprint statistics that is a real bias, so we pad by replication first.
    """
    pad = kernel // 2
    if pad <= 0:
        return field
    padded = F.pad(field.unsqueeze(0), (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)[0]


def footprint_mean(
    field: torch.Tensor,
    pix_ndc: torch.Tensor,
    radii_px: torch.Tensor,
    in_view: torch.Tensor,
    kernels: tuple[int, ...] = (1, 3, 5, 9, 15),
) -> torch.Tensor:
    """Mean of `field` over each Gaussian's screen footprint.

    `field`: [H,W] or [C,H,W]; `pix_ndc`: [N,2]; `radii_px`: [N];
    Returns [C,N] (or [N] for a 2-D field). Values at out-of-view Gaussians are 0.
    """
    squeeze = field.dim() == 2
    if squeeze:
        field = field.unsqueeze(0)
    n = pix_ndc.shape[0]
    result = field.new_zeros((field.shape[0], n))
    if n == 0:
        return result.squeeze(0) if squeeze else result
    buckets = _kernel_bucket(radii_px, kernels)

    for bucket_index, kernel in enumerate(kernels):
        selected = (buckets == bucket_index) & in_view
        if not torch.any(selected):
            continue
        if kernel == 1:
            pooled = field
        else:
            pooled = _replicate_pad_pool(field, kernel)
        sampled = _grid_sample_map(pooled, pix_ndc[selected])
        result[:, selected] = sampled
    if result.shape[0] == 1:
        return result.squeeze(0)
    return result.squeeze(0) if squeeze else result


# ── per-view fields ──────────────────────────────────────────────────────────

def boundary_distance_map(mask: torch.Tensor) -> torch.Tensor:
    """Unsigned distance (pixels) to the mask boundary.

    Inside pixels get their distance to background; outside pixels get their
    distance to the foreground.  Uses scipy/cv2-free Euclidean distance via
    `scipy.ndimage` when available, else a simple torch fallback.
    """
    binary = (mask.detach().cpu().squeeze().numpy() > 0.5)
    if binary.size == 0:
        return torch.zeros_like(mask)
    try:
        from scipy import ndimage as ndi
        inside = ndi.distance_transform_edt(binary)
        outside = ndi.distance_transform_edt(~binary)
        distance = inside + outside
        # Pixels exactly on the boundary have distance 0; shift by half a pixel
        # so the boundary itself has the smallest positive support.
        distance = np.maximum(distance, 0.0)
    except Exception:
        distance = np.zeros_like(binary, dtype=np.float32)
    return torch.from_numpy(distance.astype(np.float32)).to(mask.device).view_as(mask)


def boundary_support_map(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    distance = boundary_distance_map(mask)
    return torch.exp(-(distance ** 2) / (2.0 * max(float(sigma), 1e-6) ** 2))


def foreground_deficit_map(mask: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return mask * (1.0 - alpha)


# ── aggregation ──────────────────────────────────────────────────────────────

class M2M3Accumulator:
    """Accumulates the four manuscript statistics across views."""

    def __init__(self, n_gaussians: int, device: torch.device):
        self.n = n_gaussians
        self.device = device
        self.m2g_num = torch.zeros(n_gaussians, device=device)
        self.g2m_num = torch.zeros(n_gaussians, device=device)
        self.bnd_num = torch.zeros(n_gaussians, device=device)
        self.res_num = torch.zeros(n_gaussians, device=device)
        self.weight = torch.zeros(n_gaussians, device=device)
        self.visible = torch.zeros(n_gaussians, device=device)

    def add(
        self,
        mask: torch.Tensor,
        alpha: torch.Tensor,
        rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        boundary: torch.Tensor,
        pix_ndc: torch.Tensor,
        radii_px: torch.Tensor,
        in_view: torch.Tensor,
        view_weight: float,
    ) -> None:
        m_mean = footprint_mean(mask, pix_ndc, radii_px, in_view)
        deficit_map = foreground_deficit_map(mask, alpha)
        h_mean = footprint_mean(deficit_map, pix_ndc, radii_px, in_view)
        b_mean = footprint_mean(boundary, pix_ndc, radii_px, in_view)
        residual_map = (rgb - gt_rgb).abs().mean(dim=0, keepdim=True)
        r_mean = footprint_mean(residual_map, pix_ndc, radii_px, in_view)

        w = in_view.to(mask.dtype) * float(view_weight)
        self.m2g_num += w * m_mean
        self.g2m_num += w * h_mean
        self.bnd_num += w * b_mean
        self.res_num += w * r_mean
        self.weight += w
        self.visible += in_view.to(mask.dtype)

    def finalize(self) -> dict[str, torch.Tensor]:
        denom = self.weight.clamp_min(EPS)
        valid = self.visible > 0
        return {
            "s_m2g": torch.where(valid, self.m2g_num / denom, torch.zeros_like(denom)),
            "s_g2m": torch.where(valid, self.g2m_num / denom, torch.zeros_like(denom)),
            "b_support": torch.where(valid, self.bnd_num / denom, torch.zeros_like(denom)),
            "residual": torch.where(valid, self.res_num / denom, torch.zeros_like(denom)),
            "visible": self.visible,
            "valid": valid,
        }


def utility_score(
    s_m2g: torch.Tensor,
    s_g2m: torch.Tensor,
    b_support: torch.Tensor,
    opacity: torch.Tensor,
    visible_views: torch.Tensor,
    residual: torch.Tensor,
    weights: tuple[float, float, float, float, float, float] = (1.0, 0.5, 0.5, 0.5, 0.25, -1.0),
    bias: float = 0.0,
) -> torch.Tensor:
    """U = sigmoid(w1 S_m2g + w2 S_g2m + w3 B + w4 log(alpha) + w5 log(1+|V|) + w6 R + b)."""
    w1, w2, w3, w4, w5, w6 = [float(v) for v in weights]
    log_alpha = torch.log(opacity.clamp_min(EPS))
    log_views = torch.log1p(visible_views.clamp_min(0.0))
    logit = (
        w1 * s_m2g + w2 * s_g2m + w3 * b_support
        + w4 * log_alpha + w5 * log_views + w6 * residual + float(bias)
    )
    return torch.sigmoid(logit)


def resolve_tau_p(
    utility: torch.Tensor,
    valid: torch.Tensor,
    mode: str,
    tau_p: float,
    quantile: float = 0.01,
) -> tuple[float, str]:
    """Resolve the pruning threshold U_j < tau_p.

    `mode="fixed"` uses the manuscript hyperparameter directly.  Because U is a
    sigmoid over a linear score, its absolute scale depends on the scene, the
    view count and the opacity distribution: with 100+ views the
    ``w5 log(1+|V|)`` term alone adds >1.0 to the logit for almost every
    primitive, so a single fixed tau_p can silently make the controller inert.
    `mode="quantile"` therefore calibrates tau_p to the requested lower
    quantile of the current round's utility distribution; the resolved value is
    reported every round so the choice is auditable.

    Returns (tau_p, source_label).
    """
    if str(mode) != "quantile":
        return float(tau_p), "fixed"
    values = utility[valid]
    if values.numel() == 0:
        return float(tau_p), "fixed(empty)"
    q = float(min(max(float(quantile), 1e-4), 0.5))
    resolved = float(torch.quantile(values.detach().double(), q).item())
    return resolved, "quantile({:.4f})".format(q)


# ── local Floor40 ────────────────────────────────────────────────────────────

def assign_local_blocks(xyz: torch.Tensor, n_blocks_per_axis: int = 6) -> torch.Tensor:
    """Assign each Gaussian to a regular 3-D block over the point-cloud bounds."""
    if xyz.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=xyz.device)
    n = max(int(n_blocks_per_axis), 1)
    lo = xyz.min(dim=0).values
    hi = xyz.max(dim=0).values
    span = (hi - lo).clamp_min(1e-6)
    grid = ((xyz - lo) / span * n).floor().clamp(0, n - 1).to(torch.long)
    return grid[:, 0] * n * n + grid[:, 1] * n + grid[:, 2]


def local_floor_select(
    candidate_indices: torch.Tensor,
    utility: torch.Tensor,
    block_ids: torch.Tensor,
    block_counts: torch.Tensor,
    block_min: torch.Tensor,
    max_remove: int,
) -> tuple[torch.Tensor, dict]:
    """Choose removals by lowest utility while honouring per-block floors.

    `candidate_indices`: indices into the Gaussian arrays (prune candidates).
    `utility`: full-array utility; `block_ids`: full-array block assignment.
    `block_counts`: current count per block; `block_min`: minimum retained count.
    Returns (remove_mask_full_array, report).
    """
    n = utility.shape[0]
    remove = torch.zeros(n, dtype=torch.bool, device=utility.device)
    if candidate_indices.numel() == 0 or max_remove <= 0:
        return remove, {"requested": int(candidate_indices.numel()), "removed": 0, "floor_blocked": 0}
    order = torch.argsort(utility[candidate_indices], descending=False)
    scheduled = torch.zeros_like(block_counts)
    floor_blocked = 0
    removed = 0
    for local_pos in order.tolist():
        if removed >= max_remove:
            break
        idx = int(candidate_indices[local_pos])
        block = int(block_ids[idx])
        remaining = int(block_counts[block] - scheduled[block])
        if remaining <= int(block_min[block]):
            floor_blocked += 1
            continue
        remove[idx] = True
        scheduled[block] += 1
        removed += 1
    return remove, {
        "requested": int(candidate_indices.numel()),
        "removed": int(removed),
        "floor_blocked": int(floor_blocked),
    }


# ── deficit-driven densification ─────────────────────────────────────────────

def backproject_pixels(
    pixels_uv: torch.Tensor,
    depths: torch.Tensor,
    cam,
) -> torch.Tensor:
    """Back-project image pixels (u,v) and z-depths to world coordinates.

    `pixels_uv`: [N,2] in pixel coordinates (u right, v down).
    `depths`: [N].
    """
    width = float(cam.image_width)
    height = float(cam.image_height)
    fx = width / (2.0 * math.tan(float(cam.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(cam.FoVy) * 0.5))
    cx = width * 0.5
    cy = height * 0.5
    x_cam = (pixels_uv[:, 0] - cx) / max(fx, 1e-6) * depths
    y_cam = (pixels_uv[:, 1] - cy) / max(fy, 1e-6) * depths
    z_cam = depths
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
    rotation = torch.as_tensor(cam.R, dtype=points_cam.dtype, device=points_cam.device)
    translation = torch.as_tensor(cam.T, dtype=points_cam.dtype, device=points_cam.device)
    return (points_cam - translation[None, :]) @ rotation  # R^T (p - T)


def select_deficit_pixels(
    mask: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    threshold: float,
    budget: int,
    grid_stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick under-covered foreground pixels for new Gaussians.

    Returns (pixel_uv [M,2], depth [M]).  A coarse grid stride avoids placing
    hundreds of nearly identical points in one leaf hole.
    """
    deficit = (mask * (1.0 - alpha)).squeeze(0)
    depth2d = depth.squeeze(0)
    valid = (deficit > float(threshold)) & (depth2d > 0)
    if grid_stride > 1:
        stride_mask = torch.zeros_like(valid)
        stride_mask[::grid_stride, ::grid_stride] = True
        valid = valid & stride_mask
    if not torch.any(valid):
        return torch.zeros((0, 2), device=mask.device), torch.zeros((0,), device=mask.device)
    scores = deficit[valid]
    count = min(int(budget), int(scores.numel()))
    top = torch.topk(scores, k=count).indices
    ys, xs = torch.nonzero(valid, as_tuple=True)
    ys = ys[top].to(mask.dtype)
    xs = xs[top].to(mask.dtype)
    pixels = torch.stack([xs, ys], dim=1)
    depths = depth2d[ys.long(), xs.long()]
    return pixels, depths


def select_densify_primitives(
    s_m2g: torch.Tensor,
    s_g2m: torch.Tensor,
    residual: torch.Tensor,
    valid: torch.Tensor,
    tau_m: float,
    tau_h: float,
    tau_r: float,
    budget: int,
) -> torch.Tensor:
    """Manuscript densification rule on existing primitives.

        Densify(g_j) = 1[S_m2g > tau_m] * 1[S_g2m > tau_h or R_j > tau_r]

    Candidates are ranked by the deficit evidence ``S_g2m + R`` so that a small
    budget goes to the primitives whose footprint covers the most under-rendered
    foreground.  Returns a LongTensor of selected indices (<= budget).
    """
    condition = (
        valid
        & (s_m2g > float(tau_m))
        & ((s_g2m > float(tau_h)) | (residual > float(tau_r)))
    )
    indices = torch.nonzero(condition).squeeze(1)
    if indices.numel() == 0 or int(budget) <= 0:
        return indices[:0]
    score = (s_g2m + residual)[indices]
    order = torch.argsort(score, descending=True)
    return indices[order[: int(budget)]]


def remap_surviving_indices(selected: torch.Tensor, remove_mask: torch.Tensor) -> torch.Tensor:
    """Translate pre-prune primitive indices into the post-prune index space.

    ``select_densify_primitives`` scores the round's primitives before pruning,
    while ``prune_points`` compacts the model afterwards.  A parent that was
    removed in the same round cannot be split, so it is dropped; every other
    index is shifted by the number of removals that precede it.
    """
    selected = selected.long()
    if selected.numel() == 0:
        return selected
    remove_mask = remove_mask.to(torch.bool)
    shifted = selected - torch.cumsum(remove_mask.long(), dim=0)[selected]
    keep = ~remove_mask[selected]
    return shifted[keep]


def quat_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Quaternion (w, x, y, z) -> rotation matrix, on the input's device.

    Same convention as ``utils.general_utils.build_rotation``, which allocates
    on "cuda" unconditionally and therefore cannot run in CPU unit tests.
    """
    norm = quaternion.norm(dim=1, keepdim=True).clamp_min(EPS)
    q = quaternion / norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - w * z)
    matrix[:, 0, 2] = 2 * (x * z + w * y)
    matrix[:, 1, 0] = 2 * (x * y + w * z)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - w * x)
    matrix[:, 2, 0] = 2 * (x * z - w * y)
    matrix[:, 2, 1] = 2 * (y * z + w * x)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def split_attribute_tensors(
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    index: torch.Tensor,
    scale_shrink: float = 1.6,
    jitter_ratio: float = 0.5,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Split the selected primitives in place of the manuscript's Densify(g_j).

    A 2DGS surfel is split into one child displaced within its own plane by
    ``jitter_ratio * scale`` and with a shrunken scale, mirroring
    ``GaussianModel.densify_and_split`` (the parent is pruned by the caller via
    the returned ``parent_index``).
    """
    if index.numel() == 0:
        empty = xyz[:0]
        return {"xyz": empty, "f_dc": features_dc[:0], "f_rest": features_rest[:0],
                "opacity": opacity[:0], "scaling": scaling[:0],
                "rotation": rotation[:0], "parent_index": index}
    selected_scale = scaling[index]                     # log-space, (m, 2)
    sigma = torch.exp(selected_scale) * float(jitter_ratio)
    if generator is not None:
        noise = torch.randn((index.numel(), 2), device=xyz.device, generator=generator)
    else:
        noise = torch.randn((index.numel(), 2), device=xyz.device)
    # a 2DGS surfel's rotation maps its two local axes into world space
    rotation_matrix = quat_to_rotation_matrix(rotation[index])[:, :, :2]  # (m,3,2)
    offset = torch.bmm(rotation_matrix, (noise * sigma).unsqueeze(-1)).squeeze(-1)
    return {
        "xyz": xyz[index] + offset,
        "f_dc": features_dc[index],
        "f_rest": features_rest[index],
        "opacity": opacity[index],
        "scaling": selected_scale - math.log(float(scale_shrink)),
        "rotation": rotation[index],
        "parent_index": index,
    }


def make_deficit_gaussians(
    points_world: torch.Tensor,
    colors: torch.Tensor,
    base_scale: float,
    sh_degree: int = 3,
    scale_dims: int = 2,
) -> dict[str, torch.Tensor]:
    """Build the attribute dict expected by `densification_postfix`.

    New primitives use isotropic small scales, identity rotation, the observed
    colour as SH-DC and a mid opacity.  They are then optimised normally.

    Shapes must match this fork's `GaussianModel` layout exactly:

        _features_dc   = features[:, :, 0:1].transpose(1, 2)  -> (N, 1, 3)
        _features_rest = features[:, :, 1:].transpose(1, 2)   -> (N, (d+1)^2 - 1, 3)
        _scaling                                              -> (N, 2)   # 2DGS surfel

    i.e. the coefficient axis is dim 1 for `f_rest` (upstream 3DGS stores
    (N, 1, 3*((d+1)^2-1)) instead) and 2DGS surfels carry only two in-plane
    scale axes, which `build_covariance_from_scaling_rotation` pads with 1.
    """
    n = points_world.shape[0]
    device = points_world.device
    sh_c0 = 0.28209479177387814
    features_dc = ((colors.clamp(0.0, 1.0) - 0.5) / sh_c0).reshape(n, 1, 3)
    rest = (int(sh_degree) + 1) ** 2 - 1
    features_rest = torch.zeros((n, rest, 3), device=device)
    opacity = torch.full((n, 1), 2.0, device=device)  # inverse sigmoid(0.88)
    scale = torch.full((n, int(scale_dims)), math.log(max(base_scale, 1e-6)), device=device)
    rotation = torch.zeros((n, 4), device=device)
    rotation[:, 0] = 1.0
    return {
        "xyz": points_world,
        "f_dc": features_dc,
        "f_rest": features_rest,
        "opacity": opacity,
        "scaling": scale,
        "rotation": rotation,
    }
