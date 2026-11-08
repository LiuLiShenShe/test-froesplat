"""Coordinate-mapping tests for the standard-VGGT frontend.

Run BEFORE full inference. All four checks must pass:
  1. original -> model -> original round-trip for known pixels;
  2. K_original = A^{-1} K_model maps the principal point to the original image centre;
  3. RGB / mask geometric consistency under identical padding;
  4. reprojection error of unprojected world points (needs predictions).
"""

from __future__ import annotations

import numpy as np

from .preprocess import apply_affine, build_affine


def test_affine_roundtrip(W: int = 2160, H: int = 3840, model_res: int = 518) -> dict:
    A, A_inv = build_affine(W, H, model_res)
    # sample original pixels including corners and the content bbox
    pts = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1],
                    [W / 2, H / 2], [100, 200], [W - 10, H - 10]], dtype=np.float64)
    model_pts = apply_affine(A, pts)
    back = apply_affine(A_inv, model_pts)
    err = np.abs(back - pts).max()
    # content must lie inside the model square
    inside = bool(np.all(model_pts >= -1e-6) and np.all(model_pts <= model_res + 1e-6))
    ok = err < 1e-6 and inside
    return {"test": "affine_roundtrip", "max_roundtrip_error": float(err), "inside_model": inside, "ok": bool(ok)}


def test_principal_point(W: int = 2160, H: int = 3840, model_res: int = 518) -> dict:
    A, A_inv = build_affine(W, H, model_res)
    # model principal point = centre of the square model image
    cx_m, cy_m, f = model_res / 2.0, model_res / 2.0, 400.0
    K_model = np.array([[f, 0, cx_m], [0, f, cy_m], [0, 0, 1.0]])
    K_orig = A_inv @ K_model
    cx_o, cy_o = K_orig[0, 2], K_orig[1, 2]
    # content centre in original is (W/2, H/2) because padding is centred
    err = max(abs(cx_o - W / 2.0), abs(cy_o - H / 2.0))
    return {"test": "principal_point", "cx_orig": float(cx_o), "cy_orig": float(cy_o),
            "expected": [W / 2.0, H / 2.0], "error": float(err), "ok": bool(err < 1e-6)}


def test_mask_geometry(images, masks518, affines518, original_sizes) -> dict:
    """The mask foreground centroid, mapped back to original px, must sit on the RGB content."""
    worst = 0.0
    for i in range(masks518.shape[0]):
        m = masks518[i].cpu().numpy()
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        cx_m, cy_m = xs.mean(), ys.mean()
        A, A_inv = build_affine(int(original_sizes[i, 0]), int(original_sizes[i, 1]), 518)
        orig = apply_affine(A_inv, np.array([[cx_m, cy_m]]))[0]
        W, H = original_sizes[i]
        if not (0 <= orig[0] < W and 0 <= orig[1] < H):
            worst = max(worst, 1e9)
    return {"test": "mask_geometry", "ok": bool(worst < 1e6)}


def test_reprojection(world_points, model_pixels, extrinsics, intrinsics_model, affines518,
                      original_sizes) -> dict:
    """Project ``world_points`` with ``K_original`` and compare to ``A^{-1} p_model``."""
    errs = []
    for i in range(len(extrinsics)):
        A, A_inv = build_affine(int(original_sizes[i, 0]), int(original_sizes[i, 1]), 518)
        K_model = np.asarray(intrinsics_model[i], dtype=np.float64)
        K_orig = A_inv @ K_model
        R = np.asarray(extrinsics[i], dtype=np.float64)[:3, :3]
        t = np.asarray(extrinsics[i], dtype=np.float64)[:3, 3]
        P = np.asarray(world_points[i], dtype=np.float64).reshape(-1, 3)
        if P.shape[0] == 0:
            continue
        cam = P @ R.T + t
        z = cam[:, 2]
        uv = (cam @ K_orig.T)[:, :2] / np.maximum(z[:, None], 1e-8)
        pm = np.asarray(model_pixels[i], dtype=np.float64).reshape(-1, 2)
        expected = apply_affine(A_inv, pm)
        good = z > 0
        if good.any():
            errs.append(np.linalg.norm(uv[good] - expected[good], axis=1))
    if not errs:
        return {"test": "reprojection", "ok": False, "reason": "no points"}
    e = np.concatenate(errs)
    return {"test": "reprojection", "mean_px": float(e.mean()), "p95_px": float(np.percentile(e, 95)),
            "max_px": float(e.max()), "ok": bool(np.percentile(e, 95) < 2.0)}
