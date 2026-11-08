#!/usr/bin/env python3
"""Part 6 pre-training checks: standard VGGT vs VGGT-Omega, mapped to original 2160x3840 coords.

Produces (under experiments/KongQueZhuYu/standard_vggt_comparison/):
  Standard_VGGT_Frontend/visualizations/
    panel_inputs_12views.png        GT / 518 square-pad model input / mask
    depth_conf_12views.png          per-view: Omega depth/conf vs Standard depth/conf (orig coords)
    depth_sharpness.csv            foreground depth sharpness metrics, 12 views
    pointcloud_6views.png           Omega fused cloud vs Standard raw cloud, 3 viewpoints
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vggt"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PLANT = "KongQueZhuYu"
WS = os.path.join(os.path.dirname(ROOT), "ForeSplat_VGGT_Omega_Frontend")
E1 = os.path.join(WS, "runs", "full_sequence_batch", PLANT, "GGGS_Dataset")
OUT = os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison", "Standard_VGGT_Frontend", "visualizations")
os.makedirs(OUT, exist_ok=True)

W0, H0 = 2160, 3840
S518 = 518 / 3840.0            # standard VGGT: original -> 518 (max_dim = 3840)
OMEGA_H, OMEGA_W = 688, 384    # VGGT-Omega balanced@512 model grid (no crop: aspect 1.778 in [0.5, 2])
X0 = (518 - W0 * S518) / 2.0   # content x-start in the 518 grid: 113.31
VIEWS = np.linspace(0, 207, 12).astype(int)


def load_std_depth_conf(i):
    name = f"crop_{i:04d}"
    d = np.load(os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison",
                             "Standard_VGGT_Frontend", "depth", name + ".npy"))
    c = np.load(os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison",
                             "Standard_VGGT_Frontend", "confidence", name + ".npy"))
    return d[..., 0] if d.ndim == 3 else d, c


def load_omega_depth_conf(i):
    name = f"crop_{i:04d}"
    d = np.load(os.path.join(E1, "per_frame", f"depth_{name}.npy"))
    c = np.load(os.path.join(E1, "per_frame", f"confidence_{name}.npy"))
    return d, c


def std_to_original(arr, order=Image.BILINEAR):
    """518-square content strip -> 2160x3840. Content: x in [X0, 518-X0], full height."""
    im = Image.fromarray(arr.astype(np.float32))
    w = int(round(W0 * S518))
    strip = im.crop((int(round(X0)), 0, int(round(X0)) + w, 518))
    return np.asarray(strip.resize((W0, H0), order))


def omega_to_original(arr, order=Image.BILINEAR):
    im = Image.fromarray(arr.astype(np.float32))
    return np.asarray(im.resize((W0, H0), order))


def fg_mask_orig(i):
    m = Image.open(os.path.join(E1, "masks", f"crop_{i:04d}.png")).convert("L")
    return np.asarray(m) > 127


def norm_fg(depth, fg):
    lo, hi = np.percentile(depth[fg], [2, 98])
    out = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)
    out[~fg] = 0.5
    return out


def grad_mag(norm_depth, fg):
    gx = np.gradient(norm_depth, axis=1)
    gy = np.gradient(norm_depth, axis=0)
    return np.hypot(gx, gy)[fg]


def main():
    names = sorted(f for f in os.listdir(os.path.join(E1, "images")) if f.endswith(".png"))
    assert len(names) == 208

    # ---- Panel 1: 12-view input comparison --------------------------------
    fig, axes = plt.subplots(12, 3, figsize=(9, 48))
    for r, i in enumerate(VIEWS):
        rgb = Image.open(os.path.join(E1, "images", names[i])).convert("RGB")
        m = Image.open(os.path.join(E1, "masks", names[i].replace(".png", ".png"))).convert("L")
        model_rgb = Image.new("RGB", (518, 518), (0, 0, 0))
        rgb_sq = rgb.resize((int(round(W0 * S518)), 518), Image.BICUBIC)
        model_rgb.paste(rgb_sq, (int(round(X0)), 0))
        model_mask = np.zeros((518, 518), np.uint8)
        m518 = np.asarray(m.resize((int(round(W0 * S518)), 518), Image.NEAREST))
        model_mask[:, int(round(X0)):int(round(X0)) + m518.shape[1]] = m518
        axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"GT {names[i]}", fontsize=8)
        axes[r, 1].imshow(model_rgb); axes[r, 1].set_title("std-VGGT 518 input", fontsize=8)
        axes[r, 2].imshow(model_mask, cmap="gray"); axes[r, 2].set_title("model mask@518", fontsize=8)
        for a in axes[r]:
            a.axis("off")
    fig.suptitle("Standard VGGT input check: GT vs 518 square-pad input vs mask (12 evenly spaced views)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "panel_inputs_12views.png"), dpi=60)
    plt.close(fig)

    # ---- Panel 2: depth / confidence comparison --------------------------
    fig, axes = plt.subplots(12, 5, figsize=(20, 48))
    rows = []
    for r, i in enumerate(VIEWS):
        fg = fg_mask_orig(i)
        d_o, c_o = load_omega_depth_conf(i)
        d_s, c_s = load_std_depth_conf(i)
        d_o_o = omega_to_original(d_o)
        d_s_o = std_to_original(d_s)
        c_o_o = omega_to_original(c_o)
        c_s_o = std_to_original(c_s)

        axes[r, 0].imshow(norm_fg(d_o_o, fg), cmap="viridis")
        axes[r, 0].set_title(f"Omega depth {names[i]}", fontsize=8)
        axes[r, 1].imshow(norm_fg(d_s_o, fg), cmap="viridis")
        axes[r, 1].set_title("Std depth", fontsize=8)
        axes[r, 2].imshow(norm_fg(c_o_o, fg), cmap="magma")
        axes[r, 2].set_title("Omega conf", fontsize=8)
        axes[r, 3].imshow(norm_fg(c_s_o, fg), cmap="magma")
        axes[r, 3].set_title("Std conf", fontsize=8)
        # difference of normalized depths (foreground only)
        diff = np.abs(norm_fg(d_o_o, fg) - norm_fg(d_s_o, fg))
        diff[~fg] = 0
        axes[r, 4].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[r, 4].set_title("|Δdepth| (norm)", fontsize=8)
        for a in axes[r]:
            a.axis("off")

        g_o = grad_mag(norm_fg(d_o_o, fg), fg)
        g_s = grad_mag(norm_fg(d_s_o, fg), fg)
        rows.append(dict(view=i, omega_grad=g_o.mean(), std_grad=g_s.mean(),
                         omega_depth_lo=float(d_o_o[fg].min()), omega_depth_hi=float(d_o_o[fg].max()),
                         std_depth_lo=float(d_s_o[fg].min()), std_depth_hi=float(d_s_o[fg].max())))
    fig.suptitle("Depth / confidence on ORIGINAL 2160x3840 coords (per-view 2-98% normalized). Cols: Omega depth | Std depth | Omega conf | Std conf | |diff|", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth_conf_12views.png"), dpi=60)
    plt.close(fig)

    import csv
    with open(os.path.join(OUT, "depth_sharpness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    gm_o = np.mean([r["omega_grad"] for r in rows]); gm_s = np.mean([r["std_grad"] for r in rows])
    print(f"foreground depth-gradient: Omega {gm_o:.4f}  Std {gm_s:.4f} (orig-res, normalized depth)")

    # ---- Panel 3: point cloud comparison (6 views) ------------------------
    from plyfile import PlyData

    def load_cloud(path):
        p = PlyData.read(path)
        v = p["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], 1), np.stack([v["red"], v["green"], v["blue"]], 1).astype(float) / 255

    xyz_o, rgb_o = load_cloud(os.path.join(E1, "sparse", "0", "points3D.ply"))
    xyz_s, rgb_s = load_cloud(os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison",
                                           "Standard_VGGT_Dataset", "sparse", "0", "points3D.ply"))

    def frame(xyz):
        c = xyz.mean(0)
        return xyz - c

    xyz_o, xyz_s = frame(xyz_o), frame(xyz_s)

    viewpoints = [(20, -70, "iso"), (-90, -5, "front X"), (0, -5, "side Y")]
    fig = plt.figure(figsize=(16, 16))
    k = 1
    for elev, azim, lab in viewpoints:
        for xyz, rgb, tag in ((xyz_o, rgb_o, "VGGT-Omega fused"), (xyz_s, rgb_s, "Standard VGGT raw")):
            ax = fig.add_subplot(3, 2, k, projection="3d")
            k += 1
            idx = np.random.default_rng(0).choice(len(xyz), 60000, replace=False)
            ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2],
                       c=rgb[idx], s=0.2, marker=".")
            ax.set_title(f"{tag} — {lab} (60k subsample)", fontsize=9)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            m = xyz[idx]
            ax.set_xlim(m[:, 0].min(), m[:, 0].max())
            ax.set_ylim(m[:, 1].min(), m[:, 1].max())
            ax.set_zlim(m[:, 2].min(), m[:, 2].max())
    fig.suptitle("Point cloud comparison: VGGT-Omega fused (E1) vs Standard VGGT raw depth back-projection", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pointcloud_6views.png"), dpi=60)
    plt.close(fig)
    print("visualizations written to", OUT)


if __name__ == "__main__":
    main()
