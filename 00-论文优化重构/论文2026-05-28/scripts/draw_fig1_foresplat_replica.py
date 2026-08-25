#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SVG = OUT_DIR / "fig1_foresplat_overview.svg"
PNG = OUT_DIR / "fig1_foresplat_overview_preview.png"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


C = {
    "black": "#111111",
    "grey": "#6E6E6E",
    "line": "#A8A8A8",
    "light": "#F8F8F8",
    "purple": "#7C43B3",
    "purple_l": "#F2ECFA",
    "blue": "#0B63B6",
    "blue_l": "#EDF5FF",
    "green": "#147515",
    "green_l": "#EEF8EA",
    "gold": "#B58200",
    "gold_l": "#FFF7E4",
    "orange": "#E46B19",
    "orange_l": "#FFF1E8",
    "brown": "#B45B1D",
    "brown_l": "#FFF3EC",
    "dark_brown": "#8A4A1D",
    "phen": "#6A6A6A",
    "phen_l": "#F2F2F2",
}


def rounded(ax, x, y, w, h, ec, fc, lw=1.1, r=0.10, z=1):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.014,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=8, color=None, weight="normal", ha="center", va="center", z=20):
    ax.text(x, y, s, fontsize=size, color=color or C["black"], fontweight=weight, ha=ha, va=va, linespacing=1.12, zorder=z)


def arrow(ax, x1, y1, x2, y2, color="#222222", lw=1.2, dashed=False, ms=10, rad=0):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=(0, (3, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=25,
    )
    ax.add_patch(a)
    return a


def star_points(cx, cy, r1, r2, n=5):
    pts = []
    for i in range(n * 2):
        a = math.pi / 2 + i * math.pi / n
        r = r1 if i % 2 == 0 else r2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def star(ax, x, y, r=0.055, fill="#F1B60B", ec="#333333", z=30):
    ax.add_patch(Polygon(star_points(x, y, r, r * 0.44), closed=True, facecolor=fill, edgecolor=ec, linewidth=0.45, zorder=z))


def camera_icon(ax, x, y, s=1.0, color="#4F5B66", z=25):
    ax.add_patch(Rectangle((x - 0.075 * s, y - 0.045 * s), 0.15 * s, 0.09 * s, facecolor=color, edgecolor="#222", lw=0.45, zorder=z))
    ax.add_patch(Rectangle((x - 0.035 * s, y + 0.045 * s), 0.07 * s, 0.025 * s, facecolor=color, edgecolor="#222", lw=0.35, zorder=z))
    ax.add_patch(Circle((x, y), 0.032 * s, facecolor="#DCE5EF", edgecolor="#222", lw=0.35, zorder=z + 1))


def leaf(ax, x, y, w, h, angle, fc="#5FAE45", ec="#2C6A22", z=12, alpha=1.0):
    ax.add_patch(Ellipse((x, y), w, h, angle=angle, facecolor=fc, edgecolor=ec, lw=0.45, alpha=alpha, zorder=z))


def draw_potted_plant(ax, cx, cy, s=1.0, z=10, grey=False, mesh=False):
    green1, green2 = ("#7A7A7A", "#BDBDBD") if grey else ("#4F9D38", "#75BC4E")
    ec = "#555555" if grey else "#2F6D26"
    pot_fc = "#9B7557" if not grey else "#D0D0D0"
    pot_ec = "#6E513C" if not grey else "#888888"
    if mesh:
        pot_fc, pot_ec = "#A7D091", "#4D8A38"
    ax.add_patch(Polygon([(cx - 0.27 * s, cy - 0.45 * s), (cx + 0.27 * s, cy - 0.45 * s), (cx + 0.20 * s, cy - 0.76 * s), (cx - 0.20 * s, cy - 0.76 * s)], facecolor=pot_fc, edgecolor=pot_ec, lw=0.7, zorder=z))
    ax.add_patch(Rectangle((cx - 0.30 * s, cy - 0.48 * s), 0.60 * s, 0.07 * s, facecolor=pot_fc, edgecolor=pot_ec, lw=0.6, zorder=z + 1))
    for x2, y2 in [(0, 0.30), (-0.20, 0.15), (0.23, 0.20), (-0.08, 0.44), (0.13, 0.43)]:
        ax.plot([cx, cx + x2 * s], [cy - 0.40 * s, cy + y2 * s], color=ec, lw=1.0 * s, zorder=z + 2)
    leaves = [(-0.25, 0.12, 0.46, 0.15, 22), (0.28, 0.17, 0.48, 0.15, -24), (-0.08, 0.44, 0.44, 0.14, 76), (0.17, 0.42, 0.42, 0.14, 54), (0.02, 0.28, 0.40, 0.14, -4)]
    for dx, dy, w, h, ang in leaves:
        if mesh:
            mesh_leaf(ax, cx + dx * s, cy + dy * s, w * s, h * s, ang, z=z + 4)
        else:
            leaf(ax, cx + dx * s, cy + dy * s, w * s, h * s, ang, green2, ec, z=z + 4)


def mesh_leaf(ax, x, y, w, h, angle, z=14):
    th = math.radians(angle)
    ux, uy = math.cos(th), math.sin(th)
    vx, vy = -math.sin(th), math.cos(th)
    pts = []
    for t, side in [(-0.5, 0), (-0.18, 0.48), (0.18, 0.35), (0.5, 0), (0.18, -0.35), (-0.18, -0.48)]:
        pts.append((x + t * w * ux + side * h * vx, y + t * w * uy + side * h * vy))
    for tri in [(0, 1, 2), (0, 2, 5), (2, 3, 4), (2, 4, 5)]:
        ax.add_patch(Polygon([pts[i] for i in tri], closed=True, facecolor="#A8D68B", edgecolor="#3D7F2B", lw=0.5, zorder=z))


def draw_photo(ax, x, y, w, h, z=10):
    rounded(ax, x, y, w, h, "#CFCFCF", "#FFFFFF", lw=0.8, r=0.06, z=z)
    ax.add_patch(Rectangle((x + 0.05 * w, y + 0.06 * h), 0.90 * w, 0.84 * h, facecolor="#F1F1F1", edgecolor="none", zorder=z + 1))
    draw_potted_plant(ax, x + 0.50 * w, y + 0.58 * h, s=0.37 * min(w, h), z=z + 2)
    # Sparse red feature tracks on the floor, as in the reference.
    rng = np.random.default_rng(0)
    for _ in range(22):
        px = x + (0.18 + 0.65 * rng.random()) * w
        py = y + (0.08 + 0.16 * rng.random()) * h
        ax.add_patch(Circle((px, py), 0.006, facecolor="#D66B5B", edgecolor="none", alpha=0.8, zorder=z + 5))


def draw_mask(ax, cx, cy, s=1.0, z=12):
    ax.add_patch(Rectangle((cx - 0.38 * s, cy - 0.42 * s), 0.76 * s, 0.84 * s, facecolor="black", edgecolor="black", lw=0.6, zorder=z))
    verts = [
        (cx - 0.05 * s, cy - 0.30 * s),
        (cx - 0.34 * s, cy - 0.08 * s),
        (cx - 0.24 * s, cy + 0.15 * s),
        (cx - 0.04 * s, cy + 0.10 * s),
        (cx - 0.02 * s, cy + 0.34 * s),
        (cx + 0.18 * s, cy + 0.15 * s),
        (cx + 0.33 * s, cy - 0.02 * s),
        (cx + 0.12 * s, cy - 0.18 * s),
        (cx + 0.08 * s, cy - 0.34 * s),
        (cx - 0.05 * s, cy - 0.30 * s),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(verts) - 1)
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="white", edgecolor="white", lw=0.4, zorder=z + 1))


def draw_quality(ax, x, y, w, h):
    ax.plot([x + 0.08 * w, x + 0.08 * w], [y + 0.18 * h, y + 0.75 * h], color="black", lw=3)
    ax.plot([x + 0.08 * w, x + 0.42 * w], [y + 0.18 * h, y + 0.18 * h], color="black", lw=3)
    heights = [0.18, 0.36, 0.23, 0.15]
    for i, bh in enumerate(heights):
        bx = x + (0.20 + 0.10 * i) * w
        ax.add_patch(Rectangle((bx, y + 0.18 * h), 0.055 * w, bh * h, facecolor="#D8D8D8", edgecolor="#333", lw=0.5, zorder=12))
    ax.add_patch(Circle((x + 0.78 * w, y + 0.34 * h), 0.07 * h, facecolor="#6FC36D", edgecolor="#2B802D", lw=0.7, zorder=12))
    ax.plot([x + 0.75 * w, x + 0.78 * w, x + 0.83 * w], [y + 0.34 * h, y + 0.30 * h, y + 0.39 * h], color="white", lw=1.3, zorder=13)
    ax.add_patch(Circle((x + 0.78 * w, y + 0.17 * h), 0.07 * h, facecolor="#F07C76", edgecolor="#A9342C", lw=0.7, zorder=12))
    ax.plot([x + 0.745 * w, x + 0.815 * w], [y + 0.135 * h, y + 0.205 * h], color="white", lw=1.1, zorder=13)
    ax.plot([x + 0.745 * w, x + 0.815 * w], [y + 0.205 * h, y + 0.135 * h], color="white", lw=1.1, zorder=13)


def draw_colmap_pose(ax, x, y, w, h):
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(80, 2))
    pts[:, 0] = x + w / 2 + pts[:, 0] * 0.22 * w
    pts[:, 1] = y + h / 2 + pts[:, 1] * 0.18 * h
    ax.scatter(pts[:, 0], pts[:, 1], s=2.2, c="#E45B4E", alpha=0.55, zorder=12)
    for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        cx = x + w / 2 + math.cos(a) * 0.38 * w
        cy = y + h / 2 + math.sin(a) * 0.32 * h
        camera_icon(ax, cx, cy, s=0.62, z=14)
        ax.plot([cx, x + w / 2], [cy, y + h / 2], color="#777", lw=0.35, alpha=0.45, zorder=11)
    ax.plot([x + w / 2, x + w / 2], [y - 0.06, y - 0.25], color="black", lw=1.0, linestyle=(0, (2, 3)), zorder=20)


def scatter_cloud(ax, cx, cy, sx, sy, color, n=120, alpha=0.65, seed=1, size=5):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 2))
    ax.scatter(cx + pts[:, 0] * sx, cy + pts[:, 1] * sy, s=size, c=color, alpha=alpha, edgecolors="none", zorder=12)


def draw_rendering(ax, x, y, w, h):
    ax.add_patch(Circle((x + 0.50 * w, y + 0.55 * h), 0.30 * h, facecolor="#DCEFD2", edgecolor="none", alpha=0.65, zorder=10))
    scatter_cloud(ax, x + 0.50 * w, y + 0.55 * h, 0.18 * w, 0.18 * h, "#53A735", n=160, alpha=0.48, seed=3, size=3)
    draw_photo(ax, x + 0.06 * w, y + 0.20 * h, 0.22 * w, 0.38 * h, z=13)
    draw_photo(ax, x + 0.72 * w, y + 0.20 * h, 0.22 * w, 0.38 * h, z=13)
    for cx, cy in [(x + 0.18 * w, y + 0.78 * h), (x + 0.82 * w, y + 0.78 * h), (x + 0.02 * w, y + 0.52 * h), (x + 0.98 * w, y + 0.52 * h)]:
        camera_icon(ax, cx, cy, s=0.62, z=14)
        ax.plot([cx, x + 0.5 * w], [cy, y + 0.55 * h], color="#888", lw=0.45, zorder=11)


def draw_plant_mesh_cube(ax, cx, cy, s=1.0, grey=False):
    # cube grid
    col = "#B6B6B6" if grey else "#999999"
    for i in range(6):
        t = -0.45 + i * 0.18
        ax.plot([cx - 0.45 * s, cx + 0.45 * s], [cy + t * s, cy + (t + 0.24) * s], color=col, lw=0.45, zorder=10)
        ax.plot([cx + t * s, cx + (t + 0.24) * s], [cy - 0.45 * s, cy + 0.45 * s], color=col, lw=0.45, zorder=10)
    ax.add_patch(Rectangle((cx - 0.45 * s, cy - 0.45 * s), 0.90 * s, 0.90 * s, facecolor="none", edgecolor=col, lw=0.7, zorder=11))
    draw_potted_plant(ax, cx, cy - 0.02 * s, s=0.68 * s, grey=grey, mesh=grey, z=12)


def section(ax, x, y, w, h, title, title_color, fc, ec, number=None, title_size=13):
    rounded(ax, x, y, w, h, ec, fc, lw=1.15, r=0.10, z=1)
    if number:
        txt(ax, x + w / 2, y + h - 0.34, f"{number}. {title}", size=title_size, color=title_color, weight="bold")
    else:
        txt(ax, x + w / 2, y + h - 0.36, title, size=title_size, color=title_color, weight="bold")


def inner(ax, x, y, w, h, title=None, ec="#BBBBBB", fc="#FFFFFF", color="#111", title_size=8.6):
    rounded(ax, x, y, w, h, ec, fc, lw=0.85, r=0.08, z=5)
    if title:
        txt(ax, x + w / 2, y + h - 0.22, title, size=title_size, color=color, weight="bold")


def main():
    fig, ax = plt.subplots(figsize=(17.8, 9.8))
    ax.set_xlim(0, 17.8)
    ax.set_ylim(0, 9.8)
    ax.axis("off")

    y0, h = 2.55, 6.85
    gap = 0.25
    xs = [0.05, 1.72, 3.92, 6.15, 9.20, 11.35, 13.90, 15.75]
    ws = [1.45, 1.95, 1.95, 2.85, 1.90, 2.30, 1.60, 2.00]

    # Input
    section(ax, xs[0], y0, ws[0], h, "Input", C["black"], "#FBFBFB", "#777777", title_size=12)
    txt(ax, xs[0] + ws[0] / 2, y0 + h - 0.68, "Multi-view\nRGB Images", size=8.2, weight="bold")
    py = [7.05, 5.75, 4.05]
    for yy in py:
        draw_photo(ax, xs[0] + 0.17, yy, 1.10, 0.90)
    txt(ax, xs[0] + ws[0] / 2, y0 + 0.70, r"$N$ views", size=10)
    txt(ax, xs[0] + ws[0] / 2, y0 + 1.40, "...", size=12)

    # FSAM3
    section(ax, xs[1], y0, ws[1], h, "FSAM3", C["purple"], C["purple_l"], C["purple"], "1", title_size=12)
    inner(ax, xs[1] + 0.13, 7.15, 1.69, 1.55, "Quality Filtering", C["purple"], "#FFFFFF", C["purple"])
    draw_quality(ax, xs[1] + 0.35, 7.35, 1.20, 0.95)
    arrow(ax, xs[1] + ws[1] / 2, 7.15, xs[1] + ws[1] / 2, 6.95, C["purple"], lw=0.9, ms=8)
    inner(ax, xs[1] + 0.13, 5.25, 1.69, 1.65, "Plant Segmentation", C["purple"], "#FFFFFF", C["purple"])
    draw_mask(ax, xs[1] + ws[1] / 2, 5.83, s=0.74)
    arrow(ax, xs[1] + ws[1] / 2, 5.25, xs[1] + ws[1] / 2, 5.02, C["purple"], lw=0.9, ms=8)
    inner(ax, xs[1] + 0.13, 3.28, 1.69, 1.65, "Main Component\nRefinement", C["purple"], "#FFFFFF", C["purple"])
    draw_mask(ax, xs[1] + ws[1] / 2, 3.82, s=0.74)
    txt(ax, xs[1] + ws[1] / 2, y0 + 0.42, "Aligned Foreground\nMasks", size=8.0, color=C["purple"], weight="bold")

    # COLMAP
    section(ax, xs[2], y0, ws[2], h, "COLMAP", C["blue"], C["blue_l"], C["blue"], "2", title_size=12)
    inner(ax, xs[2] + 0.13, 6.73, 1.69, 1.96, "Camera Pose\nEstimation", C["blue"], "#FFFFFF", C["black"])
    draw_colmap_pose(ax, xs[2] + 0.30, 6.95, 1.34, 1.25)
    arrow(ax, xs[2] + ws[2] / 2, 6.73, xs[2] + ws[2] / 2, 6.45, C["blue"], lw=0.9, ms=8)
    inner(ax, xs[2] + 0.13, 4.78, 1.69, 1.75, "Sparse Point\nTriangulation", C["blue"], "#FFFFFF", C["black"])
    scatter_cloud(ax, xs[2] + ws[2] / 2, 5.34, 0.33, 0.25, "#9C9C9C", n=95, alpha=0.7, seed=5, size=5)
    arrow(ax, xs[2] + ws[2] / 2, 4.78, xs[2] + ws[2] / 2, 4.52, C["blue"], lw=0.9, ms=8)
    inner(ax, xs[2] + 0.13, 3.00, 1.69, 1.55, "Foreground Point\nFiltering", C["blue"], "#FFFFFF", C["black"])
    scatter_cloud(ax, xs[2] + ws[2] / 2, 3.45, 0.35, 0.22, "#4FAA27", n=105, alpha=0.75, seed=6, size=5)

    # Plant-aware 2DGS
    section(ax, xs[3], y0, ws[3], h, "Plant-aware 2DGS", C["green"], C["green_l"], "#78A95E", "3", title_size=12)
    inner(ax, xs[3] + 0.13, 7.15, 2.59, 1.55, "Differentiable Rendering", "#78A95E", "#FFFFFF", C["black"])
    draw_rendering(ax, xs[3] + 0.32, 7.31, 2.20, 0.98)
    rows = [
        ("RGB Loss\n(in Foreground)", 6.38, "plant-mask"),
        ("Alpha Mask Loss", 5.60, "mask-arrow"),
        ("Background Opacity Loss", 4.82, "bg"),
    ]
    for title, yy, typ in rows:
        inner(ax, xs[3] + 0.13, yy, 2.59, 0.66, None, "#78A95E", "#FFFFFF", C["black"])
        txt(ax, xs[3] + 0.42, yy + 0.33, title, size=8.3, ha="left")
        if typ == "plant-mask":
            draw_potted_plant(ax, xs[3] + 1.78, yy + 0.38, s=0.20, z=15)
            arrow(ax, xs[3] + 2.05, yy + 0.33, xs[3] + 2.20, yy + 0.33, "#111", lw=0.8, ms=7)
            draw_mask(ax, xs[3] + 2.42, yy + 0.33, s=0.22)
        elif typ == "mask-arrow":
            draw_mask(ax, xs[3] + 1.77, yy + 0.33, s=0.23)
            arrow(ax, xs[3] + 2.00, yy + 0.33, xs[3] + 2.19, yy + 0.33, "#111", lw=0.8, ms=7)
            draw_mask(ax, xs[3] + 2.43, yy + 0.33, s=0.23)
        else:
            scatter_cloud(ax, xs[3] + 1.52, yy + 0.31, 0.16, 0.14, "#59B72A", n=55, alpha=0.7, seed=7, size=4)
            arrow(ax, xs[3] + 1.90, yy + 0.31, xs[3] + 2.16, yy + 0.31, "#111", lw=0.8, ms=7)
            draw_plant_mesh_cube(ax, xs[3] + 2.35, yy + 0.31, s=0.32, grey=True)
    arrow(ax, xs[3] + ws[3] / 2, 4.78, xs[3] + ws[3] / 2, 4.60, C["green"], lw=0.9, ms=8)
    inner(ax, xs[3] + 0.13, 3.00, 2.59, 1.55, "Optimized Gaussians", "#78A95E", "#FFFFFF", C["black"])
    scatter_cloud(ax, xs[3] + ws[3] / 2, 3.48, 0.60, 0.27, "#54AD28", n=220, alpha=0.45, seed=8, size=7)
    scatter_cloud(ax, xs[3] + ws[3] / 2, 3.48, 0.44, 0.20, "#DDA23B", n=80, alpha=0.25, seed=9, size=9)

    # View quality
    section(ax, xs[4], y0, ws[4], h, "View Quality\nSoft Weighting", C["gold"], C["gold_l"], "#E6B247", "4", title_size=11)
    inner(ax, xs[4] + 0.13, 5.98, 1.64, 2.52, "View Quality\nAssessment", "#E6B247", "#FFFFFF", C["gold"])
    for i, stars in enumerate([5, 3, 2, 1]):
        yy = 7.65 - i * 0.36
        camera_icon(ax, xs[4] + 0.40, yy, s=0.72)
        for j in range(5):
            star(ax, xs[4] + 0.83 + j * 0.16, yy, r=0.055, fill="#F2BE20" if j < stars else "#E9E9E9", ec="#555")
    txt(ax, xs[4] + ws[4] / 2, 6.12, "...", size=10)
    arrow(ax, xs[4] + ws[4] / 2, 5.98, xs[4] + ws[4] / 2, 5.74, C["gold"], lw=0.9, ms=8)
    inner(ax, xs[4] + 0.13, 3.85, 1.64, 1.88, "Soft Weights\n(0 ~ 1)", "#E6B247", "#FFFFFF", C["gold"])
    vals = [1.00, 0.73, 0.45, 0.21]
    for i, v in enumerate(vals):
        yy = 5.12 - i * 0.30
        camera_icon(ax, xs[4] + 0.36, yy, s=0.58)
        ax.add_patch(Rectangle((xs[4] + 0.60, yy - 0.07), 0.72 * v, 0.14, facecolor="#E7B530", edgecolor="#C9971F", lw=0.35))
        txt(ax, xs[4] + 1.40, yy, f"{v:.2f}", size=7.6)
    txt(ax, xs[4] + ws[4] / 2, 3.98, "...", size=10)
    arrow(ax, xs[4] + ws[4] / 2, 3.85, xs[4] + ws[4] / 2, 3.60, C["gold"], lw=0.9, ms=8)
    inner(ax, xs[4] + 0.13, 2.98, 1.64, 0.62, "Weighted Loss\nAggregation", "#E6B247", "#FFFFFF", C["gold"])
    txt(ax, xs[4] + ws[4] / 2, 3.11, r"$\sum_i\ w_i\ L_i$", size=13)

    # Gaussian pruning
    section(ax, xs[5], y0, ws[5], h, "Mask-guided\nGaussian Pruning", C["orange"], C["orange_l"], "#F19955", "5", title_size=11)
    inner(ax, xs[5] + 0.15, 5.36, 2.00, 2.54, "Mask-guided\nMulti-cue Pruning", "#F19955", "#FFFFFF", C["orange"])
    draw_mask(ax, xs[5] + 0.68, 6.42, s=0.55)
    scatter_cloud(ax, xs[5] + 1.55, 6.28, 0.27, 0.33, "#6ABF3B", n=130, alpha=0.35, seed=11, size=7)
    arrow(ax, xs[5] + 1.03, 6.02, xs[5] + 1.35, 5.88, C["orange"], lw=1.0, ms=10, rad=0.15)
    arrow(ax, xs[5] + ws[5] / 2, 5.36, xs[5] + ws[5] / 2, 5.06, C["orange"], lw=0.9, ms=8)
    inner(ax, xs[5] + 0.15, 2.72, 2.00, 2.13, "Pruned Gaussians", "#F19955", "#FFFFFF", C["orange"])
    scatter_cloud(ax, xs[5] + ws[5] / 2, 3.55, 0.48, 0.37, "#66B72E", n=230, alpha=0.58, seed=12, size=7)

    # TSDF
    section(ax, xs[6], y0, ws[6], h, "TSDF\nMeshing", C["brown"], C["brown_l"], "#C57A43", "6", title_size=11)
    inner(ax, xs[6] + 0.10, 5.90, 1.40, 2.33, "TSDF Fusion", "#C57A43", "#FFFFFF", C["black"])
    draw_plant_mesh_cube(ax, xs[6] + ws[6] / 2, 6.78, s=0.62, grey=False)
    arrow(ax, xs[6] + ws[6] / 2, 5.90, xs[6] + ws[6] / 2, 5.58, C["brown"], lw=0.9, ms=8)
    inner(ax, xs[6] + 0.10, 2.74, 1.40, 2.52, "Mesh Extraction", "#C57A43", "#FFFFFF", C["black"])
    draw_potted_plant(ax, xs[6] + ws[6] / 2, 3.82, s=0.75, grey=True, mesh=True, z=12)

    # Phenotyping
    section(ax, xs[7], y0, ws[7], h, "Phenotype\nMeasurement", C["phen"], C["phen_l"], "#9B9B9B", "7", title_size=11)
    inner(ax, xs[7] + 0.10, 5.62, 1.80, 2.55, "Plant Mesh\n(For Measurement)", "#B8B8B8", "#FFFFFF", C["black"])
    draw_potted_plant(ax, xs[7] + ws[7] / 2, 6.42, s=0.86, mesh=True, z=12)
    inner(ax, xs[7] + 0.10, 2.90, 1.80, 2.42, "Virtual Phenotypes", "#B8B8B8", "#FFFFFF", C["black"])
    bx = xs[7] + 0.45
    by = 4.63
    ax.add_patch(FancyArrowPatch((bx, by - 0.18), (bx, by + 0.18), arrowstyle="<->", mutation_scale=9, color="#67B53E", lw=1.1, zorder=15))
    txt(ax, xs[7] + 1.34, by, "Plant Height", size=8.0)
    by -= 0.45
    ax.add_patch(FancyArrowPatch((bx - 0.18, by), (bx + 0.18, by), arrowstyle="<->", mutation_scale=9, color="#67B53E", lw=1.1, zorder=15))
    txt(ax, xs[7] + 1.34, by, "Canopy Width", size=8.0)
    by -= 0.45
    leaf(ax, bx, by, 0.30, 0.14, 40, fc="#7FBC46", ec="#4D8A2A", z=15)
    txt(ax, xs[7] + 1.34, by, "Leaf Size", size=8.0)
    txt(ax, xs[7] + ws[7] / 2, 3.22, "...", size=11)

    # Main data-flow arrows between columns.
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + ws[i], y0 + h * 0.50, xs[i + 1], y0 + h * 0.50, [C["purple"], C["blue"], C["green"], C["gold"], C["orange"], C["brown"], "#8A5A2B"][i], lw=1.15, ms=11)

    # Feedback/guidance dashed arrows along bottom.
    yfb = 2.18
    colors = [C["purple"], C["blue"], C["green"], C["gold"], C["orange"], C["brown"]]
    for i, col in enumerate(colors):
        x_start = xs[i] + ws[i] / 2
        x_end = xs[i + 1] + ws[i + 1] / 2
        ax.plot([x_start, x_start, x_end, x_end], [y0, yfb, yfb, y0], color=col, lw=0.9, linestyle=(0, (3, 3)), zorder=5)
        ax.add_patch(FancyArrowPatch((x_start, yfb), (x_start, y0 + 0.02), arrowstyle="-|>", mutation_scale=8, color=col, lw=0.8, linestyle=(0, (3, 3)), zorder=6))

    # Legend box.
    leg_y = 0.98
    rounded(ax, 0.15, leg_y, 17.35, 0.95, "#9B9B9B", "#FFFFFF", lw=0.8, r=0.07, z=1)
    ax.add_patch(Rectangle((0.15, leg_y), 17.35, 0.95, facecolor="none", edgecolor="#9B9B9B", lw=0.8, linestyle=(0, (4, 3)), zorder=2))
    lx = 0.46
    camera_icon(ax, lx, leg_y + 0.48, s=0.92, z=20)
    txt(ax, lx + 0.28, leg_y + 0.48, "Camera", size=8.0, ha="left")
    legend_items = [
        (1.85, C["purple_l"], C["purple"], "FSAM3\n(Preprocessing)"),
        (3.72, C["blue_l"], C["blue"], "COLMAP\n(SfM)"),
        (5.45, C["green_l"], "#78A95E", "2DGS Optimization\n(Plant-aware)"),
        (7.55, C["gold_l"], "#E6B247", "View Weighting\n(Training Strategy)"),
        (9.90, C["orange_l"], "#F19955", "Gaussian Pruning\n(Model Refinement)"),
        (12.40, C["brown_l"], "#C57A43", "Meshing\n(Geometry)"),
        (13.95, C["phen_l"], "#9B9B9B", "Phenotyping\n(Measurement)"),
    ]
    for x, fc, ec, lab in legend_items:
        rounded(ax, x, leg_y + 0.32, 0.28, 0.32, ec, fc, lw=0.7, r=0.025, z=20)
        txt(ax, x + 0.48, leg_y + 0.48, lab, size=7.6, ha="left")
    arrow(ax, 15.45, leg_y + 0.62, 15.95, leg_y + 0.62, "black", lw=1.2, ms=11)
    txt(ax, 16.10, leg_y + 0.62, "Data Flow", size=8.0, ha="left")
    arrow(ax, 15.45, leg_y + 0.33, 15.95, leg_y + 0.33, "black", lw=1.0, dashed=True, ms=10)
    txt(ax, 16.10, leg_y + 0.33, "Feedback / Guidance", size=8.0, ha="left")

    txt(ax, 8.9, 0.28, "ForeSplat Overview.", size=12.0, weight="bold", ha="right")
    txt(ax, 9.02, 0.28, "From multi-view images to plant-only mesh and virtual phenotypes.", size=12.0, ha="left")

    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PNG, dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(SVG)
    print(PNG)


if __name__ == "__main__":
    main()
