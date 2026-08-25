#!/usr/bin/env python3
from pathlib import Path
import math

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


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SVG_PRIMARY = OUT_DIR / "fig1_foresplat_overview.svg"
SVG_ALT = OUT_DIR / "fig1_foresplat_illustration.svg"
PNG_PREVIEW = OUT_DIR / "fig1_foresplat_illustration_preview.png"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK JP",
            "WenQuanYi Zen Hei",
            "Noto Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COL = {
    "ink": "#17212B",
    "muted": "#586575",
    "pale": "#F6F8FA",
    "line": "#AAB6C2",
    "green": "#2D8A62",
    "green2": "#63B57E",
    "green_light": "#DFF0E8",
    "teal": "#2D9C95",
    "teal_light": "#DFF3F0",
    "blue": "#6078B8",
    "blue_light": "#E5EBF8",
    "violet": "#8173B5",
    "violet_light": "#ECE8F7",
    "amber": "#C78F2E",
    "amber_light": "#F7E9C9",
    "red": "#BE5B5B",
    "red_light": "#F4DEDC",
    "soil": "#9B765C",
}


def arrow(ax, start, end, color=None, lw=1.15, rad=0.0, alpha=1.0, ms=9):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or COL["line"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
        alpha=alpha,
        zorder=6,
    )
    ax.add_patch(arr)
    return arr


def label(ax, x, y, text, size=7.0, weight="normal", color=None, ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COL["ink"],
        ha=ha,
        va=va,
        linespacing=1.12,
        zorder=20,
    )


def leaf(ax, x, y, length, width, angle, fc, ec=None, alpha=1.0, z=8):
    ax.add_patch(
        Ellipse(
            (x, y),
            length,
            width,
            angle=angle,
            facecolor=fc,
            edgecolor=ec or "#3F8055",
            linewidth=0.45,
            alpha=alpha,
            zorder=z,
        )
    )


def draw_potted_plant(ax, cx, cy, scale=1.0, alpha=1.0, z=8, mesh=False):
    stem_col = "#3F7E55" if not mesh else COL["teal"]
    pot = Polygon(
        [
            (cx - 0.34 * scale, cy - 0.72 * scale),
            (cx + 0.34 * scale, cy - 0.72 * scale),
            (cx + 0.25 * scale, cy - 1.05 * scale),
            (cx - 0.25 * scale, cy - 1.05 * scale),
        ],
        closed=True,
        facecolor=COL["soil"] if not mesh else "#FFFFFF",
        edgecolor="#725741" if not mesh else COL["teal"],
        linewidth=0.7,
        alpha=0.85 * alpha,
        zorder=z,
    )
    ax.add_patch(pot)
    ax.add_patch(
        Rectangle(
            (cx - 0.38 * scale, cy - 0.74 * scale),
            0.76 * scale,
            0.10 * scale,
            facecolor="#B39174" if not mesh else COL["teal_light"],
            edgecolor="#725741" if not mesh else COL["teal"],
            linewidth=0.6,
            alpha=0.85 * alpha,
            zorder=z + 1,
        )
    )
    stems = [
        ((0.00, -0.62), (0.00, 0.35)),
        ((0.00, -0.30), (-0.26, 0.15)),
        ((0.00, -0.25), (0.28, 0.18)),
        ((0.00, -0.02), (-0.12, 0.45)),
        ((0.00, 0.00), (0.20, 0.48)),
    ]
    for (x1, y1), (x2, y2) in stems:
        ax.plot(
            [cx + x1 * scale, cx + x2 * scale],
            [cy + y1 * scale, cy + y2 * scale],
            color=stem_col,
            lw=1.15 * scale,
            alpha=alpha,
            zorder=z + 2,
        )
    if mesh:
        mesh_leaf(ax, cx - 0.36 * scale, cy + 0.17 * scale, 0.56 * scale, 0.20 * scale, 24, z=z + 4)
        mesh_leaf(ax, cx + 0.38 * scale, cy + 0.20 * scale, 0.58 * scale, 0.19 * scale, -28, z=z + 4)
        mesh_leaf(ax, cx - 0.13 * scale, cy + 0.53 * scale, 0.52 * scale, 0.18 * scale, 74, z=z + 4)
        mesh_leaf(ax, cx + 0.21 * scale, cy + 0.52 * scale, 0.48 * scale, 0.17 * scale, 52, z=z + 4)
        mesh_leaf(ax, cx + 0.04 * scale, cy + 0.31 * scale, 0.48 * scale, 0.18 * scale, -5, z=z + 4)
    else:
        leaf(ax, cx - 0.36 * scale, cy + 0.17 * scale, 0.56 * scale, 0.20 * scale, 24, "#68B783", alpha=alpha, z=z + 4)
        leaf(ax, cx + 0.38 * scale, cy + 0.20 * scale, 0.58 * scale, 0.19 * scale, -28, "#5AA773", alpha=alpha, z=z + 4)
        leaf(ax, cx - 0.13 * scale, cy + 0.53 * scale, 0.52 * scale, 0.18 * scale, 74, "#77C08B", alpha=alpha, z=z + 4)
        leaf(ax, cx + 0.21 * scale, cy + 0.52 * scale, 0.48 * scale, 0.17 * scale, 52, "#64B77E", alpha=alpha, z=z + 4)
        leaf(ax, cx + 0.04 * scale, cy + 0.31 * scale, 0.48 * scale, 0.18 * scale, -5, "#5EAE79", alpha=alpha, z=z + 4)


def mesh_leaf(ax, x, y, length, width, angle, z=12):
    theta = math.radians(angle)
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -math.sin(theta), math.cos(theta)
    pts = []
    for t, side in [(-0.50, 0), (-0.22, 0.48), (0.10, 0.34), (0.50, 0), (0.10, -0.34), (-0.22, -0.48)]:
        px = x + t * length * ux + side * width * vx
        py = y + t * length * uy + side * width * vy
        pts.append((px, py))
    tris = [(0, 1, 2), (0, 2, 5), (2, 3, 4), (2, 4, 5)]
    for tri in tris:
        ax.add_patch(
            Polygon(
                [pts[i] for i in tri],
                closed=True,
                facecolor=COL["teal_light"],
                edgecolor=COL["teal"],
                linewidth=0.55,
                zorder=z,
            )
        )
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.018, facecolor=COL["teal"], edgecolor="none", zorder=z + 1))


def draw_camera(ax, x, y, angle=0, scale=1.0):
    theta = math.radians(angle)
    # field of view points towards center-ish.
    cone_len = 0.60 * scale
    aperture = 0.25 * scale
    tip = (x + cone_len * math.cos(theta), y + cone_len * math.sin(theta))
    p1 = (x + aperture * math.cos(theta + 1.15), y + aperture * math.sin(theta + 1.15))
    p2 = (x + aperture * math.cos(theta - 1.15), y + aperture * math.sin(theta - 1.15))
    ax.add_patch(Polygon([p1, p2, tip], closed=True, facecolor=COL["blue_light"], edgecolor="none", alpha=0.65, zorder=2))
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.15 * scale, y - 0.10 * scale),
            0.30 * scale,
            0.20 * scale,
            boxstyle="round,pad=0.004,rounding_size=0.025",
            facecolor="#FFFFFF",
            edgecolor=COL["blue"],
            linewidth=0.7,
            zorder=8,
        )
    )
    ax.add_patch(Circle((x, y), 0.052 * scale, facecolor=COL["blue"], edgecolor="none", zorder=9))


def draw_image_card(ax, x, y, w, h, mode="raw", z=7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.045",
            facecolor="#FFFFFF",
            edgecolor=COL["line"],
            linewidth=0.65,
            zorder=z,
        )
    )
    ax.add_patch(Rectangle((x + 0.05 * w, y + 0.10 * h), 0.90 * w, 0.78 * h, facecolor="#F1F4F6", edgecolor="none", zorder=z + 1))
    if mode == "mask":
        ax.add_patch(Rectangle((x + 0.05 * w, y + 0.10 * h), 0.90 * w, 0.78 * h, facecolor="#EBEEF1", edgecolor="none", zorder=z + 2))
        draw_mask_shape(ax, x + 0.50 * w, y + 0.47 * h, 0.70 * min(w, h), alpha=0.92, z=z + 3)
    elif mode == "fg":
        draw_mask_shape(ax, x + 0.50 * w, y + 0.47 * h, 0.64 * min(w, h), alpha=0.90, z=z + 3)
        ax.add_patch(Polygon([(x + 0.42 * w, y + 0.22 * h), (x + 0.58 * w, y + 0.22 * h), (x + 0.55 * w, y + 0.14 * h), (x + 0.45 * w, y + 0.14 * h)], facecolor="#CFD5DB", edgecolor="none", alpha=0.55, zorder=z + 2))
    else:
        draw_potted_plant(ax, x + 0.50 * w, y + 0.70 * h, 0.34 * min(w, h), z=z + 2)


def draw_mask_shape(ax, cx, cy, scale=1.0, alpha=0.85, z=8):
    verts = [
        (cx - 0.12 * scale, cy - 0.42 * scale),
        (cx - 0.42 * scale, cy - 0.18 * scale),
        (cx - 0.34 * scale, cy + 0.15 * scale),
        (cx - 0.06 * scale, cy + 0.20 * scale),
        (cx + 0.02 * scale, cy + 0.45 * scale),
        (cx + 0.25 * scale, cy + 0.22 * scale),
        (cx + 0.43 * scale, cy + 0.00 * scale),
        (cx + 0.20 * scale, cy - 0.26 * scale),
        (cx + 0.05 * scale, cy - 0.43 * scale),
        (cx - 0.12 * scale, cy - 0.42 * scale),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(verts) - 1)
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=COL["green"],
            edgecolor="#1F704F",
            linewidth=0.6,
            alpha=alpha,
            zorder=z,
        )
    )


def draw_gaussian_plant(ax, cx, cy, scale=1.0):
    # ghost background/pot, indicating suppressed non-plant geometry.
    ax.add_patch(
        Polygon(
            [
                (cx - 0.36 * scale, cy - 0.86 * scale),
                (cx + 0.36 * scale, cy - 0.86 * scale),
                (cx + 0.27 * scale, cy - 1.15 * scale),
                (cx - 0.27 * scale, cy - 1.15 * scale),
            ],
            facecolor=COL["red_light"],
            edgecolor=COL["red"],
            linewidth=0.55,
            alpha=0.36,
            zorder=3,
        )
    )
    ax.plot([cx - 0.45 * scale, cx + 0.45 * scale], [cy - 1.00 * scale, cy - 0.72 * scale], color=COL["red"], lw=0.8, alpha=0.55, zorder=4)
    ax.plot([cx - 0.45 * scale, cx + 0.45 * scale], [cy - 0.72 * scale, cy - 1.00 * scale], color=COL["red"], lw=0.8, alpha=0.55, zorder=4)
    # mask envelope.
    ax.add_patch(
        Ellipse(
            (cx, cy - 0.02 * scale),
            1.55 * scale,
            1.95 * scale,
            angle=-5,
            facecolor=COL["green_light"],
            edgecolor=COL["green"],
            linewidth=0.8,
            alpha=0.42,
            zorder=4,
        )
    )
    # Gaussian discs placed on plant.
    discs = [
        (-0.34, 0.12, 24, 0.34, 0.13, COL["green2"]),
        (0.32, 0.16, -27, 0.34, 0.12, COL["green"]),
        (-0.10, 0.55, 72, 0.32, 0.11, COL["teal"]),
        (0.20, 0.50, 50, 0.30, 0.11, COL["green2"]),
        (0.02, 0.32, -4, 0.33, 0.12, COL["teal"]),
        (-0.03, -0.14, 84, 0.34, 0.08, COL["blue"]),
        (0.10, -0.28, 58, 0.28, 0.08, COL["blue"]),
    ]
    for dx, dy, ang, ww, hh, col in discs:
        ax.add_patch(
            Ellipse(
                (cx + dx * scale, cy + dy * scale),
                ww * scale,
                hh * scale,
                angle=ang,
                facecolor=col,
                edgecolor="#FFFFFF",
                linewidth=0.45,
                alpha=0.78,
                zorder=8,
            )
        )
    for px, py in [(0, -0.50), (-0.09, -0.24), (0.08, -0.18), (0.02, 0.05)]:
        ax.add_patch(
            Ellipse(
                (cx + px * scale, cy + py * scale),
                0.10 * scale,
                0.26 * scale,
                angle=5,
                facecolor=COL["blue"],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.76,
                zorder=7,
            )
        )


def draw_measurements(ax, cx, cy, scale=1.0):
    # height arrow
    ax.add_patch(
        FancyArrowPatch(
            (cx - 0.88 * scale, cy - 0.86 * scale),
            (cx - 0.88 * scale, cy + 0.70 * scale),
            arrowstyle="<->",
            mutation_scale=8,
            color=COL["amber"],
            linewidth=1.0,
            zorder=16,
        )
    )
    label(ax, cx - 1.02 * scale, cy - 0.08 * scale, "株高", size=6.1, color=COL["amber"], ha="right")
    # width arrow
    ax.add_patch(
        FancyArrowPatch(
            (cx - 0.56 * scale, cy - 1.22 * scale),
            (cx + 0.58 * scale, cy - 1.22 * scale),
            arrowstyle="<->",
            mutation_scale=8,
            color=COL["blue"],
            linewidth=1.0,
            zorder=16,
        )
    )
    label(ax, cx, cy - 1.38 * scale, "冠幅", size=6.1, color=COL["blue"])
    # leaf arrow
    ax.add_patch(
        FancyArrowPatch(
            (cx + 0.16 * scale, cy + 0.40 * scale),
            (cx + 0.55 * scale, cy + 0.62 * scale),
            arrowstyle="<->",
            mutation_scale=7,
            color=COL["green"],
            linewidth=0.9,
            zorder=16,
        )
    )
    label(ax, cx + 0.73 * scale, cy + 0.62 * scale, "叶长/叶宽", size=6.0, color=COL["green"], ha="left")


def add_soft_blob(ax, cx, cy, w, h, color, alpha=0.18, z=0):
    for i, a in enumerate([alpha * 0.55, alpha * 0.80, alpha]):
        ax.add_patch(
            Ellipse(
                (cx, cy),
                w * (1 - i * 0.18),
                h * (1 - i * 0.16),
                facecolor=color,
                edgecolor="none",
                alpha=a,
                zorder=z,
            )
        )


def main():
    fig_w = 183 / 25.4
    fig_h = 108 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 14.7)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    add_soft_blob(ax, 2.1, 3.42, 4.1, 3.65, COL["blue_light"], 0.55)
    add_soft_blob(ax, 5.6, 3.38, 4.0, 3.55, COL["green_light"], 0.52)
    add_soft_blob(ax, 9.0, 3.38, 4.1, 3.60, COL["violet_light"], 0.48)
    add_soft_blob(ax, 12.2, 3.30, 3.5, 3.45, COL["teal_light"], 0.55)

    label(ax, 0.40, 6.63, "ForeSplat 方法总览", size=11.0, weight="bold", ha="left")
    label(ax, 0.40, 6.28, "前景先验从图像进入三维优化，最终形成仅含植物的网格与虚拟表型值。", size=6.8, color=COL["muted"], ha="left")

    # Left scene: acquisition.
    draw_potted_plant(ax, 2.0, 3.55, 0.98, z=8)
    for x, y, ang in [(0.70, 4.65, -24), (0.92, 2.24, 20), (3.32, 4.66, -155), (3.24, 2.26, 156)]:
        draw_camera(ax, x, y, ang, 1.0)
    ax.add_patch(Arc((2.0, 3.58), 2.45, 2.45, theta1=20, theta2=340, edgecolor=COL["blue"], linewidth=0.75, linestyle=(0, (2, 2)), alpha=0.72, zorder=3))
    label(ax, 2.0, 1.16, "普通多视角 RGB 采集", size=7.4, weight="bold")
    label(ax, 2.0, 0.86, "固定装置或手持环绕", size=6.0, color=COL["muted"])

    # Image cards and foreground prior.
    draw_image_card(ax, 4.15, 4.10, 1.15, 1.03, "raw", z=8)
    draw_image_card(ax, 4.75, 3.10, 1.15, 1.03, "mask", z=8)
    draw_image_card(ax, 5.35, 2.10, 1.15, 1.03, "fg", z=8)
    arrow(ax, (3.38, 3.75), (4.10, 4.50), COL["line"], lw=1.0, rad=0.03)
    label(ax, 5.30, 5.48, "FSAM3 前景先验", size=7.4, weight="bold")
    label(ax, 5.30, 5.18, "FFT 筛帧  ·  SAM3 分割  ·  PCA 精炼", size=5.8, color=COL["muted"])
    label(ax, 6.34, 2.24, "仅前景 RGB\n与二值掩膜", size=5.8, color=COL["green"], ha="left")

    # Central Gaussian representation.
    arrow(ax, (6.48, 3.20), (7.54, 3.58), COL["green"], lw=1.25, rad=-0.05)
    draw_gaussian_plant(ax, 8.55, 3.55, 1.05)
    label(ax, 8.52, 5.55, "前景对象 2DGS 优化", size=7.4, weight="bold")
    label(ax, 8.52, 5.24, "RGB 监督、透明度约束和视角权重\n都绑定到植物前景", size=5.8, color=COL["muted"])
    ax.add_patch(
        FancyBboxPatch(
            (7.55, 1.34),
            2.18,
            0.48,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            facecolor="#FFFFFF",
            edgecolor=COL["violet"],
            linewidth=0.65,
            zorder=12,
        )
    )
    label(ax, 8.64, 1.58, "L_rgb  →  L_rgb-fg", size=6.7, color=COL["violet"])
    label(ax, 9.95, 2.08, "背景/花盆\n不再主导容量", size=5.7, color=COL["red"], ha="left")

    # Mesh and phenotyping.
    arrow(ax, (9.68, 3.55), (11.02, 3.45), COL["teal"], lw=1.25, rad=0.04)
    draw_potted_plant(ax, 12.0, 3.55, 0.95, z=10, mesh=True)
    draw_measurements(ax, 12.0, 3.55, 0.95)
    label(ax, 12.05, 5.55, "TSDF 网格化与表型测量", size=7.4, weight="bold")
    label(ax, 12.05, 5.24, "plant-only mesh  ·  virtual traits", size=5.8, color=COL["muted"])

    # Compact value chips as visual annotations, not flow boxes.
    chips = [
        (11.05, 1.15, "仅含植物网格", COL["teal_light"], COL["teal"]),
        (12.47, 1.15, "株高/冠幅", COL["amber_light"], COL["amber"]),
        (13.58, 1.15, "叶长/叶宽", COL["green_light"], COL["green"]),
    ]
    for x, y, txt, fc, ec in chips:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.98,
                0.34,
                boxstyle="round,pad=0.008,rounding_size=0.05",
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.55,
                zorder=15,
            )
        )
        label(ax, x + 0.49, y + 0.17, txt, size=5.5, color=COL["ink"])

    # One quiet conceptual sentence for reviewers.
    ax.plot([0.48, 14.10], [0.55, 0.55], color="#D9E0E7", lw=0.6, zorder=1)
    label(ax, 0.52, 0.30, "关键设计：掩膜不是后处理过滤器，而是在训练期定义待测植物对象。", size=6.0, color=COL["muted"], ha="left")
    label(ax, 14.18, 0.30, "Fig. 1", size=6.0, color=COL["muted"], ha="right")

    for p in [SVG_PRIMARY, SVG_ALT]:
        fig.savefig(p, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(PNG_PREVIEW, dpi=240, bbox_inches="tight", pad_inches=0.035)
    print(SVG_PRIMARY)
    print(SVG_ALT)
    print(PNG_PREVIEW)


if __name__ == "__main__":
    main()
