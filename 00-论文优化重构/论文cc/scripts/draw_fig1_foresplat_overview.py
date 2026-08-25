#!/usr/bin/env python3
from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
    Ellipse,
    PathPatch,
    Rectangle,
    Polygon,
    Arc,
)
from matplotlib.path import Path as MplPath


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "fig1_foresplat_overview.svg"
QA_PREVIEW_FILE = OUT_DIR / "fig1_foresplat_overview_preview.png"


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
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COL = {
    "ink": "#1F2933",
    "muted": "#586575",
    "line": "#A8B2BE",
    "grid": "#DDE4EA",
    "paper": "#FFFFFF",
    "neutral_fill": "#F5F7FA",
    "green": "#2F8F68",
    "green_light": "#DFF0E8",
    "teal": "#2A9D8F",
    "teal_light": "#DDF2F0",
    "blue": "#5D75B8",
    "blue_light": "#E7ECF8",
    "violet": "#7C6FB0",
    "violet_light": "#ECE8F6",
    "amber": "#C9912A",
    "amber_light": "#F7EACB",
    "red": "#C75D5D",
    "red_light": "#F6E1DE",
}


def add_box(ax, x, y, w, h, title, subtitle=None, fc="#FFFFFF", ec=None, lw=0.9):
    ec = ec or COL["line"]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.05",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.70,
        title,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=COL["ink"],
        zorder=5,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.2,
            color=COL["muted"],
            linespacing=1.12,
            zorder=5,
        )
    return box


def arrow(ax, start, end, color=None, lw=1.2, rad=0.0, mutation=8):
    color = color or COL["line"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=4,
    )
    ax.add_patch(arr)
    return arr


def draw_image_stack(ax, cx, cy, scale=1.0):
    w, h = 0.82 * scale, 0.58 * scale
    offsets = [(-0.16, 0.12), (0.0, 0.02), (0.16, -0.08)]
    for k, (dx, dy) in enumerate(offsets):
        x = cx - w / 2 + dx * scale
        y = cy - h / 2 + dy * scale
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.03",
            linewidth=0.55,
            edgecolor=COL["line"],
            facecolor="#F3F5F7",
            zorder=3 + k,
        )
        ax.add_patch(rect)
        # pot
        ax.add_patch(
            Polygon(
                [
                    (cx - 0.10 * scale + dx * scale, cy - 0.22 * scale + dy * scale),
                    (cx + 0.10 * scale + dx * scale, cy - 0.22 * scale + dy * scale),
                    (cx + 0.07 * scale + dx * scale, cy - 0.34 * scale + dy * scale),
                    (cx - 0.07 * scale + dx * scale, cy - 0.34 * scale + dy * scale),
                ],
                closed=True,
                facecolor="#B58B6A",
                edgecolor="#8A6B55",
                linewidth=0.4,
                zorder=5 + k,
            )
        )
        # leaves
        for ang, ox, oy, c in [
            (28, -0.09, 0.00, "#5FA878"),
            (-30, 0.11, 0.02, "#4F9A66"),
            (82, 0.00, 0.05, "#6DBB81"),
        ]:
            ax.add_patch(
                Ellipse(
                    (cx + (ox + dx) * scale, cy + (oy + dy) * scale),
                    0.22 * scale,
                    0.08 * scale,
                    angle=ang,
                    facecolor=c,
                    edgecolor="#3A7E51",
                    linewidth=0.35,
                    zorder=6 + k,
                )
            )


def draw_mask_icon(ax, cx, cy, scale=1.0):
    verts = [
        (cx - 0.18 * scale, cy - 0.24 * scale),
        (cx - 0.32 * scale, cy + 0.02 * scale),
        (cx - 0.15 * scale, cy + 0.27 * scale),
        (cx + 0.10 * scale, cy + 0.23 * scale),
        (cx + 0.29 * scale, cy + 0.03 * scale),
        (cx + 0.16 * scale, cy - 0.25 * scale),
        (cx - 0.18 * scale, cy - 0.24 * scale),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
    ]
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=COL["green"],
        edgecolor="#1E6F4F",
        linewidth=0.55,
        alpha=0.90,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.add_patch(
        Circle(
            (cx + 0.18 * scale, cy + 0.18 * scale),
            0.05 * scale,
            facecolor=COL["amber"],
            edgecolor="none",
            zorder=6,
        )
    )


def draw_camera_pose(ax, cx, cy, scale=1.0):
    for ang in [210, 270, 330]:
        rad = math.radians(ang)
        x = cx + 0.33 * scale * math.cos(rad)
        y = cy + 0.24 * scale * math.sin(rad)
        ax.add_patch(Circle((x, y), 0.035 * scale, facecolor=COL["blue"], edgecolor="none", zorder=6))
        ax.plot([x, cx], [y, cy], color=COL["blue"], linewidth=0.45, alpha=0.65, zorder=5)
    for px, py in [(-0.10, 0.02), (-0.02, 0.12), (0.09, 0.05), (0.06, -0.10), (-0.12, -0.10)]:
        ax.add_patch(
            Circle(
                (cx + px * scale, cy + py * scale),
                0.018 * scale,
                facecolor=COL["ink"],
                edgecolor="none",
                alpha=0.75,
                zorder=6,
            )
        )
    ax.add_patch(Arc((cx, cy), 0.85 * scale, 0.56 * scale, theta1=195, theta2=345, color=COL["line"], lw=0.55))


def draw_gaussians(ax, cx, cy, scale=1.0):
    specs = [
        (-0.18, -0.02, 32, COL["green"]),
        (-0.04, 0.10, -18, COL["teal"]),
        (0.14, 0.04, 22, COL["green"]),
        (0.06, -0.13, -34, COL["blue"]),
        (-0.12, -0.15, 8, COL["teal"]),
    ]
    for dx, dy, ang, col in specs:
        ax.add_patch(
            Ellipse(
                (cx + dx * scale, cy + dy * scale),
                0.24 * scale,
                0.09 * scale,
                angle=ang,
                facecolor=col,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.80,
                zorder=6,
            )
        )


def draw_weight_icon(ax, cx, cy, scale=1.0):
    xs = [-0.24, -0.10, 0.04, 0.18]
    hs = [0.18, 0.31, 0.25, 0.38]
    for x, h in zip(xs, hs):
        ax.add_patch(
            Rectangle(
                (cx + x * scale, cy - 0.18 * scale),
                0.08 * scale,
                h * scale,
                facecolor=COL["amber"],
                edgecolor="#A06F1F",
                linewidth=0.45,
                alpha=0.8,
                zorder=6,
            )
        )
    ax.text(cx + 0.17 * scale, cy + 0.27 * scale, "q_i", ha="center", va="center", fontsize=7.5, color=COL["ink"])


def draw_pruning_icon(ax, cx, cy, scale=1.0):
    for dx, dy, col, keep in [
        (-0.20, 0.10, COL["green"], True),
        (-0.02, 0.16, COL["teal"], True),
        (0.18, 0.04, COL["red"], False),
        (-0.08, -0.12, COL["blue"], True),
        (0.16, -0.16, COL["red"], False),
    ]:
        ax.add_patch(
            Circle(
                (cx + dx * scale, cy + dy * scale),
                0.055 * scale,
                facecolor=col,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.85,
                zorder=6,
            )
        )
        if not keep:
            ax.plot(
                [cx + (dx - 0.045) * scale, cx + (dx + 0.045) * scale],
                [cy + (dy - 0.045) * scale, cy + (dy + 0.045) * scale],
                color="#7C2D2D",
                lw=0.8,
                zorder=7,
            )
            ax.plot(
                [cx + (dx - 0.045) * scale, cx + (dx + 0.045) * scale],
                [cy + (dy + 0.045) * scale, cy + (dy - 0.045) * scale],
                color="#7C2D2D",
                lw=0.8,
                zorder=7,
            )


def draw_mesh_icon(ax, cx, cy, scale=1.0):
    pts = [
        (-0.23, -0.12),
        (-0.08, 0.13),
        (0.08, -0.02),
        (0.23, 0.14),
        (0.17, -0.19),
        (-0.10, -0.22),
    ]
    pts = [(cx + x * scale, cy + y * scale) for x, y in pts]
    tris = [(0, 1, 2), (2, 3, 4), (0, 2, 5), (2, 4, 5)]
    for tri in tris:
        ax.add_patch(
            Polygon(
                [pts[i] for i in tri],
                closed=True,
                facecolor=COL["teal_light"],
                edgecolor=COL["teal"],
                linewidth=0.65,
                zorder=6,
            )
        )
    for p in pts:
        ax.add_patch(Circle(p, 0.018 * scale, facecolor=COL["teal"], edgecolor="none", zorder=7))


def draw_traits_icon(ax, cx, cy, scale=1.0):
    # plant silhouette
    ax.plot([cx, cx], [cy - 0.22 * scale, cy + 0.12 * scale], color=COL["green"], lw=1.1, zorder=6)
    for ang, ox, oy in [(32, -0.10, 0.02), (-35, 0.10, 0.04), (72, -0.02, 0.13)]:
        ax.add_patch(
            Ellipse(
                (cx + ox * scale, cy + oy * scale),
                0.21 * scale,
                0.075 * scale,
                angle=ang,
                facecolor="#68B784",
                edgecolor="#3B8458",
                linewidth=0.35,
                zorder=6,
            )
        )
    # height and width rulers
    ax.plot([cx - 0.31 * scale, cx - 0.31 * scale], [cy - 0.24 * scale, cy + 0.22 * scale], color=COL["amber"], lw=0.9)
    ax.plot([cx - 0.34 * scale, cx - 0.28 * scale], [cy - 0.24 * scale, cy - 0.24 * scale], color=COL["amber"], lw=0.9)
    ax.plot([cx - 0.34 * scale, cx - 0.28 * scale], [cy + 0.22 * scale, cy + 0.22 * scale], color=COL["amber"], lw=0.9)
    ax.plot([cx - 0.19 * scale, cx + 0.22 * scale], [cy - 0.30 * scale, cy - 0.30 * scale], color=COL["blue"], lw=0.9)
    ax.plot([cx - 0.19 * scale, cx - 0.19 * scale], [cy - 0.33 * scale, cy - 0.27 * scale], color=COL["blue"], lw=0.9)
    ax.plot([cx + 0.22 * scale, cx + 0.22 * scale], [cy - 0.33 * scale, cy - 0.27 * scale], color=COL["blue"], lw=0.9)


def add_micro_steps(ax, x, y, labels, colors):
    width = 0.78
    gap = 0.06
    for i, lab in enumerate(labels):
        bx = x + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (bx, y),
                width,
                0.26,
                boxstyle="round,pad=0.005,rounding_size=0.025",
                facecolor=colors[i],
                edgecolor="none",
                zorder=5,
            )
        )
        ax.text(bx + width / 2, y + 0.13, lab, ha="center", va="center", fontsize=5.6, color=COL["ink"], zorder=6)


def main():
    # Double-column, broad method figure: 183 mm x 112 mm.
    fig_w = 183 / 25.4
    fig_h = 112 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(0.18, 6.92, "ForeSplat/F2DMAS：前景感知 2DGS 植物表型流程", fontsize=10.2, fontweight="bold", color=COL["ink"], va="top")
    ax.text(
        0.18,
        6.58,
        "从普通多视角 RGB 到仅含植物的可测量网格：前景先验贯穿初始化、训练、剪枝、网格化和表型测量。",
        fontsize=6.8,
        color=COL["muted"],
        va="top",
    )

    # Background bands.
    bands = [
        (0.15, 4.40, 14.3, 1.70, "#FAFBFC", "输入与前景先验"),
        (0.15, 2.25, 14.3, 1.75, "#FFFFFF", "前景对象 2DGS 优化"),
        (0.15, 0.42, 14.3, 1.45, "#FAFBFC", "导出与表型测量"),
    ]
    for x, y, w, h, fc, lab in bands:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=COL["grid"], linewidth=0.45, zorder=0))
        ax.text(x + 0.12, y + h - 0.14, lab, fontsize=6.2, color=COL["muted"], va="top", zorder=1)

    # Main workflow boxes.
    box_y = 4.72
    boxes = [
        (0.45, box_y, 1.35, 0.92, "原始多视角\nRGB", "普通手机/固定装置\n环绕采集", COL["neutral_fill"], COL["line"]),
        (2.25, box_y, 1.75, 0.92, "FSAM3", "FFT 筛帧 + SAM3\n+ PCA 主前景", COL["green_light"], COL["green"]),
        (4.45, box_y, 1.55, 0.92, "COLMAP", "位姿估计\n稀疏点轨迹", COL["blue_light"], COL["blue"]),
        (6.45, box_y, 2.05, 0.92, "植物感知 2DGS", "前景初始化\n前景 RGB 监督", COL["violet_light"], COL["violet"]),
        (8.95, box_y, 1.75, 0.92, "软视角加权", "低质量视角不删除\n仅调节损失贡献", COL["amber_light"], COL["amber"]),
        (11.15, box_y, 1.85, 0.92, "掩膜引导\nGaussian 剪枝", "多视角掩膜支持\n去除背景残留", COL["green_light"], COL["green"]),
        (13.35, box_y, 0.88, 0.92, "输出", "网格\n性状", COL["teal_light"], COL["teal"]),
    ]
    for b in boxes:
        add_box(ax, *b)

    centers = [(x + w / 2, y + h / 2) for x, y, w, h, *_ in boxes]
    for c1, c2 in zip(centers[:-1], centers[1:]):
        arrow(ax, (c1[0] + 0.70, c1[1]), (c2[0] - 0.75, c2[1]), color=COL["line"], lw=1.15)

    # Icons in boxes.
    draw_image_stack(ax, 1.12, 5.06, 0.75)
    draw_mask_icon(ax, 3.12, 5.00, 0.83)
    draw_camera_pose(ax, 5.22, 5.02, 0.90)
    draw_gaussians(ax, 7.48, 5.01, 0.92)
    draw_weight_icon(ax, 9.82, 5.02, 0.92)
    draw_pruning_icon(ax, 12.08, 5.02, 0.95)

    # FSAM3 micro outputs.
    add_micro_steps(
        ax,
        2.05,
        4.42,
        ["有效帧", "二值掩膜", "前景 RGB"],
        ["#EEF2F5", COL["green_light"], COL["teal_light"]],
    )
    arrow(ax, (3.12, 4.73), (7.10, 3.80), color=COL["green"], lw=1.0, rad=-0.18, mutation=7)
    ax.text(4.05, 4.05, "掩膜定义待测植物对象", fontsize=6.1, color=COL["green"], ha="left", va="center")

    # Central training objective as hero panel.
    hero = FancyBboxPatch(
        (4.75, 2.55),
        5.55,
        1.12,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor="#FFFFFF",
        edgecolor=COL["violet"],
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(hero)
    ax.text(5.02, 3.46, "训练期前景对象约束", fontsize=7.8, fontweight="bold", color=COL["ink"], va="center")
    ax.text(
        5.02,
        3.18,
        "主光度梯度来自植物像素，背景不再与植株竞争 Gaussian 容量。",
        fontsize=6.1,
        color=COL["muted"],
        va="center",
    )
    sub_x = [5.05, 6.42, 7.82, 9.10]
    subs = [
        ("前景轨迹\n初始化", COL["blue_light"], COL["blue"]),
        ("L_rgb-fg", COL["green_light"], COL["green"]),
        ("alpha 掩膜\n一致性", COL["violet_light"], COL["violet"]),
        ("背景不透明度\n抑制", COL["red_light"], COL["red"]),
    ]
    for x, (lab, fc, ec) in zip(sub_x, subs):
        ax.add_patch(
            FancyBboxPatch(
                (x, 2.70),
                1.08,
                0.34,
                boxstyle="round,pad=0.007,rounding_size=0.035",
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.55,
                zorder=4,
            )
        )
        ax.text(x + 0.54, 2.87, lab, fontsize=5.9, ha="center", va="center", color=COL["ink"], linespacing=1.0, zorder=5)

    # Soft view weighting and pruning links to hero.
    arrow(ax, (9.82, 4.72), (9.65, 3.67), color=COL["amber"], lw=1.0, rad=-0.05, mutation=7)
    ax.text(10.02, 3.96, "q_i 加权 RGB 损失", fontsize=5.8, color=COL["amber"], va="center")
    arrow(ax, (8.35, 2.55), (11.65, 1.62), color=COL["green"], lw=1.0, rad=0.08, mutation=7)
    ax.text(9.60, 2.05, "训练后清理弱支撑基元", fontsize=5.8, color=COL["green"], va="center")

    # Output panels.
    mesh_box = add_box(ax, 9.95, 0.78, 1.62, 0.76, "TSDF 网格化", "深度融合\nMarching Cubes", COL["teal_light"], COL["teal"], lw=0.85)
    trait_box = add_box(ax, 12.05, 0.78, 1.85, 0.76, "虚拟表型值", "株高 / 冠幅\n叶长 / 叶宽", COL["amber_light"], COL["amber"], lw=0.85)
    draw_mesh_icon(ax, 10.76, 1.00, 0.86)
    draw_traits_icon(ax, 12.98, 1.03, 0.82)
    arrow(ax, (12.08, 4.72), (10.78, 1.55), color=COL["green"], lw=1.05, rad=0.06, mutation=7)
    arrow(ax, (11.58, 1.16), (12.04, 1.16), color=COL["line"], lw=1.05, mutation=7)

    # Plant-only representation callout.
    callout = FancyBboxPatch(
        (0.62, 2.58),
        3.25,
        1.05,
        boxstyle="round,pad=0.012,rounding_size=0.05",
        facecolor=COL["green_light"],
        edgecolor=COL["green"],
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(callout)
    ax.text(0.86, 3.36, "核心思想", fontsize=7.6, fontweight="bold", color=COL["ink"], va="center")
    ax.text(
        0.86,
        3.04,
        "掩膜不是后处理过滤器，\n而是定义三维优化对象的先验。",
        fontsize=6.3,
        color=COL["ink"],
        va="center",
        linespacing=1.16,
    )
    ax.text(
        0.86,
        2.69,
        "目标：只重建要测量的植物。",
        fontsize=6.0,
        color=COL["green"],
        va="center",
    )
    arrow(ax, (3.86, 3.10), (4.76, 3.10), color=COL["green"], lw=1.0, mutation=7)

    # Bottom note with reviewable output.
    ax.text(
        0.62,
        0.96,
        "可审查输出",
        fontsize=7.4,
        fontweight="bold",
        color=COL["ink"],
        ha="left",
        va="center",
    )
    note_items = [
        ("仅含植物的 Gaussian 表示", COL["green"]),
        ("紧凑网格", COL["teal"]),
        ("人工-虚拟性状对照", COL["amber"]),
    ]
    x0 = 0.62
    for i, (txt, col) in enumerate(note_items):
        x = x0 + 0.05 + i * 2.65
        ax.add_patch(Circle((x, 0.62), 0.045, facecolor=col, edgecolor="none"))
        ax.text(x + 0.10, 0.62, txt, fontsize=6.0, color=COL["muted"], va="center", ha="left")

    # Small caption-like tag.
    ax.text(14.25, 0.30, "Fig. 1", fontsize=6.2, color=COL["muted"], ha="right", va="bottom")

    fig.savefig(OUT_FILE, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(QA_PREVIEW_FILE, dpi=220, bbox_inches="tight", pad_inches=0.035)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
