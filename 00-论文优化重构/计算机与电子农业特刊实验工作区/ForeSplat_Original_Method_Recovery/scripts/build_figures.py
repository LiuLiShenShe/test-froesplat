#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the figures for the recovery report from real run artefacts only.

Every panel is read out of a run directory -- nothing is drawn by hand except
axis labels.  Figures whose input is not on disk yet are skipped with a warning
so the script can be re-run as the long jobs finish.

    python scripts/build_figures.py            # whatever is available
    python scripts/build_figures.py --only sg  # one figure
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
RUN = RECOVERY / "runs/R6_RAP_FULL_real"
KQZ_RUN = RECOVERY / "runs/KongQueZhuYu_RAP_FULL_210"
FIG = RECOVERY / "figures"
TILE_W = 520
LABEL_H = 26

FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)


def tile(path: Path, label: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    scale = TILE_W / img.width
    img = img.resize((TILE_W, max(1, round(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (TILE_W, img.height + LABEL_H), "white")
    canvas.paste(img, (0, LABEL_H))
    ImageDraw.Draw(canvas).text((6, 4), label, fill="black", font=FONT)
    return canvas


def strip(tiles: list[Image.Image]) -> Image.Image:
    h = max(t.height for t in tiles)
    out = Image.new("RGB", (TILE_W * len(tiles), h), "white")
    for i, t in enumerate(tiles):
        out.paste(t, (i * TILE_W, 0))
    return out


def fig_chain(frame: str = "0075") -> Path:
    """One frame through every RAP stage, left to right."""
    rgb = ROOT / "00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked/KongQueZhuYu/images" / f"{frame}.jpg"
    stages = [
        (rgb, "input RGB"),
        (RUN / "候选掩膜/raw_instance_P1" / f"mask_{frame}_00.png", "P1 best cand."),
        (RUN / "候选掩膜/raw_instance_P2" / f"mask_{frame}_00.png", "P2 best cand."),
        (RUN / "选择后掩膜" / f"mask_{frame}.png", "after selection (SG-MPS+CDR)"),
        (RUN / "正负提示细化掩膜" / f"mask_{frame}.png", "after SB-FBPR"),
        (RUN / "残差修复掩膜" / f"mask_{frame}.png", "after residual repair"),
        (RUN / "最终掩膜" / f"mask_{frame}.png", "final (after GCFC)"),
        (RUN / "叠加图" / f"overlay_{frame}.png", "overlay"),
    ]
    tiles = []
    for path, label in stages:
        if not path.exists():
            print(f"[skip] {path}")
            continue
        tiles.append(tile(path, label))
    out = FIG / f"fig1_rap_stage_chain_{frame}.png"
    strip(tiles).save(out)
    return out


def fig_gcfc(frame: str = "0050") -> Path:
    """GCFC internals: support field -> boundary band -> added pixels."""
    d = RUN / "几何修正提示图"
    stages = [
        (d / f"GCFC_before_{frame}.png", "before"),
        (d / f"GCFC_support_field_{frame}.png", "support field H"),
        (d / f"GCFC_boundary_band_{frame}.png", "boundary band B"),
        (d / f"GCFC_added_pixels_{frame}.png", "added pixels"),
        (d / f"GCFC_after_{frame}.png", "after"),
    ]
    tiles = [tile(p, label) for p, label in stages if p.exists()]
    out = FIG / f"fig2_gcfc_stages_{frame}.png"
    strip(tiles).save(out)
    return out


def fig_sgmps() -> Path:
    """Per-prompt gate confidence vs softmax weight -- the near-tie evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader((RUN / "逐帧报告.csv").open(encoding="utf-8")))
    frames = [r["frame_id"] for r in rows]
    prompts = ["P1", "P2", "P3", "P4", "P5"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    width = 0.16
    for ax, key, title in (
            (axes[0], "sg_mps_gate_confidence", r"gate confidence $c_i^{(k)}$"),
            (axes[1], "sg_mps_softmax", r"prompt-level softmax $a^{(k)}$")):
        for j, prompt in enumerate(prompts):
            vals = []
            for row in rows:
                data = json.loads(row[key] or "{}")
                vals.append(data.get(prompt, np.nan))
            xs = np.arange(len(frames)) + (j - 2) * width
            ax.bar(xs, [0 if np.isnan(v) else v for v in vals], width,
                   label=prompt)
        ax.set_xticks(np.arange(len(frames)))
        ax.set_xticklabels(frames)
        ax.set_xlabel("frame")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.4, 0.68)
    axes[1].legend(ncol=5, fontsize=9)
    axes[1].axhline(0.2, ls="--", lw=1, color="grey")
    fig.suptitle("SG-MPS on KongQueZhuYu: five prompts are within ~0.03 of each "
                 "other, so $\\tau_s$=0.25 makes the gate a near-uniform "
                 "(dashed = uniform 1/5)", fontsize=10)
    fig.tight_layout()
    out = FIG / "fig3_sgmps_confidence_vs_softmax.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_ablation() -> Path | None:
    src = RECOVERY / "metrics/dev_ablation_per_scene.csv"
    if not src.exists():
        print(f"[skip] {src} not written yet")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    order = ["R0_single_P2_base", "R1_single_P2_cdr", "R3n_multi_nogate_cdr",
             "R3g_multi_gate_margin", "R2_multi_SGMPS_base", "R3_multi_SGMPS_cdr",
             "R4b_R3_SPNP_boxes", "R4p_R3_SPNP_points", "R5_R4p_repair",
             "R6_R5_GCFC"]
    scenes = sorted({r["scene"] for r in rows})
    lookup = {(r["config"], r["scene"]): float(r["f1_mean"]) for r in rows}

    fig, ax = plt.subplots(figsize=(12, 4.4))
    width = 0.8 / len(scenes)
    for j, scene in enumerate(scenes):
        vals = [lookup.get((c, scene), np.nan) for c in order]
        xs = np.arange(len(order)) + (j - (len(scenes) - 1) / 2) * width
        ax.bar(xs, [0 if np.isnan(v) else v for v in vals], width, label=scene)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1 vs mask_potted_plant (mean over frames)")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.95, ls=":", lw=1, color="grey")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    ax.set_title("RAP-FSAM3 ablation on the cross-scene dev set "
                 "(16 frames / 4 scenes, disjoint from the frozen KongQueZhuYu set)")
    fig.tight_layout()
    out = FIG / "fig4_dev_ablation_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_downstream_2x2() -> Path | None:
    """The mask-source x M2M3-GO 2x2, four metrics side by side."""
    src = RECOVERY / "metrics/downstream_2x2.csv"
    if not src.exists():
        print(f"[skip] {src} not written yet")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in csv.DictReader(src.open(encoding="utf-8"))
            if r["status"] == "COMPLETE"]
    if not rows:
        print("[skip] no COMPLETE downstream runs yet")
        return None
    labels = [f"{r['mask_source']}\nM2M3-GO={'on' if r['m2m3_go']=='True' else 'off'}"
              for r in rows]
    panels = [("psnr_fg", "PSNR_fg (dB) ↑"), ("ssim_fg", "SSIM_fg ↑"),
              ("lpips_fg", "LPIPS_fg ↓"),
              ("outside_leak", "outside-mask ratio ↓")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    colors = ["#4477aa" if "C_R1" in r["tag"] else "#ee6677" for r in rows]
    for ax, (key, title) in zip(axes, panels):
        vals = [float(r[key]) for r in rows]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        lo = min(vals)
        ax.set_ylim(lo * 0.97 if lo > 0.05 else 0, max(vals) * 1.03)
    fig.suptitle("Downstream 2x2 on KongQueZhuYu (30k, res 2, 27 test views): "
                 "blue = frozen C_R1 masks, red = full RAP masks", fontsize=10)
    fig.tight_layout()
    out = FIG / "fig5_downstream_2x2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_mesh_silhouette() -> Path | None:
    """Two side-by-side plant silhouettes (COLMAP units) + trait deltas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = RECOVERY / "metrics/mesh_geometry.json"
    if not src.exists():
        print(f"[skip] {src} not written yet")
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    tags = [t["tag"] for t in data["tags"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, tag, yaxis in ((axes[0], "C_R1_baseline", "span_a"),
                           (axes[1], "RAP_FULL", "span_a")):
        npz = np.load(RECOVERY / "metrics" / f"mesh_silhouette_{tag}.npz")
        x = npz["span_a"] if yaxis == "span_a" else npz["span_b"]
        y = npz["up"]
        ax.scatter(x, y, s=0.35, c="#334b5c", alpha=0.25, linewidths=0)
        ax.set_title(f"{tag}\nheight = "
                     f"{next(t['height_along_up_units'] for t in data['tags'] if t['tag']==tag):.3f} u, "
                     f"span = "
                     f"{next(t['canopy_span_max_units'] for t in data['tags'] if t['tag']==tag):.3f} u",
                     fontsize=10)
        ax.set_xlabel("canopy span (COLMAP units)")
        ax.set_ylabel("height along camera-up (units)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
    ax = axes[2]
    labels = tags
    heights = [next(t["height_along_up_units"] for t in data["tags"] if t["tag"] == g)
               for g in labels]
    spans = [next(t["canopy_span_max_units"] for t in data["tags"] if t["tag"] == g)
             for g in labels]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, heights, 0.4, label="height (u)", color="#4477aa")
    ax.bar(xs + 0.2, spans, 0.4, label="max span (u)", color="#ee6677")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("COLMAP units (not cm)")
    ax.set_ylim(0, max(heights + spans) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    base_h, base_s = heights[0], spans[0]
    for x, h, s in zip(xs, heights, spans):
        ax.annotate(f"{100*(h/base_h-1):+.1f}%", (x - 0.2, h), ha="center",
                    va="bottom", fontsize=7)
        ax.annotate(f"{100*(s/base_s-1):+.1f}%", (x + 0.2, s), ha="center",
                    va="bottom", fontsize=7)
    ax.set_title("Exploratory geometry deltas vs C_R1 baseline\n"
                 "(no metric scale: no verified real-world distance)", fontsize=10)
    fig.suptitle("TSDF mesh + exploratory phenotype geometry (30k, res 2, "
                 "standard TSDF; % vs baseline annotated)", fontsize=10)
    fig.tight_layout()
    out = FIG / "fig6_mesh_phenotype_geometry.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma list of: chain,gcfc,sgmps,ablation,downstream,mesh")
    cli = ap.parse_args()
    wanted = [w for w in cli.only.split(",") if w]
    FIG.mkdir(parents=True, exist_ok=True)

    todo = {"chain": lambda: fig_chain(), "gcfc": lambda: fig_gcfc(),
            "sgmps": fig_sgmps, "ablation": fig_ablation,
            "downstream": fig_downstream_2x2, "mesh": fig_mesh_silhouette}
    for name, fn in todo.items():
        if wanted and name not in wanted:
            continue
        try:
            out = fn()
        except Exception as exc:  # keep going: long jobs may still be running
            print(f"[warn] {name} failed: {exc!r}")
            continue
        if out:
            print(f"[fig] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
