from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from direct_common import DIFFICULTIES, OLD, SWITCHES, WORK, frames, local_path, read_csv, read_mask, relative, sha256, write_csv, write_text


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    destination = WORK / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    results = {row["frame_id"]: row for row in read_csv(WORK / "R0_R1_TEST32_RESULTS.csv")}
    manifest = {(row["frame_id"], row["rasterizer"]): row for row in read_csv(WORK / "RASTERIZED_MASK_MANIFEST.csv")}
    index, legends = [], []
    display_size = (360, 640)
    for frame in frames():
        stem = frame["frame_id"]
        row = results[stem]
        gt = manifest[(stem, "opencv")]
        rgb_path = local_path(frame["rgb_path"])
        gt_path = local_path(gt["mask_path"])
        prediction_paths = [WORK / "runs" / f"full_{policy}" / "最终掩膜" / f"mask_{stem}.png" for policy in ("base", "r1")]
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB").resize(display_size, Image.Resampling.LANCZOS))
        full_masks = [read_mask(path) for path in [gt_path] + prediction_paths]
        masks = [np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(display_size, Image.Resampling.NEAREST))
                 for mask in full_masks]
        fig, axes = plt.subplots(1, 4, figsize=(12, 6.1), layout="constrained")
        titles = ["RGB", "Rasterized contour GT\nOpenCV, not Gate-certified",
                  f"R0: {row['r0_selected_id']}\nF1={float(row['r0_f1']):.4f}, IoU={float(row['r0_iou_fg']):.4f}\n"
                  f"Area={int(full_masks[1].sum()):,} px (non-empty)",
                  f"R1: {row['r1_selected_id']}\nF1={float(row['r1_f1']):.4f}, IoU={float(row['r1_iou_fg']):.4f}\n"
                  f"Area={int(full_masks[2].sum()):,} px (non-empty)"]
        for panel, axis, title in zip([rgb] + masks, axes, titles, strict=True):
            axis.imshow(panel, cmap="gray", vmin=0, vmax=255)
            axis.set_title(title, fontsize=8)
            axis.set_axis_off()
        warning = " | GEOMETRY_AMBIGUITY: original contour retained" if row["geometry_warning"] == "GEOMETRY_AMBIGUITY" else ""
        fig.suptitle(f"{stem} | {row['difficulty_labels']} | R1-R0 F1={float(row['delta_f1']):+.6f}{warning}", fontsize=10)
        path = destination / f"comparison_{stem}.png"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=160, facecolor="white")
        if stem in SWITCHES:
            fig.savefig(destination / f"comparison_{stem}.pdf", dpi=300, facecolor="white")
        plt.close(fig)
        legend = (f"**{stem}.** RGB | deterministically rasterized potted_plant contour (minus real blue_cube if present) | "
                  f"frozen R0 | frozen R1. Archived grouping={row['difficulty_labels']}; "
                  f"F1 delta={float(row['delta_f1']):+.8f}, {row['f1_effect']}. "
                  + ("GEOMETRY_AMBIGUITY: the source point sequence intersects/backtracks in at least one shape; "
                     f"main-target shape affected={row['main_gt_shape_ambiguous']}. This panel is not unambiguous certifiedGT. "
                     if row["geometry_warning"] == "GEOMETRY_AMBIGUITY" else "No detected intersection warning is not human certification. ")
                  + "Display-only360×640; metrics use unchanged2160×3840. All32 frames are shown, not just gains.")
        legends.append(legend)
        index.append({"frame_id": stem, "figure_path": relative(path), "figure_sha256": sha256(path),
                      "rgb_path": relative(rgb_path), "rgb_sha256": frame["rgb_sha256"], "gt_sha256": gt["mask_sha256"],
                      "r0_mask_sha256": sha256(prediction_paths[0]), "r1_mask_sha256": sha256(prediction_paths[1]),
                      "difficulty_labels": row["difficulty_labels"], "geometry_warning": row["geometry_warning"],
                      "display_width_per_panel": 360, "display_height_per_panel": 640,
                      "evaluated_original_width": 2160, "evaluated_original_height": 3840,
                      "r1_minus_r0_f1": row["delta_f1"], "legend": legend})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    ordered = sorted(results.values(), key=lambda row: row["frame_id"])
    positions = np.arange(len(ordered))
    differences = np.array([float(row["delta_f1"]) for row in ordered])
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in differences]
    axes[0].bar(positions, differences, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("R1 - R0 F1")
    axes[0].set_xticks(positions, [row["frame_id"] for row in ordered], rotation=90, fontsize=6)
    axes[0].set_title("All32 paired differences — zeros retained")
    sensitivity = {row["frame_id"]: row for row in read_csv(WORK / "RASTERIZATION_SENSITIVITY.csv")}
    disagreement = [1 - float(sensitivity[row["frame_id"]]["raster_mask_iou"]) for row in ordered]
    axes[1].scatter(positions, disagreement, c=["#CC79A7" if row["geometry_warning"] == "GEOMETRY_AMBIGUITY" else "#009E73" for row in ordered])
    axes[1].set_ylabel("1 - cross-raster maskIoU")
    axes[1].set_xticks(positions, [row["frame_id"] for row in ordered], rotation=90, fontsize=6)
    axes[1].set_title("OpenCV vs Pillow; magenta=geometry warning, green=other")
    fig.savefig(destination / "paired_and_raster_sensitivity.png", dpi=180)
    fig.savefig(destination / "paired_and_raster_sensitivity.svg")
    fig.savefig(destination / "paired_and_raster_sensitivity.pdf")
    plt.close(fig)
    write_csv(destination / "FIGURE_SOURCE_INDEX.csv", index)
    write_text(destination / "FIGURE_LEGENDS.md", "# Figure contract and legends\n\nEvidence: actual32RGB, actual masks and complete per-frame metrics. "
               "Conclusion: show measured changes and uncertainty, not certify target identity. No AI-generated scientific pixels. "
               "Backend: existing savedPython preference; matplotlib only.\n\n" + "\n\n".join(legends))
    gallery = ["<!doctype html><meta charset='utf-8'><title>Fixed contour TEST32 gallery</title>",
               "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{width:100%}section{margin:30px 0}</style>",
               "<h1>Fixed-contour rasterization evaluation: complete32-frame gallery</h1>",
               "<p>Actual source images and masks. Not original-Gate-certifiedGT. Difficulty labels are existing archived groupings.</p>"]
    for code in list(DIFFICULTIES) + ["none"]:
        group = [row for row in index if code in row["difficulty_labels"].split(";")]
        if not group:
            continue
        gallery.append(f"<h2>{code}: {DIFFICULTIES.get(code, 'No recorded difficulty')} ({len(group)} frames; overlaps allowed)</h2>")
        for row in group:
            gallery.append(f"<section><h3>{row['frame_id']}: deltaF1={float(row['r1_minus_r0_f1']):+.6f}; {row['geometry_warning']}</h3>"
                           f"<img src='{Path(row['figure_path']).name}' alt='{row['frame_id']} RGB GT R0 R1'></section>")
    write_text(destination / "DIFFICULTY_GALLERY.html", "\n".join(gallery))
    print("Saved32 real comparison plates, two switchPDFs, all-frame paired/sensitivity plots and overlapping difficulty gallery", flush=True)


if __name__ == "__main__":
    main()
