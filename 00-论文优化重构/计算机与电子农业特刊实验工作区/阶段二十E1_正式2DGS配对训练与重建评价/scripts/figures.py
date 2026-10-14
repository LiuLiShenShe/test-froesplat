from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import E0, GROUPS, MATRIX, SAMPLES, STAGE, config_for, read_csv, sha256, write_csv


def image_plates(sample, metrics, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    names = sorted({row["loader_image_name"] for row in metrics if row["sample"] == sample})
    selected = [names[index] for index in (0, len(names) // 2, len(names) - 1)]
    lookup = {(row["group"], row["loader_image_name"]): row for row in metrics if row["sample"] == sample}
    figure_sources = []
    for mode in ("heldout", "fixed_crop"):
        figure, axes = plt.subplots(3, 4, figsize=(7.2, 8.5), layout="constrained")
        for row_index, name in enumerate(selected):
            reference = lookup[("A_raw_RGB_control_v1", name)]["reference_display_path"]
            for column, group in enumerate(("Original RGB",) + GROUPS):
                source = reference if column == 0 else lookup[(group, name)]["render_rgb_path"]
                with Image.open(source) as image:
                    array = np.array(image.convert("RGB"))
                height, width = array.shape[:2]
                crop_xyxy = [0, 0, width, height]
                if mode == "fixed_crop":
                    crop_xyxy = [width // 4, height // 4, 3 * width // 4, 3 * height // 4]
                    array = array[crop_xyxy[1]:crop_xyxy[3], crop_xyxy[0]:crop_xyxy[2]]
                axes[row_index, column].imshow(array, interpolation="nearest")
                axes[row_index, column].set_axis_off()
                label = "Original RGB" if column == 0 else group.split("_")[0]
                if sample == "HongZhang" and column in (2, 3):
                    label += " (shared B=C)"
                axes[row_index, column].set_title(f"{label}\n{name}", fontsize=7)
                figure_sources.append({"sample": sample, "figure": f"{sample}_{mode}.png", "view": name,
                                       "group": group, "source_path": source, "source_sha256": sha256(source),
                                       "crop_xyxy": json.dumps(crop_xyxy), "display_adjustment": "none",
                                       "selection": "predeclared indices 0,N//2,N-1; center crop 0.25..0.75"})
        figure.suptitle(f"{sample} | {'held-out views' if mode == 'heldout' else 'fixed central crop (descriptive ROI)'}", fontsize=9)
        figure.savefig(output_dir / f"{sample}_{mode}.png", dpi=300)
        plt.close(figure)
    figure, axes = plt.subplots(3, 3, figsize=(7.2, 8.5), layout="constrained")
    for row_index, name in enumerate(selected):
        for column, group in enumerate(GROUPS):
            source = lookup[(group, name)]["render_alpha_path"]
            with Image.open(source) as image:
                array = np.array(image)
            axes[row_index, column].imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[row_index, column].set_axis_off()
            label = group.split("_")[0]
            if sample == "HongZhang" and group in ("B_R0", "C_R1"):
                label += " (shared B=C)"
            axes[row_index, column].set_title(f"{label}\n{name}", fontsize=7)
            figure_sources.append({"sample": sample, "figure": f"{sample}_alpha.png", "view": name,
                                   "group": group, "source_path": source, "source_sha256": sha256(source),
                                   "crop_xyxy": json.dumps([0, 0, array.shape[1], array.shape[0]]),
                                   "display_adjustment": "fixed grayscale 0..255; no adaptive contrast",
                                   "selection": "predeclared indices 0,N//2,N-1"})
    figure.suptitle(f"{sample} | rendered alpha (white=opaque); not independent silhouette GT", fontsize=9)
    figure.savefig(output_dir / f"{sample}_alpha.png", dpi=300)
    plt.close(figure)
    return figure_sources


def training_panels(sample, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
    styles = {"A_raw_RGB_control_v1": ("#2563EB", "-"), "B_R0": ("#D97706", "--"), "C_R1": ("#7C3AED", ":")}
    for matrix_sample, group in MATRIX:
        if matrix_sample != sample:
            continue
        rows = read_csv(STAGE / f"metrics/{sample}_{group}_TRAINING_CURVE.csv")
        rows = rows[::100] + [rows[-1]]
        color, style = styles[group]
        iterations = [int(row["iteration"]) for row in rows]
        axes[0].plot(iterations, [float(row["rgb_loss"]) for row in rows], color=color, ls=style, lw=0.8, label=group.split("_")[0])
        axes[1].plot(iterations, [int(row["gaussians"]) for row in rows], color=color, ls=style, lw=0.8)
        axes[2].plot(iterations, [float(row["wall_seconds"]) / 60 for row in rows], color=color, ls=style, lw=0.8)
    for axis, ylabel in zip(axes, ("Native RGB loss", "Gaussian count", "Observed wall time (min)")):
        axis.set_xlabel("Iteration")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6)
    figure.suptitle(f"{sample} | losses use different input RGB; not common-reference accuracy", fontsize=8)
    figure.savefig(output_dir / f"{sample}_training.png", dpi=300)
    figure.savefig(output_dir / f"{sample}_training.svg")
    figure.savefig(output_dir / f"{sample}_training.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=SAMPLES)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                               "axes.titlesize": 8, "axes.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    output_dir = STAGE / "figures"
    output_dir.mkdir(exist_ok=True)
    metrics = read_csv(STAGE / "HELDOUT_PER_VIEW_METRICS.csv")
    rows = []
    for sample in (args.sample,) if args.sample else SAMPLES:
        rows.extend(image_plates(sample, metrics, output_dir))
        training_panels(sample, output_dir)
    write_csv(STAGE / "reports/FIGURE_SOURCE_INDEX.csv", rows)
    print("Figures generated from real held-out renders, deterministic selection", len(rows))


if __name__ == "__main__":
    main()
