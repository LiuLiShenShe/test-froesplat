#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Fig. 9 panels (a-f) from the phenotypic measurement spreadsheet.

Input Excel columns expected:
- 品种
- 株高真值, 株高虚拟植
- 冠幅真值, 冠幅虚拟植
- 叶长真值1, 叶长虚拟植1, 叶宽真值1, 叶宽虚拟植1
- 叶长真值2, 叶长虚拟植2, 叶宽真值2, 叶宽虚拟植2
- 叶长真值3, 叶长虚拟植3, 叶宽真值3, 叶宽虚拟植3

Outputs:
- Fig9a_plant_height_scatter.png
- Fig9b_canopy_width_scatter.png
- Fig9c_leaf_length_scatter.png
- Fig9d_leaf_width_scatter.png
- Fig9e_residual_distribution.png
- Fig9f_summary_metrics.png
- Fig9_summary_metrics.csv
- Fig9_combined_draft.png  (optional draft assembly for checking)

Usage:
    python generate_fig9_panels.py --excel 植株数据.xlsx --outdir fig9_panels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score


# Nature-style, clean figure settings.
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

COLORS = {
    "Plant height": "#2B5DAA",
    "Canopy width": "#0B7A75",
    "Leaf length": "#E66101",
    "Leaf width": "#6A3D9A",
}

PANEL_LABEL_KW = dict(fontsize=13, fontweight="bold", ha="left", va="top")


def read_traits(excel_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read manual and virtual measurements from Excel."""
    df = pd.read_excel(excel_path)

    traits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "Plant height": (
            df["株高真值"].to_numpy(dtype=float),
            df["株高虚拟植"].to_numpy(dtype=float),
        ),
        "Canopy width": (
            df["冠幅真值"].to_numpy(dtype=float),
            df["冠幅虚拟植"].to_numpy(dtype=float),
        ),
        "Leaf length": (
            np.concatenate([df["叶长真值1"], df["叶长真值2"], df["叶长真值3"]]).astype(float),
            np.concatenate([df["叶长虚拟植1"], df["叶长虚拟植2"], df["叶长虚拟植3"]]).astype(float),
        ),
        "Leaf width": (
            np.concatenate([df["叶宽真值1"], df["叶宽真值2"], df["叶宽真值3"]]).astype(float),
            np.concatenate([df["叶宽虚拟植1"], df["叶宽虚拟植2"], df["叶宽虚拟植3"]]).astype(float),
        ),
    }
    return traits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_pred - y_true
    return {
        "N": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_cm": float(np.sqrt(np.mean(residual ** 2))),
        "MAE_cm": float(np.mean(np.abs(residual))),
        "MAPE_percent": float(np.mean(np.abs(residual / y_true)) * 100),
        "Bias_cm": float(np.mean(residual)),
    }


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.18, 1.10, label, transform=ax.transAxes, **PANEL_LABEL_KW)


def plot_scatter_panel(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_name: str,
    panel_label: str | None = None,
) -> None:
    """Scatter plot: manual vs virtual measurements with 1:1 line and linear fit."""
    color = COLORS[trait_name]
    metrics = compute_metrics(y_true, y_pred)
    slope, intercept = np.polyfit(y_true, y_pred, 1)

    # Axis range with a small margin.
    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))
    margin = (high - low) * 0.08 if high > low else 1.0
    xmin, xmax = low - margin, high + margin

    ax.scatter(
        y_true,
        y_pred,
        s=18,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        zorder=3,
    )
    ax.plot([xmin, xmax], [xmin, xmax], "--", color="0.55", lw=0.9, zorder=1, label="1:1")
    xfit = np.linspace(xmin, xmax, 100)
    ax.plot(xfit, slope * xfit + intercept, color=color, lw=1.2, zorder=2, label="Fit")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(trait_name, color=color, pad=4, fontweight="bold")
    ax.set_xlabel("Manual measurement (cm)")
    ax.set_ylabel("Virtual measurement (cm)")
    ax.grid(True, color="0.92", lw=0.6)
    ax.tick_params(direction="out", length=3, width=0.8)

    txt = (
        f"N = {metrics['N']}\n"
        f"R² = {metrics['R2']:.4f}\n"
        f"RMSE = {metrics['RMSE_cm']:.2f} cm\n"
        f"MAE = {metrics['MAE_cm']:.2f} cm"
    )
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.7),
    )
    if panel_label:
        add_panel_label(ax, panel_label)


def plot_residual_panel(ax, traits: Dict[str, Tuple[np.ndarray, np.ndarray]], panel_label: str | None = None) -> None:
    """Panel e: residual distributions for four traits."""
    names = list(traits.keys())
    residuals = [traits[name][1] - traits[name][0] for name in names]
    data_positions = np.arange(1, len(names) + 1)

    bp = ax.boxplot(
        residuals,
        positions=data_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", lw=1.0),
        boxprops=dict(lw=0.8),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="white", markeredgecolor="0.3", alpha=0.8),
    )
    for box, name in zip(bp["boxes"], names):
        box.set_facecolor(COLORS[name])
        box.set_alpha(0.72)

    # Add jittered points for transparency.
    rng = np.random.default_rng(2026)
    for i, (name, res) in enumerate(zip(names, residuals), start=1):
        jitter = rng.normal(0, 0.045, size=len(res))
        ax.scatter(
            np.full(len(res), i) + jitter,
            res,
            s=7,
            color=COLORS[name],
            alpha=0.45,
            linewidth=0,
            zorder=2,
        )

    ax.axhline(0, color="0.25", lw=0.9, ls="--")
    ax.set_xticks(data_positions)
    ax.set_xticklabels([f"{name}\n(N={len(traits[name][0])})" for name in names])
    ax.set_ylabel("Residual (virtual − manual, cm)")
    ax.set_title("Residual distribution", pad=5, fontweight="bold")
    ax.grid(True, axis="y", color="0.92", lw=0.6)
    ax.tick_params(direction="out", length=3, width=0.8)
    if panel_label:
        add_panel_label(ax, panel_label)


def plot_summary_panel(ax, metrics_df: pd.DataFrame, panel_label: str | None = None) -> None:
    """Panel f: compact metric table. Easier to assemble than multi-axis bars."""
    ax.axis("off")
    display_df = metrics_df.copy()
    display_df["MAE (cm)"] = display_df["MAE_cm"].map(lambda v: f"{v:.2f}")
    display_df["RMSE (cm)"] = display_df["RMSE_cm"].map(lambda v: f"{v:.2f}")
    display_df["MAPE (%)"] = display_df["MAPE_percent"].map(lambda v: f"{v:.2f}")
    display_df["Bias (cm)"] = display_df["Bias_cm"].map(lambda v: f"{v:.2f}")
    display_df["R²"] = display_df["R2"].map(lambda v: f"{v:.4f}")
    display_df["N"] = display_df["N"].astype(int).astype(str)

    cols = ["Trait", "N", "MAE (cm)", "RMSE (cm)", "MAPE (%)", "Bias (cm)", "R²"]
    cell_text = display_df[cols].values.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.10, 0.15, 0.15, 0.16, 0.14, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.45)

    # Header and trait-name coloring.
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("0.35")
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(fontweight="bold")
        elif col == 0:
            trait_name = display_df.iloc[row - 1]["Trait"]
            cell.set_text_props(color=COLORS[trait_name], fontweight="bold")

    ax.set_title("Summary metrics", pad=10, fontweight="bold")
    if panel_label:
        ax.text(-0.02, 1.08, panel_label, transform=ax.transAxes, **PANEL_LABEL_KW)


def save_individual_panels(traits: Dict[str, Tuple[np.ndarray, np.ndarray]], metrics_df: pd.DataFrame, outdir: Path) -> None:
    """Save Fig. 9a-f as separate images for manual assembly."""
    outdir.mkdir(parents=True, exist_ok=True)

    scatter_specs = [
        ("a", "Plant height", "Fig9a_plant_height_scatter.png"),
        ("b", "Canopy width", "Fig9b_canopy_width_scatter.png"),
        ("c", "Leaf length", "Fig9c_leaf_length_scatter.png"),
        ("d", "Leaf width", "Fig9d_leaf_width_scatter.png"),
    ]
    for label, trait_name, filename in scatter_specs:
        fig, ax = plt.subplots(figsize=(2.7, 2.7))
        plot_scatter_panel(ax, traits[trait_name][0], traits[trait_name][1], trait_name, label)
        fig.tight_layout(pad=0.6)
        fig.savefig(outdir / filename, bbox_inches="tight")
        fig.savefig(outdir / filename.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    plot_residual_panel(ax, traits, "e")
    fig.tight_layout(pad=0.6)
    fig.savefig(outdir / "Fig9e_residual_distribution.png", bbox_inches="tight")
    fig.savefig(outdir / "Fig9e_residual_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    plot_summary_panel(ax, metrics_df, "f")
    fig.tight_layout(pad=0.6)
    fig.savefig(outdir / "Fig9f_summary_metrics.png", bbox_inches="tight")
    fig.savefig(outdir / "Fig9f_summary_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def save_combined_draft(traits: Dict[str, Tuple[np.ndarray, np.ndarray]], metrics_df: pd.DataFrame, outdir: Path) -> None:
    """Save a draft assembled version. You can still manually reassemble final panels."""
    fig = plt.figure(figsize=(12.0, 7.0))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.05], hspace=0.48, wspace=0.42)

    top_traits = ["Plant height", "Canopy width", "Leaf length", "Leaf width"]
    for i, trait_name in enumerate(top_traits):
        ax = fig.add_subplot(gs[0, i])
        plot_scatter_panel(ax, traits[trait_name][0], traits[trait_name][1], trait_name, chr(ord("a") + i))

    ax_e = fig.add_subplot(gs[1, 0:2])
    plot_residual_panel(ax_e, traits, "e")

    ax_f = fig.add_subplot(gs[1, 2:4])
    plot_summary_panel(ax_f, metrics_df, "f")

    fig.subplots_adjust(top=0.96)
    fig.savefig(outdir / "Fig9_combined_draft.png", bbox_inches="tight")
    fig.savefig(outdir / "Fig9_combined_draft.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig. 9 panels from phenotypic measurements.")
    parser.add_argument("--excel", type=Path, default=Path("植株数据.xlsx"), help="Path to the Excel file.")
    parser.add_argument("--outdir", type=Path, default=Path("fig9_panels"), help="Output directory.")
    args = parser.parse_args()

    traits = read_traits(args.excel)

    rows = []
    for trait_name, (manual, virtual) in traits.items():
        row = {"Trait": trait_name}
        row.update(compute_metrics(manual, virtual))
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[["Trait", "N", "MAE_cm", "RMSE_cm", "MAPE_percent", "Bias_cm", "R2"]]

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.outdir / "Fig9_summary_metrics.csv", index=False, encoding="utf-8-sig")

    save_individual_panels(traits, metrics_df, args.outdir)
    save_combined_draft(traits, metrics_df, args.outdir)

    print("Saved panels to:", args.outdir.resolve())
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
