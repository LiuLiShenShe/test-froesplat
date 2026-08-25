#!/usr/bin/env python3
"""Summarize Experiment 4 B0-B5 prior-injection ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/fj/F2DMAS")
OUTPUT_ROOT = ROOT / "00-论文优化重构" / "数据管理" / "06-实验输出" / "KongQueZhuYu"
WORKSPACE = ROOT / "00-论文优化重构" / "计算机与电子农业特刊实验工作区"
TABLE_PATH = WORKSPACE / "04-结果表格模板" / "实验四_先验注入位置消融结果表.csv"
FIG_ROOT = WORKSPACE / "05-图件与论文映射" / "实验四_2DGS先验注入位置消融"
SOURCE_DIR = FIG_ROOT / "source_data"
FIG_DIR = FIG_ROOT / "figures"


B_ROWS = [
    {
        "setting_id": "B0",
        "setting_name": "标准2DGS",
        "method_tag": "E2_2dgs_baseline",
        "input_mask": "否",
        "fg_init": "否",
        "rgb_loss": "否",
        "opacity_loss": "否",
        "pruning": "否",
        "condition": "标准全场景2DGS基线",
        "note": "映射自既有 E2_2dgs_baseline。",
    },
    {
        "setting_id": "B1",
        "setting_name": "仅输入遮罩",
        "method_tag": "E3_fsam3_preprocess",
        "input_mask": "是",
        "fg_init": "否",
        "rgb_loss": "否",
        "opacity_loss": "否",
        "pruning": "否",
        "condition": "mask_mode=preprocess",
        "note": "训练图像仅做前景预处理；前景分离强但全图重建指标明显下降。",
    },
    {
        "setting_id": "B2",
        "setting_name": "仅稀疏点初始化",
        "method_tag": "B2_foreground_track_init_only",
        "input_mask": "否",
        "fg_init": "是",
        "rgb_loss": "否",
        "opacity_loss": "否",
        "pruning": "否",
        "condition": "foreground_track init only",
        "note": "本轮补跑严格缺项；mask 只用于筛选 COLMAP 初始点，不参与 RGB/alpha/opacity loss。",
    },
    {
        "setting_id": "B3",
        "setting_name": "仅损失约束",
        "method_tag": "A5_fg_rgb_alpha_bg_loss",
        "input_mask": "否",
        "fg_init": "否",
        "rgb_loss": "是",
        "opacity_loss": "是",
        "pruning": "否",
        "condition": "foreground RGB + alpha mask + background opacity losses",
        "note": "映射自 S10 A5；无前景初始化和剪枝。",
    },
    {
        "setting_id": "B4",
        "setting_name": "仅事后剪枝",
        "method_tag": "B4_mask_pruning_only",
        "input_mask": "否",
        "fg_init": "否",
        "rgb_loss": "否",
        "opacity_loss": "否",
        "pruning": "是",
        "condition": "mask pruning only",
        "note": "本轮补跑严格缺项；mask 只用于 pruning score，不参与训练损失。",
    },
    {
        "setting_id": "B5",
        "setting_name": "完整ForeSplat",
        "method_tag": "F1_high_precision_foreground",
        "input_mask": "可选",
        "fg_init": "是",
        "rgb_loss": "是",
        "opacity_loss": "是",
        "pruning": "是",
        "condition": "A6 foreground-track init + losses + mask pruning",
        "note": "映射自既有 F1_high_precision_foreground，等价于 A6+M4 高精度前景对象设置。",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_result(results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    keys = sorted(k for k, v in results.items() if isinstance(v, dict) and k != "per_view")
    if not keys:
        return "", {}
    key = keys[-1]
    return key, results[key]


def count_gaussians(method_dir: Path, iteration: int = 30000) -> int | None:
    ply = method_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        return None
    with ply.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break
    return None


def step_time(run_status: dict[str, Any], step_name: str) -> float | str:
    for step in run_status.get("steps", []):
        if step.get("name") == step_name:
            return step.get("elapsed_seconds", "")
    return ""


def pruning_summary(method_dir: Path) -> tuple[int | str, int | str]:
    reports = sorted((method_dir / "pruning").glob("pruning_iter_*.json"))
    if not reports:
        return "", ""
    removed = 0
    last_after: int | str = ""
    for report_path in reports:
        report = read_json(report_path)
        removed += int(report.get("removed", 0))
        last_after = report.get("gaussians_after", last_after)
    return removed, last_after


def foreground_success(fg_metrics: dict[str, Any], full_metrics: dict[str, Any]) -> str:
    fg_psnr = fg_metrics.get("PSNR_fg")
    outside = fg_metrics.get("outside_nonblack_ratio_mean")
    leakage = fg_metrics.get("leakage_energy_ratio_mean")
    full_psnr = full_metrics.get("PSNR")
    if fg_psnr is None or outside is None or leakage is None:
        return "未评估"
    if outside < 0.05 and leakage < 0.10 and fg_psnr >= 23:
        return "是"
    if outside < 0.05 and leakage < 0.10:
        return "部分"
    if full_psnr is not None and full_psnr >= 23:
        return "全图可用但前景未分离"
    return "否"


def as_table_value(value: Any, digits: int = 6) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def summarize_row(spec: dict[str, str]) -> dict[str, Any]:
    method_dir = OUTPUT_ROOT / spec["method_tag"]
    run_status = read_json(method_dir / "run_status.json")
    _, full_metrics = latest_result(read_json(method_dir / "results.json"))
    _, fg_metrics = latest_result(read_json(method_dir / "foreground_object_results.json"))
    init_report = read_json(method_dir / "foreground_init_pcd_report.json")
    pruning_removed, pruning_last_after = pruning_summary(method_dir)

    row = {
        "设置编号": spec["setting_id"],
        "设置名称": spec["setting_name"],
        "输入遮罩": spec["input_mask"],
        "稀疏点初始化": spec["fg_init"],
        "RGB损失": spec["rgb_loss"],
        "透明度损失": spec["opacity_loss"],
        "剪枝": spec["pruning"],
        "样本名": "KongQueZhuYu",
        "代表条件": spec["condition"],
        "PSNR_fg": as_table_value(fg_metrics.get("PSNR_fg")),
        "SSIM_fg": as_table_value(fg_metrics.get("SSIM_fg")),
        "LPIPS_fg": as_table_value(fg_metrics.get("LPIPS_fg_black_bg")),
        "外部非黑比例": as_table_value(fg_metrics.get("outside_nonblack_ratio_mean")),
        "泄漏能量": as_table_value(fg_metrics.get("leakage_energy_ratio_mean")),
        "高斯数量": as_table_value(count_gaussians(method_dir), digits=0),
        "训练时间秒": as_table_value(step_time(run_status, "train"), digits=3),
        "网格化时间秒": "未统一统计",
        "前景对象成功": foreground_success(fg_metrics, full_metrics),
        "备注": spec["note"],
        "source_method_tag": spec["method_tag"],
        "source_output_dir": str(method_dir),
        "full_psnr": as_table_value(full_metrics.get("PSNR")),
        "full_ssim": as_table_value(full_metrics.get("SSIM")),
        "full_lpips": as_table_value(full_metrics.get("LPIPS")),
        "init_points_before": as_table_value(init_report.get("points_before"), digits=0),
        "init_points_after": as_table_value(init_report.get("points_after"), digits=0),
        "init_kept_ratio": as_table_value(init_report.get("kept_ratio")),
        "pruning_removed_total": as_table_value(pruning_removed, digits=0),
        "pruning_last_gaussians_after": as_table_value(pruning_last_after, digits=0),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, Any]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [row["设置编号"] for row in rows]
    names = [row["设置名称"] for row in rows]

    def f(row: dict[str, Any], key: str) -> float:
        value = row.get(key, "")
        return float(value) if value not in ("", None, "未统一统计") else np.nan

    psnr = [f(row, "PSNR_fg") for row in rows]
    leakage = [f(row, "泄漏能量") for row in rows]
    gaussians = [f(row, "高斯数量") / 1000.0 for row in rows]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
        }
    )
    colors = ["#4E79A7", "#59A14F", "#F28E2B"]
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 56 / 25.4), constrained_layout=True)
    for ax, values, title, ylabel, color in zip(
        axes,
        [psnr, leakage, gaussians],
        ["Foreground quality", "Background leakage", "Gaussian compactness"],
        ["PSNR_fg (dB)", "Leakage energy", "Gaussians (k)"],
        colors,
    ):
        ax.bar(labels, values, color=color, width=0.68)
        ax.set_title(title, fontsize=8.2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                text = f"{value:.2f}" if ylabel != "Gaussians (k)" else f"{value:.0f}"
                ax.text(idx, value, text, ha="center", va="bottom", fontsize=6.0)
    fig.suptitle("Experiment 4: where to inject the visual-foundation prior into 2DGS", fontsize=9)
    fig.text(
        0.5,
        -0.03,
        "B0: standard 2DGS; B1: input masking; B2: foreground-track init; B3: foreground losses; B4: mask pruning; B5: combined ForeSplat.",
        ha="center",
        fontsize=6.4,
    )
    stem = FIG_DIR / "Fig_E4_prior_injection_B0_B5_metrics"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tif"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    note_lines = [
        "# 实验四 2DGS 先验注入位置消融图板说明",
        "",
        "## 存放内容",
        "",
        "- `figures/Fig_E4_prior_injection_B0_B5_metrics.*`：B0-B5 指标图，展示前景质量、泄漏和高斯数量。",
        "- `source_data/experiment4_b0_b5_summary.csv`：B0-B5 完整源数据和映射路径。",
        "- `source_data/experiment4_b0_b5_paper_table.csv`：与工作区正式结果表同口径的数据。",
        "",
        "## 口径",
        "",
        "B2 和 B4 为 2026-06-05/06 本轮补跑的严格缺项；其余设置映射自既有 S10/S18 可复用输出。B3 使用 foreground RGB、alpha mask 与 background opacity losses 的组合，代表“损失约束”注入位置。",
        "",
        "## 分图标签",
        "",
    ]
    note_lines.extend(f"- `{label}`：{name}" for label, name in zip(labels, names))
    (FIG_ROOT / "图板说明.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = [summarize_row(spec) for spec in B_ROWS]
    table_fields = [
        "设置编号",
        "设置名称",
        "输入遮罩",
        "稀疏点初始化",
        "RGB损失",
        "透明度损失",
        "剪枝",
        "样本名",
        "代表条件",
        "PSNR_fg",
        "SSIM_fg",
        "LPIPS_fg",
        "外部非黑比例",
        "泄漏能量",
        "高斯数量",
        "训练时间秒",
        "网格化时间秒",
        "前景对象成功",
        "备注",
    ]
    source_fields = table_fields + [
        "source_method_tag",
        "source_output_dir",
        "full_psnr",
        "full_ssim",
        "full_lpips",
        "init_points_before",
        "init_points_after",
        "init_kept_ratio",
        "pruning_removed_total",
        "pruning_last_gaussians_after",
    ]
    write_csv(TABLE_PATH, rows, table_fields)
    write_csv(SOURCE_DIR / "experiment4_b0_b5_paper_table.csv", rows, table_fields)
    write_csv(SOURCE_DIR / "experiment4_b0_b5_summary.csv", rows, source_fields)
    make_figure(rows)

    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {SOURCE_DIR / 'experiment4_b0_b5_summary.csv'}")
    print(f"Wrote {FIG_DIR / 'Fig_E4_prior_injection_B0_B5_metrics.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
