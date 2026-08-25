#!/usr/bin/env python3
"""Summarize KongQueZhuYu A0-A6 foreground-object objective ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/fj/F2DMAS")
OUTPUT_ROOT = ROOT / "00-论文优化重构" / "数据管理" / "06-实验输出" / "KongQueZhuYu"
SUMMARY_DIR = ROOT / "00-论文优化重构" / "数据管理" / "05-评测结果" / "KongQueZhuYu" / "S10_small_matrix"

METHODS = [
    {
        "ablation_id": "A0",
        "method_tag": "E2_2dgs_baseline",
        "method": "full-scene 2DGS + foreground eval",
        "fg_init": False,
        "fg_rgb_loss": False,
        "alpha_mask_loss": False,
        "bg_opacity_loss": False,
        "notes": "Existing E2 output reused as A0.",
    },
    {
        "ablation_id": "A1",
        "method_tag": "E3_fsam3_preprocess",
        "method": "mask preprocess foreground training",
        "fg_init": False,
        "fg_rgb_loss": "implicit",
        "alpha_mask_loss": False,
        "bg_opacity_loss": False,
        "notes": "Existing E3 output reused as A1.",
    },
    {
        "ablation_id": "A2",
        "method_tag": "A2_alpha_mask_loss_only",
        "method": "alpha mask loss only",
        "fg_init": False,
        "fg_rgb_loss": False,
        "alpha_mask_loss": True,
        "bg_opacity_loss": False,
        "notes": "",
    },
    {
        "ablation_id": "A3",
        "method_tag": "A3_bg_opacity_only",
        "method": "background opacity only",
        "fg_init": False,
        "fg_rgb_loss": False,
        "alpha_mask_loss": False,
        "bg_opacity_loss": True,
        "notes": "",
    },
    {
        "ablation_id": "A4",
        "method_tag": "A4_alpha_mask_bg_opacity",
        "method": "alpha mask loss + background opacity",
        "fg_init": False,
        "fg_rgb_loss": False,
        "alpha_mask_loss": True,
        "bg_opacity_loss": True,
        "notes": "",
    },
    {
        "ablation_id": "A5",
        "method_tag": "A5_fg_rgb_alpha_bg_loss",
        "method": "foreground RGB loss + alpha mask loss + background opacity",
        "fg_init": False,
        "fg_rgb_loss": True,
        "alpha_mask_loss": True,
        "bg_opacity_loss": True,
        "notes": "",
    },
    {
        "ablation_id": "A6",
        "method_tag": "A6_foreground_track_init_fg_rgb_alpha_bg",
        "method": "foreground track init + foreground RGB loss + alpha mask loss + background opacity",
        "fg_init": True,
        "fg_rgb_loss": True,
        "alpha_mask_loss": True,
        "bg_opacity_loss": True,
        "notes": "Core A6 without M4 pruning.",
    },
    {
        "ablation_id": "A6+M4",
        "method_tag": "F1_high_precision_foreground",
        "method": "A6 + lightweight mask pruning",
        "fg_init": True,
        "fg_rgb_loss": True,
        "alpha_mask_loss": True,
        "bg_opacity_loss": True,
        "notes": "Existing F1 output; not the clean A6 ablation row.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_result(results: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if not results:
        return None, {}
    keys = [key for key, value in results.items() if isinstance(value, dict) and key != "per_view"]
    if not keys:
        return None, {}
    key = sorted(keys)[-1]
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


def summarize(row_spec: dict[str, Any]) -> dict[str, Any]:
    method_dir = OUTPUT_ROOT / str(row_spec["method_tag"])
    run_status = read_json(method_dir / "run_status.json")
    full_key, full_metrics = latest_result(read_json(method_dir / "results.json"))
    fg_key, fg_metrics = latest_result(read_json(method_dir / "foreground_object_results.json"))
    init_report = read_json(method_dir / "foreground_init_pcd_report.json")

    row = dict(row_spec)
    row.update(
        {
            "sample": "KongQueZhuYu",
            "status": run_status.get("status", "missing"),
            "output_dir": str(method_dir),
            "full_result_key": full_key,
            "full_psnr": full_metrics.get("PSNR"),
            "full_ssim": full_metrics.get("SSIM"),
            "full_lpips": full_metrics.get("LPIPS"),
            "fg_result_key": fg_key,
            "fg_psnr": fg_metrics.get("PSNR_fg"),
            "fg_ssim": fg_metrics.get("SSIM_fg"),
            "fg_lpips_black_bg": fg_metrics.get("LPIPS_fg_black_bg"),
            "outside_nonblack_ratio_mean": fg_metrics.get("outside_nonblack_ratio_mean"),
            "leakage_energy_ratio_mean": fg_metrics.get("leakage_energy_ratio_mean"),
            "outside_energy_mean": fg_metrics.get("outside_energy_mean"),
            "inside_energy_mean": fg_metrics.get("inside_energy_mean"),
            "mask_ratio_mean": fg_metrics.get("mask_ratio_mean"),
            "gaussians_30000": count_gaussians(method_dir),
            "init_points_before": init_report.get("points_before"),
            "init_points_after": init_report.get("points_after"),
            "init_kept_ratio": init_report.get("kept_ratio"),
        }
    )
    return row


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "ablation_id",
        "method_tag",
        "fg_init",
        "fg_rgb_loss",
        "alpha_mask_loss",
        "bg_opacity_loss",
        "fg_psnr",
        "fg_ssim",
        "fg_lpips_black_bg",
        "outside_nonblack_ratio_mean",
        "leakage_energy_ratio_mean",
        "gaussians_30000",
        "status",
    ]
    lines = [
        "# KongQueZhuYu A0-A6 foreground-object objective 消融",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "说明：A0/A1 复用既有 E2/E3 正式输出；A6+M4 复用既有 `F1_high_precision_foreground`，用于说明当前已验证完整结果。",
            "核心论文消融应优先比较 A0-A6；M4 pruning、M1 foreground gate 和 M5 mesh 作为后续模块单独汇报。",
            "foreground-only 分离阈值暂定为 outside_nonblack < 0.05 且 leakage < 0.10。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize(spec) for spec in METHODS]
    json_path = SUMMARY_DIR / "kongquezhu_A0_A6_ablation_summary.json"
    csv_path = SUMMARY_DIR / "kongquezhu_A0_A6_ablation_summary.csv"
    md_path = SUMMARY_DIR / "kongquezhu_A0_A6_ablation_summary.md"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    headers = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
