#!/usr/bin/env python3
"""Summarize A6 results over the representative sample set."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/fj/F2DMAS")
DOC_ROOT = ROOT / "00-论文优化重构"
OUTPUT_ROOT = DOC_ROOT / "数据管理" / "06-实验输出"
SUMMARY_DIR = DOC_ROOT / "数据管理" / "05-评测结果" / "S12_representative_A6_extension"
S12_CONFIG_DIR = (
    DOC_ROOT
    / "数据管理"
    / "07-运行脚本与超参"
    / "S12-代表样本A6扩展"
    / "configs"
)
KONGQUEZHU_A6 = (
    DOC_ROOT
    / "数据管理"
    / "07-运行脚本与超参"
    / "S10-KongQueZhuYu小矩阵"
    / "configs"
    / "kongquezhu_A6_foreground_track_init_fg_rgb_alpha_bg.json"
)
CONFIGS = [
    KONGQUEZHU_A6,
    S12_CONFIG_DIR / "xiankelai1_A6_foreground_track_init_fg_rgb_alpha_bg.json",
    S12_CONFIG_DIR / "caomei2_A6_foreground_track_init_fg_rgb_alpha_bg.json",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_result(results: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    keys = [key for key, value in results.items() if key != "per_view" and isinstance(value, dict)]
    if not keys:
        return None, {}
    key = sorted(keys)[-1]
    return key, results[key]


def output_dir_for(config: dict[str, Any]) -> Path:
    direct = OUTPUT_ROOT / config["sample"] / config["method_tag"]
    if direct.exists():
        return direct
    candidates = sorted((OUTPUT_ROOT / config["sample"]).glob(f'{config["method_tag"]}_*'))
    return candidates[-1] if candidates else direct


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


def sample_role(sample: str) -> str:
    roles = {
        "KongQueZhuYu": "复杂背景/主样本",
        "XianKeLai1": "薄叶/细结构",
        "CaoMei2": "密集叶/遮挡",
    }
    return roles.get(sample, "")


def sanitize_report_for(config: dict[str, Any]) -> dict[str, Any]:
    source = Path(config["source_path"])
    return read_json(source / "sanitize_report.json")


def summarize_config(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    method_dir = output_dir_for(config)
    run_status = read_json(method_dir / "run_status.json")
    full_key, full_metrics = latest_result(read_json(method_dir / "results.json"))
    fg_key, fg_metrics = latest_result(read_json(method_dir / "foreground_object_results.json"))
    init_report = read_json(method_dir / "foreground_init_pcd_report.json")
    sanitize = sanitize_report_for(config)

    return {
        "sample": config.get("sample"),
        "role": sample_role(config.get("sample", "")),
        "method_tag": config.get("method_tag"),
        "status": run_status.get("status", "missing"),
        "output_dir": str(method_dir),
        "source_path": config.get("source_path"),
        "full_result_key": full_key,
        "full_psnr": full_metrics.get("PSNR"),
        "full_ssim": full_metrics.get("SSIM"),
        "full_lpips": full_metrics.get("LPIPS"),
        "fg_result_key": fg_key,
        "fg_psnr": fg_metrics.get("PSNR_fg"),
        "fg_ssim": fg_metrics.get("SSIM_fg"),
        "fg_lpips_black_bg": fg_metrics.get("LPIPS_fg_black_bg"),
        "fg_lpips_crop": fg_metrics.get("LPIPS_fg_crop"),
        "outside_nonblack_ratio_mean": fg_metrics.get("outside_nonblack_ratio_mean"),
        "leakage_energy_ratio_mean": fg_metrics.get("leakage_energy_ratio_mean"),
        "mask_ratio_mean": fg_metrics.get("mask_ratio_mean"),
        "gaussians_30000": count_gaussians(method_dir),
        "init_points_before": init_report.get("points_before"),
        "init_points_after": init_report.get("points_after"),
        "init_kept_ratio": init_report.get("kept_ratio"),
        "sanitized_images_before": sanitize.get("images_before"),
        "sanitized_images_after": sanitize.get("images_after"),
        "sanitized_dropped_count": sanitize.get("dropped_count"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "sample",
        "role",
        "status",
        "fg_psnr",
        "fg_ssim",
        "fg_lpips_black_bg",
        "outside_nonblack_ratio_mean",
        "leakage_energy_ratio_mean",
        "gaussians_30000",
        "sanitized_images_after",
        "sanitized_dropped_count",
    ]
    lines = [
        "# S12 representative A6 extension summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    lines.extend(
        [
            "",
            "Interpretation focus: A6 should preserve foreground-object quality across the thin-structure and dense-occlusion samples, while keeping leakage metrics low enough for plant-only export / mesh follow-up.",
            "KongQueZhuYu uses the original final_locked scene; XianKeLai1 and CaoMei2 use sanitized COLMAP views only because a few locked image/mask files are unreadable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_config(path) for path in CONFIGS]
    json_path = SUMMARY_DIR / "representative_A6_summary.json"
    csv_path = SUMMARY_DIR / "representative_A6_summary.csv"
    md_path = SUMMARY_DIR / "representative_A6_summary.md"

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
