#!/usr/bin/env python3
"""Summarize KongQueZhuYu E2/E3/E6/E7/E8 matrix outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/fj/F2DMAS")
OUTPUT_ROOT = ROOT / "00-论文优化重构" / "数据管理" / "06-实验输出" / "KongQueZhuYu"
SUMMARY_DIR = ROOT / "00-论文优化重构" / "数据管理" / "05-评测结果" / "KongQueZhuYu" / "S10_small_matrix"

METHODS = [
    ("E2_2dgs_baseline", "Standard 2DGS"),
    ("E3_fsam3_preprocess", "FSAM3-preprocessed 2DGS"),
    ("E6_mask_constrained", "Mask-constrained 2DGS"),
    ("E7_mask_pruning", "Mask-constrained + pruning"),
    ("E8_full_plant_aware", "Full Plant-aware 2DGS"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_result(results: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if not results:
        return None, {}
    keys = sorted(results)
    return keys[-1], results[keys[-1]]


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


def pruning_summary(method_dir: Path) -> dict[str, Any]:
    pruning_dir = method_dir / "pruning"
    reports = sorted(pruning_dir.glob("pruning_iter_*.json")) if pruning_dir.exists() else []
    removed_total = 0
    last_after = None
    iterations = []
    for report_path in reports:
        report = read_json(report_path)
        removed_total += int(report.get("removed", 0))
        last_after = report.get("gaussians_after", last_after)
        iterations.append(report.get("iteration"))
    return {
        "pruning_reports": len(reports),
        "pruning_removed_total": removed_total if reports else None,
        "pruning_last_gaussians_after": last_after,
        "pruning_iterations": ";".join(str(v) for v in iterations if v is not None),
    }


def summarize_method(method_tag: str, label: str) -> dict[str, Any]:
    method_dir = OUTPUT_ROOT / method_tag
    run_status = read_json(method_dir / "run_status.json")
    baseline_guard = read_json(method_dir / "baseline_guard.json")
    result_key, metrics = latest_result(read_json(method_dir / "results.json"))
    mesh_metrics = read_json(method_dir / "mesh_metrics.json")

    row: dict[str, Any] = {
        "sample": "KongQueZhuYu",
        "method_tag": method_tag,
        "method": label,
        "status": run_status.get("status", "missing"),
        "result_key": result_key,
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "gaussians_30000": count_gaussians(method_dir),
        "uses_mask": baseline_guard.get("uses_mask"),
        "uses_h_vqg": baseline_guard.get("uses_h_vqg"),
        "uses_m3_mask_loss": baseline_guard.get("uses_m3_mask_loss"),
        "uses_m4_pruning": baseline_guard.get("uses_m4_pruning"),
        "uses_m5_edge_meshing": baseline_guard.get("uses_m5_edge_meshing"),
        "output_dir": str(method_dir),
    }
    row.update(pruning_summary(method_dir))
    if mesh_metrics:
        raw_mesh = mesh_metrics.get("raw_mesh", {})
        post_mesh = mesh_metrics.get("post_mesh", {})
        edge_metrics = mesh_metrics.get("edge_metrics") or {}
        row.update(
            {
                "mesh_mode": mesh_metrics.get("meshing_mode"),
                "raw_mesh_vertices": raw_mesh.get("vertices"),
                "raw_mesh_triangles": raw_mesh.get("triangles"),
                "post_mesh_vertices": post_mesh.get("vertices"),
                "post_mesh_triangles": post_mesh.get("triangles"),
                "mesh_used_masks": edge_metrics.get("used_masks"),
                "mesh_mean_shrink_scale": edge_metrics.get("mean_shrink_scale"),
            }
        )
    return row


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = ["method_tag", "status", "psnr", "ssim", "lpips", "gaussians_30000", "pruning_removed_total", "mesh_mode", "post_mesh_vertices"]
    lines = [
        "# KongQueZhuYu 小矩阵结果汇总",
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
            "说明：E3 训练阶段使用 `mask_mode=preprocess`，渲染/评测阶段覆盖为 `mask_mode=alpha`，以保留全图 GT 指标口径。",
            "E8 训练阶段使用 H-VQG retained list，渲染阶段覆盖为全测试集，以和 E2/E3/E6/E7 保持可比。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_method(tag, label) for tag, label in METHODS]

    json_path = SUMMARY_DIR / "kongquezhu_small_matrix_summary.json"
    csv_path = SUMMARY_DIR / "kongquezhu_small_matrix_summary.csv"
    md_path = SUMMARY_DIR / "kongquezhu_small_matrix_summary.md"

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
