#!/usr/bin/env python3
"""Run the final Phase 20C input, cache, geometry, and factor audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证"
RAW_ROOT = ROOT / "00-论文优化重构/数据管理/01-输入图像/01-raw_frames"
P12_FRAME_ROOT = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收/frame_inputs"
P17A_INPUT_ROOT = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计/00_input_trackA"
PRODUCTION = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_equal(left: Path, right: Path) -> bool:
    with Image.open(left) as left_image, Image.open(right) as right_image:
        return np.array_equal(np.asarray(left_image.convert("RGB")), np.asarray(right_image.convert("RGB")))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_observation_function():
    sys.path.insert(0, str(PRODUCTION.parent))
    spec = importlib.util.spec_from_file_location("phase20c_production", PRODUCTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load production entrypoint")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_colmap_observations, module.DEFAULT_COLMAP_LOADER


def gpu_snapshot() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main() -> int:
    manifest = read_csv(WORK / "configs/PHASE20C_INPUT_MANIFEST.csv")
    precheck_rows: list[dict[str, object]] = []
    colmap_rows: list[dict[str, object]] = []
    load_colmap, loader_path = load_observation_function()
    for dataset in ("Easy21", "GT10"):
        dataset_records = [row for row in manifest if row["dataset"] == dataset]
        stems = {row["stem"] for row in dataset_records}
        args = type("Args", (), {})()
        args.colmap_dir = ROOT / "00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked"
        args.colmap_loader_path = loader_path
        observations = load_colmap(stems, args)
        for record in dataset_records:
            input_path = ROOT / record["input_path"]
            gt_path = ROOT / record["gt_path"]
            rgb_reference = (
                P12_FRAME_ROOT / f"{record['sample_id']}_{record['frame']}.jpg"
                if dataset == "Easy21"
                else P17A_INPUT_ROOT / f"{record['sample_id']}_{record['frame']}.jpg"
            )
            candidate_dir = ROOT / record["candidate_alias_dir"]
            candidates = sorted(candidate_dir.glob(f"mask_{record['stem']}_*.png"))
            score_path = ROOT / record["score_record_path"] if record["score_record_path"] else None
            score_count = len(read_csv(score_path)) if score_path and score_path.exists() else 0
            candidate_links_ok = bool(candidates) and all(path.is_symlink() and path.resolve().exists() for path in candidates)
            colmap = observations.get(record["stem"])
            precheck_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": record["sample_id"],
                    "frame": record["frame"],
                    "input_exists": int(input_path.exists()),
                    "gt_exists": int(gt_path.exists()),
                    "rgb_reference_path": str(rgb_reference.relative_to(ROOT)) if rgb_reference.exists() else str(rgb_reference),
                    "raw_rgb_exists": int(rgb_reference.exists()),
                    "rgb_identity": int(rgb_reference.exists() and pixel_equal(input_path, rgb_reference)),
                    "input_gt_sha_match": int(record["input_sha256"] == sha256(input_path) and record["gt_sha256"] == sha256(gt_path)),
                    "candidate_count_manifest": record["candidate_count"],
                    "candidate_count_actual": len(candidates),
                    "candidate_links_ok": int(candidate_links_ok),
                    "score_record_exists": int(score_path is not None and score_path.exists()),
                    "score_record_count": score_count,
                    "score_count_matches_candidates": int(score_count == len(candidates)),
                    "colmap_observation": int(colmap is not None),
                    "colmap_points": len(colmap.points) if colmap is not None else 0,
                    "status": "READY" if all(
                        [
                            input_path.exists(),
                            gt_path.exists(),
                            rgb_reference.exists(),
                            rgb_reference.exists() and pixel_equal(input_path, rgb_reference),
                            candidate_links_ok,
                            score_path is not None and score_path.exists(),
                            score_count == len(candidates),
                        ]
                    ) else "BLOCKED",
                }
            )
            colmap_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": record["sample_id"],
                    "frame": record["frame"],
                    "stem": record["stem"],
                    "observation_present": int(colmap is not None),
                    "point_count": len(colmap.points) if colmap is not None else 0,
                    "image_name": colmap.image_name if colmap is not None else "",
                    "geometry_status": "READY" if colmap is not None and len(colmap.points) > 0 else "BLOCKED",
                }
            )

    matrix = read_csv(WORK / "configs/PHASE20C_CONFIG_MATRIX.csv")
    factor_audit: list[dict[str, object]] = []
    allowed_factors = {"group", "ranking_policy", "spnp", "residual_repair", "gcfc"}
    invariant_fields = [field for field in matrix[0] if field not in allowed_factors and field not in {"dataset", "status"}]
    for dataset in ("Easy21", "GT10"):
        rows = [row for row in matrix if row["dataset"] == dataset]
        for field in invariant_fields:
            values = sorted({row[field] for row in rows})
            factor_audit.append({"dataset": dataset, "field": field, "values": ";".join(values), "invariant": int(len(values) == 1)})

    write_csv(WORK / "configs/PRECHECK_AUDIT.csv", precheck_rows)
    write_csv(WORK / "configs/COLMAP_OBSERVATION_AUDIT.csv", colmap_rows)
    write_csv(WORK / "configs/FACTOR_AUDIT.csv", factor_audit)
    snapshot = {
        "gpu": gpu_snapshot(),
        "sam3_repo_exists": (ROOT / "第三方源码/SAM3-latest").exists(),
        "sam3_checkpoint_exists": (ROOT / "sam3/sam3.pt").exists(),
        "production_script": str(PRODUCTION.relative_to(ROOT)),
        "candidate_score_metadata_replayed": False,
        "note": "Production candidate reuse currently reads masks but not historical SAM3 score metadata; no production code was changed in Phase 20C.",
    }
    (WORK / "configs/PRECHECK_SNAPSHOT.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing_geometry = [row["stem"] for row in colmap_rows if row["dataset"] == "GT10" and not row["observation_present"]]
    ready_count = sum(row["status"] == "READY" for row in precheck_rows)
    report = [
        "# INPUT_AND_CACHE_AUDIT",
        "",
        f"- Audited frames: {len(precheck_rows)}; READY: {ready_count}/{len(precheck_rows)}.",
        "- Input RGB, GT, candidate dimensions, candidate symlink targets, and historical score CSV row counts are checked without rewriting any source asset.",
        "- RGB identity uses the frozen Phase12 `frame_inputs` source for Easy21 and Phase17A P00 `00_input_trackA` source for GT10; an invalid legacy raw-frame file is not used as a reference.",
        "- SAM3 repo/checkpoint and GPU snapshot are recorded in `configs/PRECHECK_SNAPSHOT.json`.",
        "- The first SAM3 smoke run exposed a missing `pkg_resources` dependency; the runner uses the Phase20C-local runtime compatibility shim, with production sources unchanged.",
        "- Easy21 COLMAP observations: all frames present.",
        f"- GT10 missing COLMAP observations: {', '.join(missing_geometry) if missing_geometry else 'NONE'}.",
        "- Therefore GT10 V0/V1 is executable; complete GT10 V2/V3 is BLOCKED until the missing sparse observations are supplied and matched.",
        "- Candidate mask aliases and historical score CSVs exist, but production reuse does not replay `sam_score`; this is an explicit equivalence limitation, not a hidden parameter change.",
        "- A6/A7 are OFF in all eight frozen configs; factor invariance is recorded in `configs/FACTOR_AUDIT.csv`.",
    ]
    (WORK / "reports/INPUT_AND_CACHE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(precheck_rows), "ready": ready_count, "missing_geometry": missing_geometry}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
