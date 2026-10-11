#!/usr/bin/env python3
"""Prepare immutable Phase 20C inputs, aliases, manifests, and configs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证"
B20 = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十B_历史实验与证据修复"
P12 = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收"
P16 = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十六_HardCase_GT构建与锁定"
P17A = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计"
P17B = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七B_上游失败归因与候选Oracle审计"
P6 = P12 / "P6_raw_baseline"
P00 = P17A / "02_trackA_variants/P00_Control"
COLMAP_ROOT = ROOT / "00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked"
PRODUCTION = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py"
SAM3_REPO = ROOT / "第三方源码/SAM3-latest"
SAM3_CHECKPOINT = ROOT / "sam3/sam3.pt"

EASY_SAMPLES = {
    "CaoMei1": ["0000", "0025", "0050", "0075", "0100"],
    "ChangShouHua2": ["0000", "0025", "0050", "0075", "0100"],
    "DouBanLv1": ["0000", "0025", "0050", "0075", "0100"],
    "KongQueZhuYu": ["0000", "0025", "0050", "0075", "0100"],
    "XianKeLai1": ["0000"],
}
GT10_SAMPLES = {
    "BaiZhang": ["0033", "0034", "0035", "0036", "0037"],
    "DouBanLv2": ["0037", "0038", "0039", "0040", "0041"],
}
GROUPS = {
    "V0": {"ranking_policy": "base", "refinement": False},
    "V1": {"ranking_policy": "r1", "refinement": False},
    "V2": {"ranking_policy": "base", "refinement": True},
    "V3": {"ranking_policy": "r1", "refinement": True},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved(path: Path) -> Path:
    return path.resolve(strict=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def link(source: Path, destination: Path) -> None:
    source = resolved(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source:
            return
        raise RuntimeError(f"Refusing to overwrite existing path: {destination}")
    destination.symlink_to(source)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def candidate_files(source_root: Path, sample: str, frame: str) -> list[Path]:
    return sorted((source_root / "候选掩膜/raw_instance_P6").glob(f"mask_{sample}_{frame}_*.png"))


def score_file(source_root: Path, sample: str, frame: str) -> Path | None:
    path = source_root / "候选掩膜" / f"候选评分明细_{sample}_{frame}.csv"
    return path if path.exists() else None


def source_record(dataset: str, sample: str, frame: str) -> tuple[Path, Path, Path, list[Path], Path | None]:
    if dataset == "Easy21":
        input_root = B20 / "phase20c_inputs" / sample
        input_path = input_root / f"{frame}.jpg"
        gt_path = P12 / "GT_potted_clean" / sample / f"mask_potted_clean_{frame}.png"
        source_root = P6
        source_candidates = candidate_files(source_root, sample, frame)
        score = score_file(source_root, sample, frame)
    else:
        input_path = P17A / "00_input_trackA" / f"{sample}_{frame}.jpg"
        gt_path = P16 / "GT_potted_clean_challenge" / sample / f"mask_potted_clean_{frame}.png"
        source_root = P00
        source_candidates = candidate_files(source_root, sample, frame)
        score = score_file(source_root, sample, frame)
    return resolved(input_path), resolved(gt_path), source_root, source_candidates, score


def ensure_empty_or_create(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise RuntimeError(f"Output/input preparation directory is not empty: {path}")
    else:
        path.mkdir(parents=True)


def prepare_dataset(dataset: str, samples: dict[str, list[str]]) -> list[dict[str, str]]:
    input_dir = WORK / "configs/inputs" / dataset
    candidate_dir = WORK / "configs/candidates" / dataset
    metadata_dir = WORK / "configs/candidate_metadata" / dataset
    for path in (input_dir, candidate_dir / "候选掩膜/raw_instance_P6", metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for sample, frames in samples.items():
        for frame in frames:
            input_path, gt_path, source_root, candidates, score = source_record(dataset, sample, frame)
            stem = f"{sample}_{frame}"
            input_alias = input_dir / f"{stem}.jpg"
            link(input_path, input_alias)
            alias_candidates: list[Path] = []
            for candidate in candidates:
                suffix = candidate.stem.rsplit("_", 1)[-1]
                alias = candidate_dir / "候选掩膜/raw_instance_P6" / f"mask_{stem}_{suffix}.png"
                link(candidate, alias)
                alias_candidates.append(alias)
            score_alias = ""
            if score is not None:
                score_alias_path = metadata_dir / f"候选评分明细_{stem}.csv"
                link(score, score_alias_path)
                score_alias = rel(score_alias_path)
            input_size = image_size(input_path)
            gt_size = image_size(gt_path)
            candidate_sizes = [image_size(candidate) for candidate in candidates]
            size_ok = input_size == gt_size and all(size == input_size for size in candidate_sizes)
            candidate_hashes = ";".join(sha256(candidate) for candidate in candidates)
            records.append(
                {
                    "dataset": dataset,
                    "sample_id": sample,
                    "frame": frame,
                    "stem": stem,
                    "input_path": rel(input_path),
                    "input_alias": rel(input_alias),
                    "gt_type": "potted_clean",
                    "gt_path": rel(gt_path),
                    "candidate_source_root": rel(source_root),
                    "candidate_alias_dir": rel(candidate_dir / "候选掩膜/raw_instance_P6"),
                    "candidate_count": str(len(candidates)),
                    "candidate_hashes": candidate_hashes,
                    "score_record_path": score_alias,
                    "input_sha256": sha256(input_path),
                    "gt_sha256": sha256(gt_path),
                    "input_size": f"{input_size[0]}x{input_size[1]}",
                    "gt_size": f"{gt_size[0]}x{gt_size[1]}",
                    "coordinate_system_ok": str(size_ok),
                    "colmap_dir": rel(COLMAP_ROOT / sample) if (COLMAP_ROOT / sample).exists() else "",
                    "status": "READY" if size_ok and candidates and score is not None else "BLOCKED",
                    "block_reason": "" if size_ok and candidates and score is not None else "missing size/candidate/score asset",
                }
            )
    return records


def command_for(dataset: str, group: str) -> list[str]:
    config_path = WORK / "configs" / f"{dataset}_{group}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["command"]


def write_configs(dataset: str, records: list[dict[str, str]]) -> None:
    input_dir = WORK / "configs/inputs" / dataset
    candidate_dir = WORK / "configs/candidates" / dataset
    output_root = WORK / "runs"
    for group, policy in GROUPS.items():
        refinement = policy["refinement"]
        config: dict[str, object] = {
            "phase": "20C",
            "dataset": dataset,
            "group": group,
            "ranking_policy": policy["ranking_policy"],
            "refinement": "ON" if refinement else "OFF",
            "prompt_list": "P6",
            "default_prompt_id": "P6",
            "candidate_mode": "per_instance",
            "prompt_selection_mode": "single",
            "prompt_ensemble": False,
            "score_weights": "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
            "mask_threshold": 0.5,
            "candidate_cache": "P6 raw instance candidate masks",
            "input_dir": str(input_dir.resolve()),
            "candidate_source_dir": str(candidate_dir.resolve()),
            "output_dir": str((output_root / group / dataset).resolve()),
            "sam3_repo": str(SAM3_REPO.resolve()),
            "sam3_checkpoint": str(SAM3_CHECKPOINT.resolve()),
            "device": "cuda:0",
            "use_spnp_refinement": refinement,
            "spnp_backend": "sam3_if_supported" if refinement else "none",
            "use_residual_repair": refinement,
            "thin_repair_mode": "adjacent" if refinement else "none",
            "use_corrective_geometry": refinement,
            "corrective_geometry_backend": "colmap_tracks" if refinement else "none",
            "colmap_dir": str(COLMAP_ROOT.resolve()) if refinement else None,
            "a6_cross_view_consensus": False,
            "a7_memory_propagation": False,
            "candidate_score_metadata_replayed": False,
            "candidate_score_metadata_source": str((WORK / "configs/candidate_metadata" / dataset).resolve()),
            "input_records": len(records),
        }
        command = [
            "python",
            str(PRODUCTION.resolve()),
            "--input_dir",
            config["input_dir"],
            "--output_dir",
            config["output_dir"],
            "--sam3_repo",
            config["sam3_repo"],
            "--sam3_checkpoint",
            config["sam3_checkpoint"],
            "--device",
            config["device"],
            "--candidate_mode",
            config["candidate_mode"],
            "--prompt_list",
            config["prompt_list"],
            "--default_prompt_id",
            config["default_prompt_id"],
            "--prompt_selection_mode",
            config["prompt_selection_mode"],
            "--score_weights",
            config["score_weights"],
            "--ranking_policy",
            config["ranking_policy"],
            "--reuse_existing_candidates",
            "--candidate_source_dir",
            config["candidate_source_dir"],
            "--force",
        ]
        if refinement:
            command.extend(
                [
                    "--use_spnp_refinement",
                    "--spnp_backend",
                    "sam3_if_supported",
                    "--use_residual_repair",
                    "--thin_repair_mode",
                    "adjacent",
                    "--use_corrective_geometry",
                    "--corrective_geometry_backend",
                    "colmap_tracks",
                    "--colmap_dir",
                    config["colmap_dir"],
                ]
            )
        config["command"] = command
        path = WORK / "configs" / f"{dataset}_{group}.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    for path in (WORK / "configs", WORK / "runs", WORK / "metrics", WORK / "figures", WORK / "reports", WORK / "tests"):
        path.mkdir(parents=True, exist_ok=True)
    easy_records = prepare_dataset("Easy21", EASY_SAMPLES)
    gt10_records = prepare_dataset("GT10", GT10_SAMPLES)
    all_records = easy_records + gt10_records
    write_csv(WORK / "configs/PHASE20C_INPUT_MANIFEST.csv", all_records)
    matrix: list[dict[str, str]] = []
    for dataset, records in (("Easy21", easy_records), ("GT10", gt10_records)):
        for group in GROUPS:
            write_configs(dataset, records)
            policy = GROUPS[group]
            matrix.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "ranking_policy": policy["ranking_policy"],
                    "spnp": "ON" if policy["refinement"] else "OFF",
                    "residual_repair": "ON" if policy["refinement"] else "OFF",
                    "gcfc": "ON" if policy["refinement"] else "OFF",
                    "prompt": "P6",
                    "candidate_mode": "per_instance",
                    "prompt_selection": "single",
                    "prompt_ensemble": "OFF",
                    "a6": "OFF",
                    "a7": "OFF",
                    "candidate_source": f"configs/candidates/{dataset}",
                    "input_count": str(len(records)),
                    "status": "READY" if all(row["status"] == "READY" for row in records) else "BLOCKED",
                }
            )
    write_csv(WORK / "configs/PHASE20C_CONFIG_MATRIX.csv", matrix)
    (WORK / "configs/PHASE20C_COMMANDS.txt").write_text(
        "\n".join(
            " ".join(command_for(dataset, group))
            for dataset in ("Easy21", "GT10")
            for group in GROUPS
        )
        + "\n",
        encoding="utf-8",
    )
    source_summary = {
        "root": str(ROOT),
        "production_script": rel(PRODUCTION),
        "sam3_repo": rel(SAM3_REPO),
        "sam3_checkpoint": rel(SAM3_CHECKPOINT),
        "phase17b_oracle": rel(P17B / "Phase17B_candidate_oracle.csv"),
        "phase17c_replay": rel(ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七C_候选排序修复开发/Phase17C_production_replay.csv"),
        "candidate_score_metadata_note": "Raw candidates and score CSVs are preserved; current production reuse loader reads raw masks only and sets sam_score=0.0.",
        "easy21_frames": len(easy_records),
        "gt10_frames": len(gt10_records),
    }
    (WORK / "configs/SOURCE_SUMMARY.json").write_text(json.dumps(source_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"easy21": len(easy_records), "gt10": len(gt10_records), "work": str(WORK)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
