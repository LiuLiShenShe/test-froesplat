from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区"
WORK = Path(__file__).resolve().parents[1]
FIX = BASE / "阶段二十C_FIX_候选元数据与精简细化验证"
ALG = BASE / "01-算法模块修改与验证"
PHASE16 = ALG / "阶段十六_HardCase_GT构建与锁定"
PHASE17A = ALG / "阶段十七A_10GT困难样本Pilot与无GT测试审计"
PHASE17C = ALG / "阶段十七C_候选排序修复开发"
SOURCE = PHASE17A / "03_trackB_variants/U00_Control"
PROD = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
BASELINE = "898cec2a"
DIFFICULTIES = {"D1": "Neighbor-plant adhesion", "D3": "Foreground-background confusion",
                "D6": "Small target", "D11": "Cluttered background", "D12": "Low contrast"}
METRICS = ["f1", "iou_fg", "miou", "hd95_px", "boundary_f1", "leakage_energy",
           "oracle_regret", "empty_mask", "severe_f1_lt_0_5_proxy", "severe_human_error",
           "target_identity_correct"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(field for row in rows for field in row))
    if not fields:
        raise ValueError("Refusing an undocumented empty CSV")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_text(path, text):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path.absolute())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_link(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace original-image link: {destination}")
    destination.symlink_to(source)


def unique_index(rows, field):
    indexed = {row[field]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {field}; full coverage cannot be assumed")
    return indexed


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{BASELINE}:{relative(path)}"], cwd=ROOT)
