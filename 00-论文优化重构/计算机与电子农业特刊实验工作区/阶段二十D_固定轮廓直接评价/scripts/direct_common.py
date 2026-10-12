from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区"
WORK = Path(__file__).resolve().parents[1]
OLD = BASE / "阶段二十D_独立GT与困难场景验证"
FIX = BASE / "阶段二十C_FIX_候选元数据与精简细化验证"
PHASEC = BASE / "阶段二十C_R1与完整细化链联合验证"
PROD = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
METRICS = ["f1", "iou_fg", "miou", "hd95_px", "boundary_f1", "leakage_energy", "oracle_regret"]
SWITCHES = ["CaoMei2_0146", "WanNianQing1_0227"]
DIFFICULTIES = {"D1": "Neighbor-plant adhesion", "D3": "Foreground-background confusion",
                "D6": "Small target", "D11": "Cluttered background", "D12": "Low contrast"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        raise ValueError("Refusing CSV without explicit coverage")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value.rstrip() + "\n")


def relative(path):
    path = Path(path).absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frames():
    rows = read_csv(OLD / "TEST32_INPUT_MANIFEST.csv")
    if len(rows) != 32 or len({row["frame_id"] for row in rows}) != 32:
        raise ValueError("Frozen TEST32 coverage mismatch")
    return rows


def read_mask(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    if set(np.unique(pixels)) - {0, 255}:
        raise ValueError(f"Not binary0/255: {path}")
    return pixels > 0


def metric_function():
    module = load_module("phase20d_direct_metric_loader", PHASEC / "scripts/compute_phase20c_metrics.py")
    return module.load_segmentation_metrics()


def protocol_module():
    return load_module("phase20d_direct_readonly_geometry", OLD / "gt_protocol_v2/scripts/protocol_v2.py")


def production_module():
    sys.path.insert(0, str(PHASEC / "scripts/runtime_compat"))
    sys.path.insert(0, str(PROD))
    return load_module("phase20d_direct_unchanged_production", PROD / "生成RAP-FSAM3掩膜.py")


def table(rows, columns):
    def display(value):
        return f"{value:.8f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for column in columns) + " |"]
    lines.extend("| " + " | ".join(display(row.get(column, "")) for column in columns) + " |" for row in rows)
    return "\n".join(lines)
