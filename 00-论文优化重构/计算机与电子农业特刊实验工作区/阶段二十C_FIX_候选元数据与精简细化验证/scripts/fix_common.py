from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区"
WORK = BASE / "阶段二十C_FIX_候选元数据与精简细化验证"
OLD = BASE / "阶段二十C_R1与完整细化链联合验证"
ALG = BASE / "01-算法模块修改与验证"
PROD = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
PYTHON = "/home/test/biosoft/enter/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def production():
    sys.path.insert(0, str(PROD))
    return load_module("phase20c_fix_production", PROD / "生成RAP-FSAM3掩膜.py")


def relative(path):
    return str(Path(path).relative_to(ROOT))


def link(source, target):
    source = Path(source).resolve(strict=True)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(source)


def manifest():
    return read_csv(OLD / "configs/PHASE20C_INPUT_MANIFEST.csv")
