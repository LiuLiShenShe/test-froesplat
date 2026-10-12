from __future__ import annotations

import argparse
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
WORK = Path(__file__).resolve().parents[1]
BASE = WORK.parent
FINAL = BASE / "阶段二十D_固定轮廓直接评价"
OLD = BASE / "阶段二十D_独立GT与困难场景验证"
PROD = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
GS = ROOT / "2d-gaussian-splatting-main"
DATA = Path("/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1")
SAMPLES = ("WanNianQing1", "CaoMei2", "HongZhang")
EXPECTED = {
    "生成RAP-FSAM3掩膜.py": "3056f3fd5a95926b295671d586bc77dbe69c0014a77439daf336ab2ab03e2be8",
    "ranking_policy.py": "c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905",
    "candidate_metadata.py": "96fa57b293fe057a5b2f9b696d1e77c3387ddf41ba16a9ad4d52000a629f6d49",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def local(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    path = Path(path).absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        raise ValueError(f"Missing CSV schema: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def colmap_io():
    return load_module("phase20e0_colmap_io", GS / "scene/colmap_loader.py")


def production():
    for filename, expected in EXPECTED.items():
        if sha256(PROD / filename) != expected:
            raise ValueError(f"Frozen production changed: {filename}")
    compat = BASE / "阶段二十C_R1与完整细化链联合验证/scripts/runtime_compat"
    sys.path.insert(0, str(compat))
    sys.path.insert(0, str(PROD))
    return load_module("phase20e0_production", PROD / "生成RAP-FSAM3掩膜.py")


def frozen_parameters(policy="r1", device="cpu"):
    parameters = json.loads((FINAL / "logs/full_r1_runtime_parameters.json").read_text())
    parameters["ranking_policy"] = policy
    parameters["device"] = device
    for key in ("use_prompt_ensemble", "use_temporal_alignment", "use_semantic_gate",
                "use_spnp_refinement", "use_residual_repair", "use_reprompt_detection",
                "use_geometry_feedback", "use_corrective_geometry", "use_cross_view_consensus",
                "use_memory_propagation", "use_spnp_evidence_guidance"):
        if parameters[key]:
            raise ValueError(f"Not V1: {key}")
    if (parameters["candidate_mode"], parameters["prompt_list"], parameters["default_prompt_id"],
        parameters["prompt_selection_mode"]) != ("per_instance", "P6", "P6", "single"):
        raise ValueError("Foreground recipe drift")
    return parameters


def namespace(parameters):
    return argparse.Namespace(**{key: Path(value) if key in ("sam3_repo", "sam3_checkpoint") else value
                                for key, value in parameters.items()})


def image_array(path, mode="RGB"):
    with Image.open(path) as image:
        image.load()
        return np.array(image.convert(mode))


def binary_mask(path):
    array = image_array(path, "L")
    if not set(np.unique(array)).issubset({0, 255}):
        raise ValueError(f"Nonbinary mask: {path}")
    return array.astype(bool)


def save_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def safe_copy(source, target):
    import shutil
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"Refusing overwrite: {target}")
    else:
        shutil.copy2(source, target)


def scene_rows(sample):
    return [row for row in read_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv") if row["sample"] == sample]


def view_key(row):
    return f"{row['sample']}_{row['raw_frame']}"
