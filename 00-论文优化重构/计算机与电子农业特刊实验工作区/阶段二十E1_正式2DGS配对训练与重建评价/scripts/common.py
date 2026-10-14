from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
STAGE = Path(__file__).resolve().parents[1]
E0 = STAGE.parent / "阶段二十E0_三维全视图与输入冻结"
GS = ROOT / "2d-gaussian-splatting-main"
DATA = Path("/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1")
PYTHON = GS / "venv/bin/python"
SAMPLES = ("WanNianQing1", "CaoMei2", "HongZhang")
GROUPS = ("A_raw_RGB_control_v1", "B_R0", "C_R1")
MATRIX = [(sample, group) for sample in SAMPLES for group in GROUPS
          if (sample, group) != ("HongZhang", "B_R0")]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def config_for(sample, group):
    return json.loads((E0 / f"reports/{sample}_{group}_TRAIN_CONFIG.json").read_text())


def import_paths():
    import sys
    sys.dont_write_bytecode = True
    for path in (GS, GS / "submodules/simple-knn", GS / "submodules/diff-surfel-rasterization"):
        sys.path.insert(0, str(path))
