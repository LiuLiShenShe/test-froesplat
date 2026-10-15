from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
STAGE = Path(__file__).resolve().parents[1]
WORK = STAGE.parent
E0 = WORK / "阶段二十E0_三维全视图与输入冻结"
E1 = WORK / "阶段二十E1_正式2DGS配对训练与重建评价"
DATA = Path("/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1")
E1_OUTPUT = DATA / "phase20e1_outputs"
E2_DATA = Path("/data/fj/F2DMAS_phase20e2_data/694743e7_v1")
E2_OUTPUT = E2_DATA / "phase20e2_outputs"
GS = ROOT / "2d-gaussian-splatting-main"
PYTHON = GS / "venv/bin/python"
SCENES = ("WanNianQing1", "CaoMei2")
GROUPS = ("B_R0", "C_R1")
SEEDS = (0, 1, 2)
NEW_SEEDS = (1, 2)
MATRIX = tuple((scene, group, seed) for scene in SCENES for seed in NEW_SEEDS for group in GROUPS)
TRAIN_ORDER = (
    ("WanNianQing1", "B_R0", 1),
    ("WanNianQing1", "C_R1", 1),
    ("WanNianQing1", "B_R0", 2),
    ("WanNianQing1", "C_R1", 2),
    ("CaoMei2", "B_R0", 1),
    ("CaoMei2", "C_R1", 1),
    ("CaoMei2", "B_R0", 2),
    ("CaoMei2", "C_R1", 2),
)
GPU_INDEX = "1"
GPU_UUID = "GPU-6836f890-98b2-5891-07ef-675c25199c65"
BASELINE_COMMIT = "694743e7201e1fd79b4d7cff89ad72a4db8a9e33"
E1_TRAIN_COMMIT = "3e909f93"
MAX_FULL_ATTEMPTS = 2


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"Cannot infer CSV fields for empty rows: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def e0_config(scene, group):
    return read_json(E0 / f"reports/{scene}_{group}_TRAIN_CONFIG.json")


def e2_config_path(scene, group, seed):
    return STAGE / f"configs/{scene}_{group}_seed{seed}_TRAIN_CONFIG.json"


def output_path(scene, group, seed, attempt=1):
    return E2_OUTPUT / scene / group / f"seed_{seed}" / f"attempt_{attempt:02d}"


def run_id(scene, group, seed, attempt=None):
    value = f"{scene}_{group}_seed{seed}"
    return value if attempt is None else f"{value}_attempt{attempt:02d}"


def import_paths():
    import sys
    sys.dont_write_bytecode = True
    for path in (GS, GS / "submodules/simple-knn", GS / "submodules/diff-surfel-rasterization"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
