"""Phase20G0-only paths and serialization; never writes historical assets."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
WORK = STAGE.parent
ROOT = STAGE.parents[2]
GS = ROOT / '2d-gaussian-splatting-main'
E0 = WORK / '阶段二十E0_三维全视图与输入冻结'
E1 = WORK / '阶段二十E1_正式2DGS配对训练与重建评价'
E2 = WORK / '阶段二十E2_多种子稳定性与前景三维评价'
E3 = WORK / '阶段二十E3_M2M3真实消融与Mesh验证'
EF = WORK / '阶段二十F_植物表型与尺度恢复'
ASSETS = Path('/data/fj/ForeSplat_Phase20G0_assets')
DATA = ASSETS / '723c02c2_v1'
UPSTREAM = ASSETS / 'upstream'
PYTHON = GS / 'venv/bin/python'
GPU_UUID = 'GPU-6836f890-98b2-5891-07ef-675c25199c65'
BASELINE = '723c02c20e281b150d38144e61c912e5128849b4'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if hasattr(x, 'item'):
            return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert, allow_nan=False) + '\n')


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, data, fields=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in data for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(data)


def import_paths():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    for p in [GS / 'submodules/simple-knn', GS / 'submodules/diff-surfel-rasterization', GS]:
        sys.path.insert(0, str(p))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def assert_gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Use the frozen physical GPU 1 for every rendering run')
    actual = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if actual != GPU_UUID:
        raise RuntimeError('GPU UUID changed')


def plan():
    p = STAGE / 'configs/EXPERIMENT_PLAN.json'
    d = read_json(p)
    if sha(p) != read_json(STAGE / 'configs/PLAN_HASH.json')['sha256']:
        raise RuntimeError('Predeclared experiment plan changed')
    return d
