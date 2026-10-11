from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

STAGE = Path(__file__).resolve().parent.parent
ROOT = STAGE.parents[2]
WORK = STAGE.parent
ALG = WORK / '01-算法模块修改与验证'
DATA = ROOT / '00-论文优化重构/数据管理'
S20 = DATA / '07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证'
BENCH = WORK / '06-实验一四样本代表集/E1_Representative4_PottedPlant_VFM'
P12 = ALG / '阶段十二_GT_v2_QA与P6正式验收'
P16 = ALG / '阶段十六_HardCase_GT构建与锁定'
P17A = ALG / '阶段十七A_10GT困难样本Pilot与无GT测试审计'
P17C = ALG / '阶段十七C_候选排序修复开发'
V00 = ALG / '阶段十四_A6A7真实数据析因消融/V00_Control'
BASE_COMMIT = 'c119498c'
cv2.setNumThreads(1)
sys.dont_write_bytecode = True


def relative(path):
    if not path:
        return 'UNKNOWN'
    source = Path(path)
    try:
        return str(source.relative_to(ROOT))
    except ValueError:
        return 'EXTERNAL_PATH_REDACTED'


def local_path(value):
    source = Path(value)
    return source if source.is_absolute() else ROOT / source


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows, fields=None):
    path = STAGE / 'tables' / name
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def report(name, body):
    (STAGE / 'reports' / name).write_text(body.rstrip() + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_origin(path):
    result = subprocess.run(['git', 'log', '-1', '--format=%H', BASE_COMMIT, '--', relative(path)], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip() or 'UNTRACKED_SOURCE'


def mask(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Unreadable mask: {relative(path)}')
    return image > 0


def dimensions(path):
    with Image.open(path) as image:
        return image.size


def rgb_equal(first, second):
    if sha256(first) == sha256(second):
        return True
    with Image.open(first) as image:
        left = np.array(image.convert('RGB'))
    with Image.open(second) as image:
        right = np.array(image.convert('RGB'))
    return np.array_equal(left, right)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def markdown_table(rows, columns):
    text = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join('---' for column in columns) + ' |\n'
    for row in rows:
        text += '| ' + ' | '.join(str(row.get(column, 'UNKNOWN')).replace('|', '/') for column in columns) + ' |\n'
    return text
