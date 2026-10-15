"""Phase-only paths, immutable protocols and read-only asset access."""
from __future__ import annotations
import csv, hashlib, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STAGE=Path(__file__).resolve().parents[1]
WORK=STAGE.parent
ROOT=STAGE.parents[2]
GS=ROOT/'2d-gaussian-splatting-main'
BRANCH=ROOT/'2d-gaussian-splatting-m2m3-floor40'
E0=WORK/'阶段二十E0_三维全视图与输入冻结'
E1=WORK/'阶段二十E1_正式2DGS配对训练与重建评价'
E2=WORK/'阶段二十E2_多种子稳定性与前景三维评价'
DATA=Path('/data/fj/F2DMAS_phase20e3_data/1771e53f_v1')
PYTHON=GS/'venv/bin/python'
GPU_INDEX='1'
GPU_UUID='GPU-6836f890-98b2-5891-07ef-675c25199c65'
SCENES=('WanNianQing1','CaoMei2','HongZhang')
BASELINE='1771e53fe147714e4c69bb9b833064ddddf098e9'

def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):d.update(b)
    return d.hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,obj):
    import numpy as np
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=convert)+'\n')
def rows(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write_csv(p,data,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in data for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(data)
def import_paths(branch=False):
    sys.dont_write_bytecode=True
    for p in [GS/'submodules/simple-knn',GS/'submodules/diff-surfel-rasterization',BRANCH if branch else GS]:
        sys.path.insert(0,str(p))
def gpu_idle():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_INDEX
    actual=subprocess.check_output(['nvidia-smi','-i',GPU_INDEX,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert actual==GPU_UUID
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(GPU_UUID in line for line in processes.splitlines()),'Designated GPU busy; stop, never switch GPU'
def config(scene):return read_json(E0/f'reports/{scene}_B_R0_TRAIN_CONFIG.json')
def reference(scene):return next(x for x in read_json(E2/'configs/NEXT_PHASE_MESH_REFERENCE.json')['models'] if x['sample']==scene)
def model_path(scene,method):
    return Path(reference(scene)['checkpoint_path']).parent if method=='M0' else DATA/'training'/scene/method/'attempt_01'
def verify_frozen(full=False):
    frozen=read_json(STAGE/'configs/FREEZE.json')
    if full:
        for x in frozen['protected']:
            assert sha(x['path'])==x['sha256'],x['path']
    for x in frozen['configs']:
        assert sha(x['path'])==x['sha256'],x['path']
    return frozen
def metric_module():
    import importlib.util
    s=importlib.util.spec_from_file_location('e2_metric_core',E2/'scripts/metric_core.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
