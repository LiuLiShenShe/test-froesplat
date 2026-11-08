"""Current environment and source identities, not reconstructed train-time env."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

AUDIT=Path(__file__).resolve().parents[1]
OMEGA=AUDIT.parents[2]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    files=[('train.py','provenance_official_train.py'),('scene/gaussian_model.py','provenance_official_gaussian_model.py'),('arguments/__init__.py','provenance_official_arguments.py')]
    rows=[];commit='23b418c2f0f19b30b251acad6fe13445dae968c7'
    for rel,snapshot in files:
        local=OMEGA/'gggs_community'/rel;remote=AUDIT/'source_snapshots'/snapshot
        rows.append(dict(local=str(local),snapshot=str(remote),local_sha256=sha(local),official_sha256=sha(remote),
            byte_identical=local.read_bytes()==remote.read_bytes(),normalized_line_endings_identical=local.read_text()==remote.read_text(),
            official_url=f'https://raw.githubusercontent.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/{commit}/{rel}'))
    versions={}
    for key in ['torch','torchvision','numpy','plyfile','Pillow','uv']:
        try:versions[key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:versions[key]=None
    torchpath=Path(importlib.metadata.distribution('torch').locate_file('torch/version.py'))
    tree=ast.parse(torchpath.read_text());torch_literal={}
    for n in tree.body:
        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):
            try:torch_literal[n.target.id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name):
                    try:torch_literal[target.id]=ast.literal_eval(n.value)
                    except (ValueError,TypeError):pass
    wrapper=OMEGA/'gggs_community/train_foresplat.py'
    output=dict(schema='foresplat.source_environment.v1',current_audit_environment=dict(python=sys.version,executable=sys.executable,
        platform=platform.platform(),installed_metadata_versions=versions,torch_version_file_literals=torch_literal,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),
        environment_scope='Current CPU audit env only; E1-E5 train-time complete freeze/cfg was not saved. E6 historical report versions retained separately.',
        official_commit=commit,official_source_comparison=rows,
        custom_training_wrapper=dict(path=str(wrapper),sha256=sha(wrapper),independent_official_source=False,
            git_status=subprocess.run(['git','-C',str(OMEGA/'gggs_community'),'status','--short','--','train_foresplat.py'],text=True,capture_output=True,check=True).stdout.strip()),
        no_training=True,no_gpu_probe=True)
    path=AUDIT/'gaussian_analysis/current_source_environment.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
