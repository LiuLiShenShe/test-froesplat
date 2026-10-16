"""Public API. No training imports, optimizer state, or checkpoint writes."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from .io import read_model, resolve_model, sha256, write_subset
from .pruning import rank_and_select, removal_budget
from .scoring import score_model


@dataclass(frozen=True)
class CompressionResult:
    output_dir: Path
    model_path: Path
    retained_indices_path: Path
    removed_indices_path: Path
    manifest_path: Path
    input_count: int
    output_count: int
    source_model_sha256: str
    output_model_sha256: str


def _json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def compress_model(input_model, output_dir, method, prune_ratio, config=None):
    """Prune a frozen 2DGS PLY into a new directory (never overwrite).

    Remove floor(N*ratio), keep highest scores; preserve source order and all
    scalar properties. config.enabled=False makes a byte-identical copy and
    identity indices, independent of method; it does not initialize a scorer.
    External scores require a file hash and source-model-bound provenance.
    Recovery optimization is deliberately unsupported in this pilot.
    """
    started = time.perf_counter()
    config = dict(config or {})
    if not isinstance(config.get('enabled',True),bool):
        raise ValueError('enabled must be a JSON boolean')
    if config.get('recovery',False) not in [False,None]:
        raise ValueError('Post-pruning recovery is not implemented; no training will be started')
    # Fail before creating output if configuration is not serializable.
    json.dumps(config,allow_nan=False)
    source = resolve_model(input_model,config.get('model_iteration'))
    given = Path(input_model).expanduser().resolve(strict=True)
    protected_dir = given if given.is_dir() else source.parent
    output = Path(output_dir).expanduser().resolve()
    if output == protected_dir or protected_dir in output.parents:
        raise ValueError('Output must be outside the protected source model directory')
    if output.exists():
        raise FileExistsError(f'Refusing to replace any existing output: {output}')
    source_hash = sha256(source)
    if config.get('expected_source_sha256',source_hash) != source_hash:
        raise ValueError('Frozen source model SHA256 mismatch')
    model = read_model(source)
    enabled = config.get('enabled',True)
    if enabled:
        removal_budget(model.count,prune_ratio)
    max_degree = model.sh_degree
    active_degree = config.get('active_sh_degree',max_degree)
    if isinstance(active_degree,bool) or not isinstance(active_degree,int) or not 0 <= active_degree <= max_degree:
        raise ValueError('Invalid active_sh_degree')
    output.mkdir(parents=True,exist_ok=False)
    header = {'api_version':1,'status':'RUNNING','created_at':datetime.now(timezone.utc).isoformat(),
              'enabled':enabled,'method':method,'requested_prune_ratio':str(prune_ratio),
              'config':config,'source_model_path':str(source),'input_model_argument':str(given),
              'source_model_sha256':source_hash,'source_bytes':source.stat().st_size,
              'input_count':model.count,'sh_degree':max_degree,'active_sh_degree':active_degree,
              'active_SH_source':'explicit config' if 'active_sh_degree' in config else 'native PLY loader convention: max degree',
              'schema':model.schema,'recovery_steps':0,'score_unit':'per original Gaussian',
              'budget_policy':'remove floor(N * Decimal(str(prune_ratio)))',
              'tie_policy':'higher score retained; equal score prefers lower original index',
              'output_row_order':'ascending original index'}
    _json(output/'compression_config.json',header)
    try:
        if enabled:
            tick = time.perf_counter()
            scores, provenance = score_model(model,method,config,source_hash)
            scoring_seconds = time.perf_counter()-tick
            retained,removed = rank_and_select(scores,prune_ratio)
            np.save(output/'importance_scores.npy',scores,allow_pickle=False)
            _json(output/'score_provenance.json',provenance)
        else:
            retained = np.arange(model.count,dtype=np.int64)
            removed = np.empty(0,dtype=np.int64)
            provenance = {'scorer':'DISABLED_IDENTITY','source_model_sha256':source_hash}
            scoring_seconds = 0.0
            _json(output/'score_provenance.json',provenance)
        destination = output/'pruned_model.ply'
        if not len(removed):
            shutil.copyfile(source,destination)
            reloaded = read_model(destination)
        else:
            reloaded = write_subset(model,retained,destination)
        if reloaded.count != len(retained) or reloaded.schema != model.schema:
            raise RuntimeError('PLY roundtrip validation failed')
        if not np.array_equal(reloaded.vertices,model.vertices[retained]):
            raise RuntimeError('PLY/source row correspondence failed')
        np.save(output/'retained_indices.npy',retained,allow_pickle=False)
        np.save(output/'removed_indices.npy',removed,allow_pickle=False)
        if len(np.unique(np.concatenate([retained,removed]))) != model.count:
            raise RuntimeError('Index partition failed')
        if sha256(source) != source_hash:
            raise RuntimeError('Source changed during compression')
        output_hash = sha256(destination)
        artifacts = {p.name:{'sha256':sha256(p),'bytes':p.stat().st_size}
                     for p in output.iterdir() if p.name != 'compression_config.json'}
        header.update(status='PASS',output_count=len(retained),removed_count=len(removed),
                      actual_prune_ratio=len(removed)/model.count,output_bytes=destination.stat().st_size,
                      output_model_sha256=output_hash,scoring_seconds=scoring_seconds,
                      compression_wall_seconds=time.perf_counter()-started,input_unchanged=True,
                      schema_unchanged=True,SH_degree_unchanged=True,all_retained_values_identical=True,
                      artifacts=artifacts)
        _json(output/'compression_config.json',header)
        (output/'source_model_sha256').write_text(source_hash+'\n')
        (output/'output_model_sha256').write_text(output_hash+'\n')
        (output/'compression_log.jsonl').write_text(json.dumps({'event':'COMPLETE',
            'input_count':model.count,'output_count':len(retained),'source_model_sha256':source_hash,
            'output_model_sha256':output_hash,'recovery_steps':0})+'\n')
        return CompressionResult(output,destination,output/'retained_indices.npy',output/'removed_indices.npy',
                                 output/'compression_config.json',model.count,len(retained),source_hash,output_hash)
    except BaseException as error:
        header.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
        _json(output/'compression_config.json',header)
        raise
