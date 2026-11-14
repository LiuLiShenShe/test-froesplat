"""Heldout results must not silently mix implementations across resumes."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_historical_two_stage as batch


def test_heldout_evaluation_keeps_fourth_visual_failure_and_requires_two_other_improvements(tmp_path, monkeypatch):
    manifest=[]
    reviews={}
    for plant in batch.DEVELOPMENT:
        out=tmp_path/plant
        out.mkdir()
        (out/'COMPLETE.json').write_text(json.dumps({'source_and_input_verification':True}))
        result={'stage1':{'score':1,'final':{'free_background':{'black_pixels':100}}},
            'stage2':{'score':1,'rolled_back':False,'initial':{'free_background':{'black_pixels':100}}}}
        (out/'cleanup_metrics.json').write_text(json.dumps(result))
        manifest.append({'plant':plant,'output':str(out)})
        reviews[str(out)]={'eligible_for_batch':plant!='WangWenCao1'}
    monkeypatch.setattr(batch,'validated_review',lambda out:reviews[str(out)])
    passed,reason=batch.development_gate(manifest)
    assert passed and 'WangWenCao1' in reason
    assert reviews[str(tmp_path/'WangWenCao1')]['eligible_for_batch'] is False
    reviews[str(tmp_path/'XiangPiShu1')]['eligible_for_batch']=False
    assert batch.development_gate(manifest)[0] is False
    reviews[str(tmp_path/'XiangPiShu1')]['eligible_for_batch']=True
    reviews[str(tmp_path/'WangWenCao1')]=None
    assert batch.development_gate(manifest)[0] is False


def test_frozen_protocol_survives_resume_and_rejects_changed_source(tmp_path, monkeypatch):
    base = tmp_path/'code'
    for name in ['cleaner/algorithm.py', 'evaluation/quality.py',
                 'auto_clean_gaussians.py', 'execution_provenance.py']:
        path = base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# measured implementation\n')
    monkeypatch.setattr(batch, 'BASE', base)
    monkeypatch.setattr(batch, 'DEVELOPMENT', ['Dev'])
    config = base/'config.yaml'
    config.write_text('method_version: frozen-test\n')
    root = tmp_path/'output'
    dev = root/'Dev'
    review = dev/'comparison/official_supersplat/DIRECT_IMAGE_REVIEW.json'
    review.parent.mkdir(parents=True)
    review.write_text('{"direct_image_inspection":true}\n')
    sources = {str(p.relative_to(base)):batch.sha(p) for p in base.rglob('*.py')}
    complete = dev/'COMPLETE.json'
    complete.write_text(json.dumps({'config':{'implementation_sources':sources,
        'method_version':'frozen-test'}}))
    manifest = [{'plant':'Dev', 'output':str(dev)}, {'plant':'Heldout'}]

    batch.freeze_heldout_protocol(root, config, manifest)
    frozen = root/'FROZEN_HELDOUT_PROTOCOL.json'
    original = frozen.read_bytes()
    batch.freeze_heldout_protocol(root, config, manifest)
    assert frozen.read_bytes() == original

    (base/'cleaner/algorithm.py').write_text('# changed after development\n')
    with pytest.raises(RuntimeError, match='Frozen heldout protocol differs'):
        batch.freeze_heldout_protocol(root, config, manifest)
    assert frozen.read_bytes() == original

    frozen.unlink()
    with pytest.raises(RuntimeError, match='different implementation'):
        batch.freeze_heldout_protocol(root, config, manifest)
    assert not frozen.exists()

    (base/'cleaner/algorithm.py').write_text('# measured implementation\n')
    config.write_text('method_version: changed-after-development\n')
    with pytest.raises(RuntimeError, match='different cleaning parameters'):
        batch.freeze_heldout_protocol(root, config, manifest)
    assert not frozen.exists()
