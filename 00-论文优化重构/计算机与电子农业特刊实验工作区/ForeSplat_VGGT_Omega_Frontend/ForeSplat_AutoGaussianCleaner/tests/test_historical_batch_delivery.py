"""GPU work stays serial while delivery overlaps, drains and fails visibly."""
import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_batch_scheduler import execute_pipeline
from compact_historical_outputs import compact
from cleaner.io import sha
from run_historical_two_stage import summarize


def test_delivery_overlaps_next_clean_without_queueing_a_third_scene():
    entered = threading.Event()
    release = threading.Event()
    seen = []
    def clean(i):
        if i == 1:
            assert entered.wait(2)
            assert 'delivered0' not in seen
            release.set()
        if i == 2:
            assert 'delivered0' in seen
        seen.append(f'clean{i}')
    def deliver(i):
        if i == 0:
            entered.set()
            assert release.wait(2)
        seen.append(f'delivered{i}')
    execute_pipeline(range(3), clean, deliver)
    assert [s for s in seen if s.startswith('clean')] == ['clean0','clean1','clean2']
    assert [s for s in seen if s.startswith('delivered')] == ['delivered0','delivered1','delivered2']


def test_serial_option_and_empty_inventory():
    seen=[]
    execute_pipeline(range(2),lambda i:seen.append(('clean',i)),
        lambda i:seen.append(('deliver',i)),overlap=False)
    assert seen == [('clean',0),('deliver',0),('clean',1),('deliver',1)]
    execute_pipeline([],lambda _:pytest.fail(),lambda _:pytest.fail())


def test_clean_failure_drains_preceding_delivery_and_stops_inventory():
    delivered=[]
    def clean(i):
        if i == 1:raise ValueError('native failure')
        assert i == 0
    with pytest.raises(ValueError,match='native failure'):
        execute_pipeline(range(5),clean,delivered.append)
    assert delivered == [0]


def test_delivery_failure_is_propagated_including_the_last_scene():
    def fail(i):raise ValueError('delivery failure')
    with pytest.raises(ValueError,match='delivery failure'):
        execute_pipeline([0],lambda _:None,fail)
    seen=[]
    with pytest.raises(ValueError,match='delivery failure'):
        execute_pipeline(range(5),seen.append,fail)
    assert seen in [[0],[0,1]]


def test_compaction_recovers_journal_after_interruption_without_duplicate_bytes(tmp_path):
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(
        source_and_input_verification=True,artifact_hashes={})))
    for stage in ['stage1_generalized','stage2_residual']:
        out=tmp_path/stage
        out.mkdir()
        model=out/'cleaned_gaussians.ply'
        model.write_bytes(b'final retained')
        (out/'COMPLETE.json').write_text(json.dumps(dict(native_sha256=sha(model))))
    folder=tmp_path/'stage1_generalized/rounds/01/proposal'
    folder.mkdir(parents=True)
    record=dict(path=str((folder/'candidate.ply').relative_to(tmp_path)),
        sha256='previously-hashed',bytes=123,decision='decision.json',reason='Completed trial')
    (folder/'RETIRED_CANDIDATE.json').write_text(json.dumps(record))
    # A crash after unlink happened before the aggregate record was published.
    first=compact(tmp_path)
    second=compact(tmp_path)
    assert first['retired'] == second['retired'] == [record]
    assert second['bytes_retired'] == 123
    assert (tmp_path/'stage1_generalized/cleaned_gaussians.ply').read_bytes() == b'final retained'


def test_progress_keeps_detailed_report_and_reports_interrupted_scene(tmp_path):
    final=tmp_path/'FINAL_TWO_STAGE_CLEANER_REPORT.md'
    final.write_text('Detailed measured results and visual failures\n')
    out=tmp_path/'plant'
    out.mkdir()
    (out/'execution.json').write_text('{}')
    summarize(tmp_path,[dict(plant='plant',output=str(out),status='pending')])
    assert final.read_text() == 'Detailed measured results and visual failures\n'
    assert json.loads((tmp_path/'batch_status.json').read_text())[0]['status'] == 'incomplete'
    assert (tmp_path/'BATCH_PROGRESS.md').is_file()
