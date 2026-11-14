#!/usr/bin/env python3
"""Retire reproducible trial PLY copies after verified complete stage export.

Final/native/viewer/removed PLYs, accepted round checkpoints, contribution caches,
source snapshots, edit IDs/CSV, decisions and metrics are retained. Only trial
candidate.ply files under rounds are removed; their SHA and bytes are recorded.
"""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json


def compact(root):
    root=Path(root).resolve()
    marker=root/'COMPLETE.json'
    result=json.loads(marker.read_text())
    if not result.get('source_and_input_verification'):
        raise ValueError('Refusing compaction before verified complete execution')
    for name,digest in result['artifact_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Final artifact hash mismatch: '+name)
    records=[]
    for stage in ['stage1_generalized','stage2_residual']:
        directory=root/stage
        done=json.loads((directory/'COMPLETE.json').read_text())
        if sha(directory/'cleaned_gaussians.ply')!=done['native_sha256']:
            raise ValueError('Stage native hash mismatch')
        for path in sorted((directory/'rounds').glob('*/*/candidate.ply')):
            decision=path.parent/'decision.json'
            if not decision.exists():
                # An untested proposal may belong to an incomplete substep.
                continue
            record=dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size,
                decision=str(decision.relative_to(root)),reason='Completed trial; final stage artifacts and exact edit provenance retained')
            write_json(path.parent/'RETIRED_CANDIDATE.json',record)
            path.unlink();records.append(record)
    output=root/'TRIAL_COMPACTION.json'
    previous=json.loads(output.read_text()) if output.exists() else dict(retired=[])
    # Every deletion has its own atomic journal written before unlink. Recover
    # those records after interruption, even if the aggregate was not written.
    retired={r['path']:r for r in previous['retired']}
    for stage in ['stage1_generalized','stage2_residual']:
        for journal in (root/stage/'rounds').glob('*/*/RETIRED_CANDIDATE.json'):
            record=json.loads(journal.read_text())
            if not (root/record['path']).exists():
                retired[record['path']]=record
    retired.update({r['path']:r for r in records})
    previous['retired']=[retired[key] for key in sorted(retired)]
    previous.update(updated=time.time(),bytes_retired=sum(r['bytes'] for r in previous['retired']),
        retained='All final, removed and accepted-round PLYs; exact stage edits; attribution; metrics; source and input provenance')
    write_json(output,previous)
    return previous


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',type=Path)
    a=p.parse_args();r=compact(a.experiment)
    print(json.dumps(dict(retired_trials=len(r['retired']),bytes_retired=r['bytes_retired'])))


if __name__=='__main__':main()
