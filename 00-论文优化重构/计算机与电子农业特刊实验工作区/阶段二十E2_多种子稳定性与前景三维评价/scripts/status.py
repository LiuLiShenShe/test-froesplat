"""Read-only progress snapshot; never inspects quality to control training."""
from __future__ import annotations
import json
from common import *


def main():
    def completed_json(path):
        try:return read_json(path)
        except (json.JSONDecodeError,FileNotFoundError):return {}
    validations=[completed_json(p) for p in (STAGE/'reports').glob('*_seed[12]_attempt*_VALIDATION.json')]
    completed=[r for r in validations if r.get('status')=='PASS']
    result={'timestamp':now(),'completed_and_reloadable':len(completed),'required':8}
    active_path=STAGE/'reports/ACTIVE_TRAINING.json'
    if active_path.exists():
        active=completed_json(active_path);result['active']=active.get('run_id',active.get('status'))
        if 'run_id' in active:
            curve=STAGE/f'logs/{active["run_id"]}_TRAINING_CURVE.csv'
            if curve.exists():
                with curve.open('rb') as handle:
                    handle.seek(0,2);size=handle.tell();handle.seek(max(0,size-64000))
                    lines=handle.read().decode(errors='replace').strip().splitlines()
                result['iteration']=0
                for line in reversed(lines):
                    last=line.split(',')
                    if len(last)!=9 or not last[0].isdigit():continue
                    try:result.update({'iteration':int(last[0]),'Gaussians':int(last[4]),'wall_seconds_since_train_entry':float(last[6])})
                    except ValueError:continue
                    break
    result['evaluations_completed']=len(list((STAGE/'reports').glob('*_EVALUATION.json')))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
