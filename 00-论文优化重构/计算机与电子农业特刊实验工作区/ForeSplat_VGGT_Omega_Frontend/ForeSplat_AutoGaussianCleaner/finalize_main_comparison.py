#!/usr/bin/env python3
"""Wait for actual C5 experiments, verify their full metrics, then produce common evaluations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import choose


def process_state(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return fields[19],fields[0]
    except (OSError,IndexError):
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs',type=Path,nargs='+',required=True)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--batch-root',type=Path,help='Admit batch outputs only after their isolated CLI child has passed final source/input validation')
    parser.add_argument('--producer-pids',type=int,nargs='+',help='Optional corresponding live producer processes; fail rather than waiting after a crashed producer')
    args=parser.parse_args();base=Path(__file__).resolve().parent
    if args.producer_pids and len(args.producer_pids)!=len(args.outputs):
        raise ValueError('One producer PID is required for each output')
    owners={path:pid for path,pid in zip(args.outputs,args.producer_pids or []) if pid>0}
    owner_identity={path:process_state(pid) for path,pid in owners.items()}
    comparison=base/'experiments/GGGS_Cleaner_Comparison'
    pending=list(args.outputs)
    def ready(output):
        if args.batch_root:
            status_file=args.batch_root/'batch_status.json'
            status=json.loads(status_file.read_text()) if status_file.exists() else {}
            record=status.get(output.name,{})
            if record.get('status') in ['failed','missing_gggs_checkpoint','invalidated','invalid_completed_output']:
                raise ValueError(f'Batch scene failed before common evaluation: {output}: {record}')
            if record.get('status')!='complete':return False
            if Path(record['output']).resolve()!=output.resolve():raise ValueError('Batch completed output path differs from requested evaluation')
        if output in owners:
            state=process_state(owners[output]);identity=owner_identity[output]
            return state is None or identity is None or state[0]!=identity[0] or state[1]=='Z'
        return True
    while pending:
        available=[]
        for path in pending:
            if (path/'INVALIDATED.json').exists():raise ValueError(f'Invalidated experiment: {path}')
            if (path/'COMPLETE.json').exists() and ready(path):available.append(path)
            elif path in owners and not(path/'COMPLETE.json').exists():
                state=process_state(owners[path]);identity=owner_identity[path]
                if state is None or identity is None or state[0]!=identity[0] or state[1]=='Z':raise ValueError(f'Producer exited without a complete result: {path}')
        output=available[0] if available else pending[0]
        if not ready(output):
            if not args.wait:raise ValueError(f'CLI/batch final verification is incomplete: {output}')
            time.sleep(10);continue
        if (output/'INVALIDATED.json').exists():raise ValueError(f'Invalidated experiment: {output}')
        if not (output/'COMPLETE.json').exists():
            if not args.wait:raise ValueError(f'Experiment not complete: {output}')
            for marker in ['FULL_VALIDATION_FAILURE.json','FAILED.json']:
                if (output/marker).exists():raise ValueError(f'Producer failed: {output/marker}')
            if output in owners:
                state=process_state(owners[output]);identity=owner_identity[output]
                if state is None or identity is None or state[0]!=identity[0] or state[1]=='Z':
                    raise ValueError(f'Producer exited without a complete result: {output}')
            time.sleep(10);continue
        if output in owners:
            state=process_state(owners[output]);identity=owner_identity[output]
            if state is not None and identity is not None and state[0]==identity[0] and state[1]!='Z':
                # The CLI audits immutable input/source after engine export.
                time.sleep(10);continue
        if (output/'INVALIDATED.json').exists() or not(output/'COMPLETE.json').exists():
            raise ValueError(f'Final immutable-source audit failed: {output}')
        marker=json.loads((output/'COMPLETE.json').read_text())
        report=json.loads((output/'cleanup_metrics.json').read_text())
        if report['config']!=marker['config']:
            raise ValueError(f'Report configuration differs from completed execution: {output}')
        recorded_selection=report['config'].get('implementation_sources',{}).get('cleaner/adaptive/selection.py')
        if recorded_selection!=sha(base/'cleaner/adaptive/selection.py'):
            raise ValueError('The original experiment selection source differs from this certificate implementation')
        specification=marker['scene']
        if sha(specification['model'])!=report['input_sha256'] or sha(output/'cleaned_gaussians.ply')!=marker['native_sha256']:
            raise ValueError(f'Native/input model differs from recorded experiment: {output}')
        for name,expected in marker['artifact_hashes'].items():
            if sha(output/name)!=expected:raise ValueError(f'Changed completed artifact: {output/name}')
        final=report['final'];initial=report['initial']
        if not final.get('metrics_complete',True) or len(final['views'])!=len(initial['views']):
            raise ValueError('Final comparison must include every original camera')
        accepted,gain,local,reasons=choose(final,initial,initial,report['config'])
        modified=report['removed_gaussians']>0 or report.get('refined_gaussians',0)>0
        write_json(output/'CURRENT_QUALITY_CERTIFICATE.json',dict(accepted=accepted,modified=modified,
            outcome='verified_measured_improvement' if accepted and modified else 'zero_modification' if not modified else 'failed_current_quality',
            reasons=reasons,measured_gain=gain,local_gain=local,input_sha256=report['input_sha256'],native_sha256=marker['native_sha256'],
            scope='Rechecks the completed all-view metrics with the current selection implementation. It does not rerun cleaning, silently replace the original source provenance or certify botanical semantics.',
            source_sha256=sha(base/'cleaner/adaptive/selection.py')))
        print(output.name,'full quality',accepted,'gain',gain,'reasons',reasons,flush=True)
        commands=[['evaluate_black_clouds.py','--outputs',str(output),'--include-original'],
            ['evaluate_cross_plant_quality.py','--outputs',str(output)],
            ['build_free_view_contact.py','--outputs',str(output)],
            ['evaluate_reference_trust.py','--outputs',str(output)]]
        for command in commands:
            subprocess.run(['uv','run','--no-project','--python',sys.executable,'python','-u',str(base/command[0]),*command[1:]],cwd=base.parent,check=True)
        if report['plant']=='KongQueZhuYu':
            subprocess.run(['uv','run','--no-project','--python',sys.executable,'python','-u',str(base/'render_historical_evaluation.py'),
                '--outputs',str(output),'--baseline',str(comparison/'baselines/C0')],cwd=base.parent,check=True)
        subprocess.run(['uv','run','--no-project','--python',sys.executable,'python',str(base/'collect_comparison.py')],cwd=base.parent,check=True)
        write_json(output/'COMMON_COMPARISON_EVALUATION_COMPLETE.json',dict(
            output=str(output.resolve()),input_sha256=report['input_sha256'],native_sha256=marker['native_sha256'],
            status='Native identity, complete original-view quality, paired before/after black clouds, automatic worst-view/free-view images and reference trust evaluated',
            direct_visual_review='Requires actual image inspection; the evaluator does not invent a visual review',
            evaluator_source_sha256=sha(Path(__file__))))
        pending.remove(output)
    manifest_path=comparison/'cross_plant_validation/plant_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    result=dict(outputs=[str(p.resolve()) for p in args.outputs],
        status='All requested native models, full foreground quality certificates, black clouds, worst-view images and reference-trust diagnostics evaluated',
        independent_test_started=(comparison/'cross_plant_validation/INDEPENDENT_TEST_PROTOCOL.json').exists(),global_method_selected=bool(manifest.get('development_decision')))
    completion=(args.batch_root/'COMMON_BATCH_EVALUATION_COMPLETE.json' if args.batch_root else
                comparison/'comparisons/MAIN_COMPARISON_EVALUATION_COMPLETE.json')
    write_json(completion,result)


if __name__=='__main__':main()
