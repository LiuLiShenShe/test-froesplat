#!/usr/bin/env python3
"""Record a development decision after real matched experiments and visual review.

This command does not promote sources, edit the plant manifest or start tests.
Those operations can use this record only after their own source checks pass.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.execution_provenance import completed_identity,resolve_variant
from ForeSplat_AutoGaussianCleaner.summarize_cross_plant_comparison import summarize,paired_clouds,verified_visual_review


def read(path):
    return json.loads(Path(path).read_text())


def require_frozen_manifest(path,frozen):
    if sha(path)!=frozen['manifest_sha256']:
        raise ValueError('Development manifest or split changed after the batch was frozen')


def require_reviewed_development(summary,rows,expected):
    if (not summary.get('all_requested_models_evaluated_and_reviewed')
            or len(rows)!=len(expected) or {r['plant'] for r in rows}!=expected):
        raise ValueError('Every real development model must have complete native, paired and visual evidence')
    if any(r['status']!='complete' or r['direct_visual_review_status']!='actual_direct_review_recorded'
            or r['paired_cloud_evaluation_status']!='verified_complete' for r in rows):
        raise ValueError('A development result is incomplete or lacks verified visual evidence')
    if any(r['conspicuous_leaf_loss_in_reviewed_views']!='False' for r in rows):
        raise ValueError('Observed conspicuous leaf loss prevents this production decision')
    if sum(r['modified']=='True' for r in rows)<2:
        raise ValueError('A single modified plant does not establish multi-plant automatic cleaning')


def require_matched_main_identity(report,frozen,common,output,marker):
    config=report['config']
    sources={k:v for k,v in frozen['sources'].items()
        if k.startswith(('cleaner/','evaluation/')) or k in ['auto_clean_gaussians.py','execution_provenance.py']}
    renderer=config.get('native_renderer_fingerprint',{})
    if (config.get('implementation_sources')!=sources
            or config.get('scene_input_fingerprint')!=frozen['scene_contents']['KongQueZhuYu']
            or not renderer or any(frozen['native_renderer_sources'].get(k)!=v for k,v in renderer.items())):
        raise ValueError('Main ablation uses different source, actual scene contents or native renderer')
    if (Path(common['output']).resolve()!=output.resolve()
            or any(common.get(k)!=marker[k] for k in ['input_sha256','native_sha256'])):
        raise ValueError('Common evaluation marker belongs to another native model')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--development-root',type=Path,required=True)
    parser.add_argument('--main-root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--decision-notes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=read(args.manifest)
    if (args.manifest.resolve().parent/'INDEPENDENT_TEST_PROTOCOL.json').exists():
        raise ValueError('Independent testing already started; this command cannot reset its development status')
    if manifest.get('development_decision'):
        raise ValueError('The manifest already records a development decision')
    staged=read(args.development_root/'STAGED_DEVELOPMENT_RESULT.json')
    if staged['exit_code']!=0:
        raise ValueError('Development CLI did not finish successfully')
    frozen=read(args.development_root/'METHOD_FREEZE.json')
    require_frozen_manifest(args.manifest,frozen)
    variant=resolve_variant(frozen['method'],frozen['configuration'])
    for rel,digest in frozen['sources'].items():
        if sha(args.source_root/rel)!=digest:
            raise ValueError('Selected development source changed: '+rel)
    if (sha(args.config)!=frozen['config_sha256']
            or sha(args.source_root/'configs/default.yaml')!=frozen['base_config_sha256']):
        raise ValueError('Selected development configuration changed')
    queue=read(args.main_root/'SERIAL_MAIN_QUEUE_STATUS.json')
    main_models=[]
    main_scene=next(j['scene'] for j in manifest['plants'] if j['plant']=='KongQueZhuYu')
    spec=read(main_scene)
    for method in ['C5-A','C5-B','C5-C','C5-D']:
        status=queue['methods'][method]
        if status.get('exit_code')!=0 or status.get('common_evaluation_exit_code')!=0:
            raise ValueError('Matched main ablation is not complete: '+method)
        output=Path(status['output']);marker=read(output/'COMPLETE.json');report=read(output/'cleanup_metrics.json')
        if (not completed_identity(output,report['config'],spec,method)
                or len(report['initial']['views'])!=208 or len(report['final']['views'])!=208
                or any(report['config'].get(k)!=v for k,v in frozen['configuration'].items())):
            raise ValueError('Main ablation is not the matched original GGGS experiment: '+method)
        require_matched_main_identity(report,frozen,read(output/'COMMON_COMPARISON_EVALUATION_COMPLETE.json'),output,marker)
        paired_clouds(output,report,marker['native_sha256'])
        verified_visual_review(output,marker)
        main_models.append(dict(method=method,output=str(output.resolve()),input_sha256=marker['input_sha256'],
            native_sha256=marker['native_sha256'],outcome=report['outcome'],removed_gaussians=report['removed_gaussians'],
            refined_gaussians=report['refined_gaussians'],cleanup_report_sha256=sha(output/'cleanup_metrics.json'),
            common_evaluation_sha256=sha(output/'COMMON_COMPARISON_EVALUATION_COMPLETE.json'),
            direct_visual_review=str((output/'quality_inspection/DIRECT_VISUAL_REVIEW.json').resolve()),
            direct_visual_review_sha256=sha(output/'quality_inspection/DIRECT_VISUAL_REVIEW.json')))
    destination=args.output.resolve().parent/'development_selection_evidence'
    summarize(args.manifest,args.development_root.resolve(),destination,'development',True)
    rows=list(csv.DictReader((destination/'CROSS_PLANT_GGGS_CLEANING.csv').open()))
    summary=read(destination/'MEASURED_GENERALIZATION_SUMMARY.json')
    require_reviewed_development(summary,rows,{j['plant'] for j in manifest['plants'] if j['group']=='development'})
    notes=args.decision_notes.read_text().strip()
    if not notes:
        raise ValueError('The decision must explain measured gains, losses and residual limitations')
    main=next(m for m in main_models if m['method']==variant)
    kong=next(r for r in rows if r['plant']=='KongQueZhuYu')
    if kong['native_sha256']!=main['native_sha256']:
        raise ValueError('Development reused a different main model')
    result=dict(selected_variant=variant,development_protocol={k:v for k,v in frozen.items() if k!='manifest_sha256'},
        manifest_before_selection_sha256=frozen['manifest_sha256'],
        plant_groups={j['plant']:j['group'] for j in manifest['plants']},
        development_results=[dict(plant=r['plant'],output=str((args.development_root/'development'/r['plant']).resolve()),
            input_sha256=r['input_sha256'],native_sha256=r['native_sha256'],modified=r['modified']=='True',
            visual_review=r['direct_visual_review'],visual_review_sha256=sha(r['direct_visual_review']),
            cleanup_report_sha256=sha(r['cleanup_report']),
            training_background_reduction=float(r['training_background_reduction']),
            free_background_reduction=float(r['free_background_reduction'])) for r in rows],
        matched_main_ablations=main_models,decision_notes=str(args.decision_notes.resolve()),
        decision_notes_sha256=sha(args.decision_notes),decision_rationale=notes,
        development_summary=str((destination/'MEASURED_GENERALIZATION_SUMMARY.json').resolve()),
        development_summary_sha256=sha(destination/'MEASURED_GENERALIZATION_SUMMARY.json'),
        independent_results_used=False,scope='Actual development evidence only; source promotion and untouched testing have not been performed by this recorder')
    if args.output.exists() and read(args.output)!=result:
        raise FileExistsError('A different development decision already exists: '+str(args.output))
    write_json(args.output,result)
    print(args.output.resolve())


if __name__=='__main__':main()
