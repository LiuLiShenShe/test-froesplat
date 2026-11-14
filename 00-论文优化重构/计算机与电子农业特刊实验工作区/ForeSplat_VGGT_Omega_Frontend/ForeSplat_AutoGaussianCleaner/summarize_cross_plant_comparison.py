#!/usr/bin/env python3
"""Join frozen native cleaning, paired evaluations and actual visual reviews.

This reporter never cleans a model, changes a configuration or selects a method.
Pending and failed jobs keep empty measurements, even if a stale PLY exists.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import choose
from ForeSplat_AutoGaussianCleaner.execution_provenance import completed_identity, resolve_variant
from ForeSplat_AutoGaussianCleaner.integration import read_cleaning_result


def read(path):
    return json.loads(Path(path).read_text())


def reduction(before, after):
    return (before-after)/before if before > 0 else None


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verified_visual_review(output, result):
    """Bind actual review records to both current contact sheets and their PNGs."""
    inspection = output/'quality_inspection'
    review = read(inspection/'DIRECT_VISUAL_REVIEW.json')
    for key in ['native_sha256', 'input_sha256']:
        if review[key] != result[key]:
            raise ValueError('Direct visual review belongs to another model')
    expected = {(inspection/name).resolve() for name in
                ['WORST_VIEW_CONTACT_SHEET.png', 'FREE_VIEW_CONTACT_SHEET.png']}
    reviewed = {Path(path).resolve() for path in review['images']}
    if not expected.issubset(reviewed):
        raise ValueError('Direct review must cover observed worst views and all automatic free views')
    hashes = {Path(path).resolve(): value for path, value in review['image_sha256'].items()}
    if any(hashes.get(path) != sha(path) for path in reviewed):
        raise ValueError('A reviewed image changed after the actual visual inspection')
    if (not review.get('scope') or not review.get('review_method')
            or any(type(review[key]) is not bool for key in
                   ['conspicuous_leaf_loss_in_reviewed_views', 'residual_black_cloud_visible'])):
        raise ValueError('Direct review must state its scope and actual leaf-loss/residual observations')
    observed = read(inspection/'QUALITY_INSPECTION.json')
    free = read(inspection/'FREE_VIEW_CONTACT_SHEET.json')
    for data in [observed, free]:
        if any(data[key] != result[key] for key in ['native_sha256', 'input_sha256']):
            raise ValueError('Visual comparison metadata belongs to another model')
        if Path(data['contact_sheet']).resolve() not in expected:
            raise ValueError('Visual comparison metadata refers to an unexpected contact sheet')
    if free['contact_sheet_sha256'] != sha(Path(free['contact_sheet'])):
        raise ValueError('Automatic free-view contact sheet changed after generation')
    for source in free['images']:
        if source['image_sha256'] != sha(Path(source['image'])):
            raise ValueError('A native free-view source PNG changed after comparison generation')
    return review


def paired_clouds(output, report, native_sha):
    before = read(output/'ORIGINAL_BLACK_CLOUD_METRICS.json')
    after = read(output/'BLACK_CLOUD_METRICS.json')
    if (before['native_sha256'] != report['input_sha256']
            or before['input_sha256'] != report['input_sha256']
            or after['native_sha256'] != native_sha
            or after['input_sha256'] != report['input_sha256']):
        raise ValueError('Paired black-cloud evaluation belongs to another native model')
    if (before['resolution_max_side'] != after['resolution_max_side']
            or before['definition'] != after['definition']):
        raise ValueError('Before/after black clouds use different measurement protocols')
    expected = {(v['index'], v['name'], background)
                for v in report['initial']['views'] for background in ['black', 'white']}
    for data in [before, after]:
        actual = [(v['index'], v['name'], v['background']) for v in data['training_views']]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Black-cloud measurement must include both backgrounds and every native camera')
    if ([(v['name'], v['background']) for v in before['free_views']]
            != [(v['name'], v['background']) for v in after['free_views']]):
        raise ValueError('Paired free-view black clouds use different camera inventories')
    row = {}
    for background in ['black', 'white']:
        for key, old in before['training_mean'][background].items():
            new = after['training_mean'][background][key]
            row.update({f'{background}_{key}_before': old,
                        f'{background}_{key}_after': new,
                        f'{background}_{key}_reduction': reduction(old, new)})
        views_old = [v for v in before['free_views'] if v['background'] == background]
        views_new = [v for v in after['free_views'] if v['background'] == background]
        for key in before['training_mean'][background]:
            old = statistics.mean(v[key] for v in views_old)
            new = statistics.mean(v[key] for v in views_new)
            row.update({f'free_{background}_{key}_before': old,
                        f'free_{background}_{key}_after': new,
                        f'free_{background}_{key}_reduction': reduction(old, new)})
    return row


def summarize(manifest_path, batch_root, destination, group='all', require_complete=False):
    manifest, frozen = read(manifest_path), read(batch_root/'METHOD_FREEZE.json')
    status = read(batch_root/'batch_status.json') if (batch_root/'batch_status.json').exists() else {}
    jobs = [j for j in manifest['plants'] if group == 'all' or j['group'] == group]
    requested = resolve_variant(frozen['method'], frozen['configuration'])
    source = Path(__file__).resolve().parent/'cleaner/adaptive/selection.py'
    ledger = manifest_path.resolve().parent/'INDEPENDENT_TEST_PROTOCOL.json'
    if any(j['group'] == 'independent_test' and status.get(j['plant'], {}).get('status') == 'complete' for j in jobs):
        independent = read(ledger)
        if (independent['protocol'] != {k:v for k,v in frozen.items() if k != 'manifest_sha256'}
                or independent['plants'] != sorted(j['plant'] for j in manifest['plants'] if j['group'] == 'independent_test')):
            raise ValueError('Independent results do not belong to the frozen heldout protocol')
    rows = []
    for job in jobs:
        spec = read(job['scene'])
        output = batch_root/job['group']/job['plant']
        current = status.get(job['plant'], {})
        row = dict(plant=job['plant'], group=job['group'], status=current.get('status', 'pending'),
                   original_native_ply=str(Path(spec['model']).resolve()),
                   scene_config=str(Path(job['scene']).resolve()), method=requested,
                   native_ply=None, supersplat_ply=None, removed_native_ply=None,
                   removed_supersplat_ply=None, original_gaussian_ids=None,
                   removed_original_gaussian_ids=None, cleanup_report=None,
                   original_gaussians=None, cleaned_gaussians=None, removed_gaussians=None,
                   refined_gaussians=None, modified=None, training_background_reduction=None,
                   free_background_reduction=None, cleaning_seconds=None, child_wall_seconds=None,
                   peak_allocated_mib=None, sampled_process_peak_mib=None,
                   conspicuous_leaf_loss_in_reviewed_views=None, residual_black_cloud_visible=None,
                   paired_cloud_evaluation_status='pending', direct_visual_review_status='pending')
        if job.get('training_dir'):
            training = Path(job['training_dir'])/'densification_events.complete.json'
            if training.exists():
                data = read(training)
                row.update(training_report=str(training.resolve()),
                           training_wall_seconds=data['wall_seconds'],
                           training_peak_allocated_gib=data.get('peak_allocated_gib'),
                           training_cost_scope=data.get('training_cost_scope', 'Recorded native training wall time; see source report for pauses'),
                           training_previous_wall_lower_bound=data.get('previous_observed_wall_seconds_lower_bound'),
                           training_repeated_iterations=data.get('repeated_iterations', 0))
        # Authoritative child status is required, regardless of stale COMPLETE/PLY files.
        if row['status'] != 'complete':
            row['failure_reason'] = current.get('reason')
            rows.append(row); continue
        try:
            if Path(current['output']).resolve() != output.resolve():
                raise ValueError('Authoritative child output differs from this batch scene')
            marker, report = read(output/'COMPLETE.json'), read(output/'cleanup_metrics.json')
            if (report['method'] != requested
                    or not completed_identity(output, report['config'], spec, requested)
                    or marker['config'] != report['config']
                    or any(report['config'].get(k) != v for k,v in frozen['configuration'].items())
                    or any(frozen['sources'].get(k) != v for k,v in report['config']['implementation_sources'].items())):
                raise ValueError('Completed result does not match the frozen native cleaning protocol')
            if report['config']['implementation_sources']['cleaner/adaptive/selection.py'] != sha(source):
                raise ValueError('This reporter cannot certify a different quality-selection implementation')
            accepted, gain, local, reasons = choose(report['final'], report['initial'], report['initial'], report['config'])
            result = read_cleaning_result(job['scene'], output)
            if result['modified'] and not accepted:
                raise ValueError('Edited model failed the original full-view quality constraints: '+str(reasons))
            row.update(native_ply=result['native_model'], supersplat_ply=result['supersplat_model'],
                       removed_native_ply=result['removed_model'],
                       removed_supersplat_ply=str((output/'removed_gaussians_supersplat.ply').resolve()),
                       original_gaussian_ids=result['original_gaussian_ids'],
                       removed_original_gaussian_ids=str((output/'removed_original_gaussian_ids.npy').resolve()),
                       cleanup_report=result['report'], input_sha256=result['input_sha256'],
                       native_sha256=result['native_sha256'], outcome=result['status'], modified=result['modified'],
                       original_gaussians=report['original_gaussians'], cleaned_gaussians=report['cleaned_gaussians'],
                       removed_gaussians=report['removed_gaussians'], refined_gaussians=report['refined_gaussians'],
                       training_background_reduction=report['artifact_reduction'],
                       free_background_reduction=report['free_artifact_reduction'],
                       cleaning_seconds=report['seconds'], child_wall_seconds=current.get('child_wall_seconds'),
                       peak_allocated_mib=report['peak_allocated_mib'],
                       sampled_process_peak_mib=report.get('gpu_memory', {}).get('sampled_process_peak_mib'),
                       gpu_sampling_interval_seconds=report.get('gpu_memory', {}).get('sampling_interval_seconds'),
                       evaluated_native_cameras=len(report['final']['views']), stop_reason=report['stop_reason'])
            initial, final = report['initial']['mean'], report['final']['mean']
            for key, old in initial.items():
                if numeric(old) and numeric(final.get(key)):
                    row.update({key+'_before': old, key+'_after': final[key], key+'_delta': final[key]-old})
            if (output/'BLACK_CLOUD_METRICS.json').exists() and (output/'ORIGINAL_BLACK_CLOUD_METRICS.json').exists():
                row.update(paired_clouds(output, report, result['native_sha256']))
                row['paired_cloud_evaluation_status'] = 'verified_complete'
                row['original_cloud_metrics'] = str((output/'ORIGINAL_BLACK_CLOUD_METRICS.json').resolve())
                row['cleaned_cloud_metrics'] = str((output/'BLACK_CLOUD_METRICS.json').resolve())
            review_path = output/'quality_inspection/DIRECT_VISUAL_REVIEW.json'
            if review_path.exists():
                review = verified_visual_review(output, result)
                row.update(direct_visual_review_status='actual_direct_review_recorded',
                           direct_visual_review=str(review_path.resolve()),
                           conspicuous_leaf_loss_in_reviewed_views=review['conspicuous_leaf_loss_in_reviewed_views'],
                           residual_black_cloud_visible=review['residual_black_cloud_visible'],
                           visual_review_scope=review['scope'])
            inspection_path = output/'quality_inspection/QUALITY_INSPECTION.json'
            if inspection_path.exists():
                inspection = read(inspection_path)
                if (inspection['native_sha256'] != result['native_sha256']
                        or inspection['input_sha256'] != result['input_sha256']):
                    raise ValueError('Visual contact sheet belongs to another model')
                row['before_after_contact_sheet'] = inspection['contact_sheet']
            row['automatic_free_view_comparison_root'] = str((output/'quality_inspection').resolve())
        except (OSError, ValueError, KeyError) as error:
            row.update(status='invalid_completed_output', failure_reason=str(error), modified=None)
        rows.append(row)
    destination.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (destination/'CROSS_PLANT_GGGS_CLEANING.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups = {}
    for name in sorted({j['group'] for j in jobs}):
        subset = [r for r in rows if r['group'] == name]
        valid = [r for r in subset if r['status'] == 'complete']
        reviewed = [r for r in valid if r['direct_visual_review_status'] == 'actual_direct_review_recorded']
        groups[name] = dict(expected=len(subset), completed=len(valid),
            modified=sum(r['modified'] for r in valid), zero_modification=sum(not r['modified'] for r in valid),
            pending=sum(r['status'] == 'pending' for r in subset),
            running=sum(r['status'] in ['running', 'waiting_for_gggs_checkpoint'] for r in subset),
            interrupted=sum(r['status'] == 'interrupted' for r in subset),
            failures=[dict(plant=r['plant'], status=r['status'], reason=r.get('failure_reason')) for r in subset
                      if r['status'] in ['failed', 'invalid_completed_output', 'invalidated', 'missing_gggs_checkpoint']],
            not_completed=[dict(plant=r['plant'], status=r['status'], reason=r.get('failure_reason'))
                           for r in subset if r['status'] != 'complete'],
            direct_visual_reviews=len(reviewed),
            conspicuous_leaf_loss_in_reviewed_views=sum(r['conspicuous_leaf_loss_in_reviewed_views'] for r in reviewed) if reviewed else None,
            visible_residual_black_clouds=sum(r['residual_black_cloud_visible'] for r in reviewed) if reviewed else None)
        for metric in ['training_background_reduction', 'free_background_reduction', 'fg_psnr_delta', 'fg_mse_delta',
                       'cleaning_seconds', 'peak_allocated_mib', 'sampled_process_peak_mib',
                       'black_largest_dark_bg_component_pixels_reduction', 'white_largest_dark_bg_component_pixels_reduction']:
            values = [r[metric] for r in valid if numeric(r.get(metric))]
            groups[name][metric] = dict(measured=len(values), mean=statistics.mean(values) if values else None,
                                       median=statistics.median(values) if values else None,
                                       minimum=min(values) if values else None, maximum=max(values) if values else None)
    all_complete = (len(rows) == len(jobs)
                    and all(r['status'] == 'complete' and r['paired_cloud_evaluation_status'] == 'verified_complete'
                            and r['direct_visual_review_status'] == 'actual_direct_review_recorded' for r in rows))
    write_json(destination/'MEASURED_GENERALIZATION_SUMMARY.json', dict(
        group=group, groups=groups, all_requested_models_evaluated_and_reviewed=all_complete,
        manifest=str(manifest_path.resolve()), batch_root=str(batch_root.resolve()),
        method_freeze=str((batch_root/'METHOD_FREEZE.json').resolve()), method_freeze_sha256=sha(batch_root/'METHOD_FREEZE.json'),
        independent_ledger=str(ledger) if ledger.exists() else None, reporter_sha256=sha(Path(__file__)),
        limitations=['No pending or failed row is counted as successful cleaning.',
                     'Image-space floating components can increase as a large cloud fragments; background mass improvement is not complete black-dot removal.',
                     'Direct reviews cover the stated automatically selected views; they are not semantic ground truth or human botanical certification.',
                     'GPU sampling is a sampled per-process lower bound; torch allocator peaks and separate training cost remain distinct.',
                     'Only fixed independent-test results support heldout generalization; this reporter never tunes or selects the method.']))
    print(json.dumps(dict(all_complete=all_complete, groups=groups), ensure_ascii=False), flush=True)
    if require_complete and not all_complete:
        raise SystemExit('Requested native/paired/visual evidence is not complete; report preserves pending and failed rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=['all', 'development', 'independent_test'], default='all')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    summarize(args.manifest, args.batch_root.resolve(), args.output.resolve(), args.group, args.require_complete)


if __name__ == '__main__':
    main()
