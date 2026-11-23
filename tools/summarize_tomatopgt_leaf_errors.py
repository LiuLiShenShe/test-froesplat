#!/usr/bin/env python3
"""Summarize TomatoPGT leaf discrepancies after the author's visual identity review."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


SUMMARY = Path(__file__).resolve().parents[1] / 'experiments/ForeSplat_TomatoPGT_Independent/summary'
TRAITS = (
    ('base_to_tip_chord', 'cm', 'length', 'model_chord_cm', 'official_chord_cm'),
    ('midsection_transverse_width', 'cm', 'width', 'model_transverse_width_cm', 'official_transverse_width_cm'),
    ('observed_one_sided_leaf_area', 'cm2', '', '', ''),
)


def source_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def disposition(row, landmark, model_field, ref_field):
    if not landmark:
        return 'no_comparable_one_sided_area_reference'
    if not row['candidate_id']:
        return 'no_matching_mesh'
    if row['status'] not in {'strong_suggestion', 'review_suggestion'}:
        return 'ambiguous_assignment'
    if row['metric_pair_qc'] != 'eligible_for_review':
        return 'scale_review'
    if not row[ref_field]:
        return 'missing_reference_landmarks'
    if not row[model_field]:
        return 'no_model_landmark_pair_passing_existing_qc'
    return 'included_reference_guided_landmarks'


def measure_subset(source):
    rows = []
    seen = set()
    for row in source:
        for trait, unit, landmark, model_field, ref_field in TRAITS:
            k = (row['scene'], row['official_point_instance'], trait)
            if k in seen:
                raise ValueError(f'Duplicate reference organ: {k}')
            seen.add(k)
            status = disposition(row, landmark, model_field, ref_field)
            model = ref = error = ''
            ends = ''
            if status == 'included_reference_guided_landmarks':
                ends = json.loads(row['endpoints_json'])[landmark]
                model = math.dist(*ends['prediction_cm'])
                ref = math.dist(*ends['reference_cm'])
                for raw, val in ((row[model_field], model), (row[ref_field], ref)):
                    if not math.isfinite(val) or val <= 0 or abs(float(raw) - val) > 0.000051:
                        raise ValueError(f'Measurement and endpoints disagree: {k}')
                if not Path(row['model_local_path']).is_file():
                    raise FileNotFoundError(row['model_local_path'])
                error = model - ref
            rows.append(dict(
                scene=row['scene'], cultivar=row['cultivar'], official_point_instance=row['official_point_instance'],
                official_graph_id=row['official_id'], candidate_id=row['candidate_id'], upstream_match_status=row['status'],
                identity_review_basis=('author_reported_visual_review_in_chat'
                                       if row['status'] in {'strong_suggestion', 'review_suggestion'} else ''),
                scale_qc=row['metric_pair_qc'],
                trait=trait, unit=unit, status=status, model=model, reference=ref, error_model_minus_reference=error,
                endpoints_json=json.dumps(ends, separators=(',', ':')) if ends else '', model_ply=row['model_local_path'],
            ))
    return rows


def summarize(rows):
    for trait, unit, *_ in TRAITS:
        cohort = [r for r in rows if r['trait'] == trait]
        eligible = [r for r in cohort if r['status'] == 'included_reference_guided_landmarks']
        d = [r['error_model_minus_reference'] for r in eligible]
        refs = [r['reference'] for r in eligible]
        se = math.fsum(x*x for x in d)
        ref_mean = mean(refs) if refs else 0
        ss = math.fsum((y-ref_mean)**2 for y in refs)
        yield dict(
            trait=trait, unit=unit, official_compounds_in_metric_scans=len(cohort), n_pairs=len(d),
            n_scans=len({r['scene'] for r in eligible}),
            mae=mean(map(abs, d)) if d else '', rmse=math.sqrt(se/len(d)) if d else '',
            mean_bias=mean(d) if d else '', r2_against_identity=1-se/ss if ss else '',
            mape_pct=100*mean(abs(x)/y for x, y in zip(d, refs)) if d else '',
            reference_min=min(refs) if refs else '', reference_max=max(refs) if refs else '',
            interpretation='reference_guided_agreement_selected_subset' if d else 'not_evaluated_no_comparable_reference',
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=SUMMARY)
    args = parser.parse_args()
    source = source_rows(args.summary / 'compound_auto_review/compound_match_suggestions.csv')
    metric = [r for r in source if r['metric_scan'] == 'True' and r['reference_scope'] == 'compound_leaf']
    comparisons = measure_subset(metric)
    errors = list(summarize(comparisons))
    write_rows(args.summary / 'final_leaf_trait_pairs.csv', comparisons)
    write_rows(args.summary / 'final_leaf_trait_errors.csv', errors)
    for row in errors:
        print(row)


if __name__ == '__main__':
    main()
