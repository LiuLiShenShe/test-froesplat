from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parents[2]
TABLES = STAGE / 'tables'
REPORTS = STAGE / 'reports'


def csv_rows(name: str):
    with (TABLES / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def test_required_outputs_exist():
    required = [
        'GT_DEFINITION_AUDIT.csv', 'SEGMENTATION_PER_FRAME_METRICS.csv',
        'SEGMENTATION_SOURCE_PROVENANCE.csv', 'TABLE3_RECALCULATED.csv',
        'TABLE4_RECALCULATED.csv', 'TABLE5_RECALCULATED.csv',
        'MASK_VERSION_COMPARISON.csv', 'MASK_MATCH_COVERAGE.csv',
        'RECONSTRUCTION_REUSE_DECISION.csv', 'PHENOTYPE_DATA_SOURCE_MATRIX.csv',
        'TABLE7_RECONSTRUCTION_SOURCE.csv', 'TABLE7_REUSABLE_RESULTS.csv',
        'TABLE7_CLAIM_STATUS.csv', 'M2M3_FORMULA_CODE_MATRIX.csv',
        'R1_FORMULA_RECHECK.csv', 'V0_V3_CONFIG_MATRIX.csv',
        'V0_V3_INPUT_MANIFEST.csv',
    ]
    for name in required:
        assert (TABLES / name).is_file(), name
    for name in [
        'GT_DEFINITION_AUDIT.md', 'SEGMENTATION_METRIC_SPEC.md',
        'TABLE_3_4_5_RECALCULATION_REPORT.md', 'MASK_INPUT_IDENTITY_AUDIT.md',
        'RECONSTRUCTION_REUSE_REPORT.md', 'SCALE_RECOVERY_AUDIT.md',
        'PHENOTYPE_PROVENANCE_AUDIT.md', 'TABLE7_PROVENANCE_AUDIT.md',
        'TABLE7_UNVERIFIED_CLAIMS.md', 'M2M3_IMPLEMENTATION_TRUTH.md',
        'M2M3_DECISION_OPTIONS.md', 'R1_182_CANDIDATE_RECHECK.md',
        'V0_V3_PREFLIGHT_REPORT.md', 'PHASE20C_EXECUTION_PLAN.md',
    ]:
        assert (REPORTS / name).is_file(), name
    assert (STAGE / 'README.md').is_file()
    assert (STAGE / 'PHASE20B_FINAL_REPORT.md').is_file()
    assert (REPORTS / 'HISTORICAL_EVIDENCE_ERRATA.md').is_file()


def test_gt_and_r1_counts():
    gt = csv_rows('GT_DEFINITION_AUDIT.csv')
    assert len(gt) == 31
    assert {row['status'] for row in gt} == {'VERIFIED'}
    r1 = csv_rows('R1_FORMULA_RECHECK.csv')
    assert len(r1) == 182
    assert {row['status'] for row in r1} == {'PASS'}


def test_recalculated_tables_have_unified_gt_and_aggregation():
    for name in ['TABLE3_RECALCULATED.csv', 'TABLE4_RECALCULATED.csv', 'TABLE5_RECALCULATED.csv']:
        rows = csv_rows(name)
        aggregate = [row for row in rows if row['sample_id'] == 'ALL' and row['aggregation'] == 'sample_equal_mean']
        assert aggregate, name
        assert {row['gt_definition'] for row in aggregate} == {'potted_clean'}
        assert {row['valid_frames'] for row in aggregate} == {'20'}
        assert {row['valid_samples'] for row in aggregate} == {'4'}


def test_mask_identity_and_reuse_are_conservative():
    coverage = csv_rows('MASK_MATCH_COVERAGE.csv')
    assert len(coverage) == 19
    assert sum(int(row['verified_pairs']) for row in coverage) == 43
    assert sum(int(row['exact_pair_count']) for row in coverage) == 0
    reuse = csv_rows('RECONSTRUCTION_REUSE_DECISION.csv')
    assert len(reuse) == 19
    assert not any(row['decision'] == 'REUSE_DIRECTLY' for row in reuse)


def test_table7_does_not_promote_paper_claims():
    claims = csv_rows('TABLE7_CLAIM_STATUS.csv')
    assert len(claims) == 4
    assert {row['status'] for row in claims} == {'UNVERIFIED'}
    pilot = [row for row in csv_rows('TABLE7_RECONSTRUCTION_SOURCE.csv') if row['artifact_type'] == 'M2M3_full_evaluation_row']
    assert len(pilot) == 18
    assert {row['status'] for row in pilot} == {'VERIFIED_AS_PILOT'}


def test_v0_v3_matrix_and_manifest():
    matrix = csv_rows('V0_V3_CONFIG_MATRIX.csv')
    assert len(matrix) == 20
    assert {(row['ranking_policy'], row['refinement']) for row in matrix} == {('base', 'OFF'), ('r1', 'OFF'), ('base', 'ON'), ('r1', 'ON')}
    manifest = csv_rows('V0_V3_INPUT_MANIFEST.csv')
    assert len(manifest) == 20
    assert {row['status'] for row in manifest} == {'READY_FOR_PHASE20C'}
    assert all(row['formal_candidate_reuse'] == 'YES' for row in manifest)


def test_configs_hold_non_factor_parameters_constant():
    configs = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((REPORTS / 'configs').glob('V*_*.json'))]
    assert len(configs) == 20
    ignored = {
        'group', 'ranking_policy', 'refinement', 'output_dir', 'sample_id',
        'input_dir', 'candidate_source_dir', 'colmap_dir', 'config_provenance',
        'use_spnp_refinement', 'spnp_backend', 'use_residual_repair',
        'thin_repair_mode', 'use_corrective_geometry', 'corrective_geometry_backend',
    }
    signatures = []
    for config in configs:
        signatures.append(json.dumps({key: value for key, value in config.items() if key not in ignored}, sort_keys=True, ensure_ascii=False))
    assert len(set(signatures)) == 1


def test_no_production_files_changed():
    result = subprocess.run(['git', 'status', '--short'], cwd=ROOT, check=True, text=True, capture_output=True)
    changed = []
    for line in result.stdout.splitlines():
        path = line[3:] if len(line) >= 4 else ''
        if path and not path.startswith(str(STAGE.relative_to(ROOT))):
            changed.append(path)
    assert changed == []
