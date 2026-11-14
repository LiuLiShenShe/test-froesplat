import pytest
from ForeSplat_AutoGaussianCleaner.prepare_development_decision import require_reviewed_development,require_matched_main_identity,require_frozen_manifest
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha


def rows():
    return [dict(plant=p,status='complete',direct_visual_review_status='actual_direct_review_recorded',
        paired_cloud_evaluation_status='verified_complete',conspicuous_leaf_loss_in_reviewed_views='False',
        modified='True') for p in ['plant1','plant2']]


SUMMARY=dict(all_requested_models_evaluated_and_reviewed=True)
EXPECTED={'plant1','plant2'}


def test_a_pending_model_cannot_be_replaced_by_a_complete_model_twice():
    data=rows();data[1]['plant']='plant1'
    with pytest.raises(ValueError):require_reviewed_development(SUMMARY,data,EXPECTED)


def test_observed_leaf_loss_blocks_production_decision():
    data=rows();data[1]['conspicuous_leaf_loss_in_reviewed_views']='True'
    with pytest.raises(ValueError):require_reviewed_development(SUMMARY,data,EXPECTED)


def test_a_zero_modification_model_does_not_supply_multi_plant_success():
    data=rows();data[1]['modified']='False'
    with pytest.raises(ValueError):require_reviewed_development(SUMMARY,data,EXPECTED)
    require_reviewed_development(SUMMARY,rows(),EXPECTED)


def main_evidence(tmp_path):
    frozen=dict(sources={'cleaner/a.py':'source'},scene_contents={'KongQueZhuYu':{'RGB':'photo'}},
        native_renderer_sources={'renderer':'binary','other_repo':'unrelated'})
    report=dict(config=dict(implementation_sources={'cleaner/a.py':'source'},scene_input_fingerprint={'RGB':'photo'},
        native_renderer_fingerprint={'renderer':'binary'}))
    marker=dict(input_sha256='original',native_sha256='result')
    common=dict(output=str(tmp_path),**marker)
    return report,frozen,common,tmp_path,marker


@pytest.mark.parametrize('changed',['scene_input_fingerprint','native_renderer_fingerprint','implementation_sources'])
def test_same_paths_do_not_allow_changed_scene_renderer_or_missing_sources(tmp_path,changed):
    evidence=main_evidence(tmp_path);evidence[0]['config'][changed]={}
    with pytest.raises(ValueError):require_matched_main_identity(*evidence)


def test_stale_common_evaluation_marker_is_rejected(tmp_path):
    evidence=main_evidence(tmp_path)
    require_matched_main_identity(*evidence)
    evidence[2]['native_sha256']='old-model'
    with pytest.raises(ValueError):require_matched_main_identity(*evidence)


def test_the_frozen_development_set_cannot_be_reduced_to_completed_plants(tmp_path):
    path=tmp_path/'manifest.json';path.write_text('{"development":["ready1","ready2","pending"]}')
    frozen=dict(manifest_sha256=sha(path));require_frozen_manifest(path,frozen)
    path.write_text('{"development":["ready1","ready2"]}')
    with pytest.raises(ValueError):require_frozen_manifest(path,frozen)
