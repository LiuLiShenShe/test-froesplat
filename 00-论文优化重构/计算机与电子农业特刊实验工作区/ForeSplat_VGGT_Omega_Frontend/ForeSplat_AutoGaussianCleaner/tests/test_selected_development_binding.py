import pytest
from ForeSplat_AutoGaussianCleaner.verify_development_decision import require_recorded_identity


def test_a_different_certified_native_model_cannot_replace_the_reviewed_model():
    record=dict(input_sha256='same-original',native_sha256='reviewed-model')
    marker=dict(input_sha256='same-original',native_sha256='replacement-model',config={'version':'same-source'})
    report=dict(config=marker['config'])
    with pytest.raises(ValueError):require_recorded_identity(record,marker,report)


def test_changed_report_configuration_is_rejected_even_with_same_native_model():
    record=dict(input_sha256='original',native_sha256='model')
    marker=dict(**record,config={'version':'frozen'})
    with pytest.raises(ValueError):require_recorded_identity(record,marker,dict(config={'version':'other'}))
    require_recorded_identity(record,marker,dict(config=marker['config']))
