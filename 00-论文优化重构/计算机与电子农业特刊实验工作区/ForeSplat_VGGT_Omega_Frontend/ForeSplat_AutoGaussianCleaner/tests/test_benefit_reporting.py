"""The audit must preserve the benefit guard's independent training/free branches."""
from ForeSplat_AutoGaussianCleaner.analyze_adaptive import group_benefit_diagnostic


def metrics(global_mass,local_mass,free_mass,free_local):
    return dict(mean=dict(bg_alpha_mass_per_fg=global_mass),
        views=[dict(bg_alpha_mass_per_fg=local_mass)],
        free_background=dict(alpha_mass=free_mass,views=[dict(alpha_mass=free_local)]))


CONFIG=dict(group_minimum_global_gain=.01,group_minimum_view_gain=.1)
CURRENT=metrics(1,1,1,1)


def test_training_and_free_maxima_cannot_pass_by_mixing_branches():
    result=group_benefit_diagnostic(metrics(.98,.95,.995,.8),CURRENT,CONFIG)
    assert not result['benefit_training_branch_pass']
    assert not result['benefit_free_branch_pass']
    assert abs(result['benefit_rejection_severity']-.5)<1e-12


def test_free_benefit_cannot_hide_training_background_regression():
    result=group_benefit_diagnostic(metrics(1.000001,1,.98,.8),CURRENT,CONFIG)
    assert not result['benefit_free_branch_pass']
    assert result['benefit_rejection_severity']>0


def test_local_floor_rejection_and_safe_free_branch_are_distinguished():
    rejected=group_benefit_diagnostic(metrics(.98,.95,.98,.95),CURRENT,CONFIG)
    accepted=group_benefit_diagnostic(metrics(1,1,.98,.8),CURRENT,CONFIG)
    assert abs(rejected['benefit_rejection_severity']-.5)<1e-12
    assert accepted['benefit_free_branch_pass']
    assert accepted['benefit_rejection_severity']==0
