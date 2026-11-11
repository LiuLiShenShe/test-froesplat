"""Hard quality constraints against the initial model prevent cumulative transparency."""


def quality_reasons(candidate, reference, config):
    q = config['quality']
    reasons = []
    a, b = candidate['mean'], reference['mean']
    for key, tolerance in [('fg_alpha', q['foreground_alpha_drop']), ('boundary_f1', q['boundary_f1_drop']),
        ('thin_coverage', q['thin_coverage_drop']), ('boundary_coverage', q['boundary_coverage_drop']),
        ('fg_coverage', q['foreground_coverage_drop']), ('skeleton_coverage', q['skeleton_coverage_drop'])]:
        if a[key] < b[key]-tolerance:
            reasons.append(f'{key}: {a[key]:.6g} < {b[key]-tolerance:.6g}')
    if a['fg_mse'] > b['fg_mse']*(1+q['foreground_mse_relative_increase'])+q['foreground_mse_absolute_increase']:
        reasons.append('foreground MSE regression')
    for key in ['depth_relative_p95', 'normal_angle_p95_deg']:
        if a[key] > q[key]:
            reasons.append(key)
    if a['new_depth_invalid_fraction'] > q['foreground_alpha_drop']:
        reasons.append('new depth holes')
    if a['depth_changed_fraction'] > q['depth_changed_fraction']:
        reasons.append('depth_changed_fraction')
    if a['fg_ssim'] < b['fg_ssim']-.001:
        reasons.append('foreground SSIM regression')
    if a.get('fg_masked_lpips') is not None and a['fg_masked_lpips'] > b['fg_masked_lpips']+.001:
        reasons.append('foreground LPIPS regression')
    for av, bv in zip(candidate['views'], reference['views']):
        if av['name'] != bv['name']:
            raise ValueError('Metric view mismatch')
        local = []
        for key, tol in [('fg_alpha', q['view_foreground_alpha_drop']), ('thin_coverage', q['view_thin_coverage_drop']),
            ('boundary_coverage', q['view_thin_coverage_drop']), ('fg_coverage', q['view_foreground_alpha_drop']),
            ('skeleton_coverage', q['view_thin_coverage_drop'])]:
            if av[key] < bv[key]-tol:
                local.append(key)
        if av['fg_mse'] > bv['fg_mse']*(1+q['view_foreground_mse_relative_increase'])+q['foreground_mse_absolute_increase']:
            local.append('fg_mse')
        for key in ['depth_relative_p95', 'normal_angle_p95_deg']:
            if av[key] > q['view_'+key]:
                local.append(key)
        if av['new_depth_invalid_fraction'] > q['view_foreground_alpha_drop']:
            local.append('depth_holes')
        if av['depth_changed_fraction'] > q['view_depth_changed_fraction']:
            local.append('depth_changed_fraction')
        if bv.get('dark_foreground_pixels', 0) > 20:
            if av['dark_foreground_alpha'] < bv['dark_foreground_alpha']-q['view_foreground_alpha_drop']:
                local.append('dark_leaf_alpha')
            if av['dark_foreground_mse'] > bv['dark_foreground_mse']*(1+q['view_foreground_mse_relative_increase'])+q['foreground_mse_absolute_increase']:
                local.append('dark_leaf_mse')
        if local:
            reasons.append(av['name']+':'+','.join(local))
    return reasons


def select(candidate, current, initial, config):
    reasons = quality_reasons(candidate, initial, config)
    # Local candidates are also constrained to current photometry/coverage; depth uses fixed initial data.
    reasons += [r for r in quality_reasons(candidate, current, config) if r not in reasons]
    before = current['mean']['bg_alpha_mass_per_fg']
    after = candidate['mean']['bg_alpha_mass_per_fg']
    gain = (before-after)/max(before, 1e-12)
    if gain < config['minimum_relative_gain']:
        reasons.append('background improvement plateau')
    return not reasons, gain, reasons
