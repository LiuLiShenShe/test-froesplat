"""Regression checks for historical identity, byte fidelity and streaming parity."""
from dataclasses import replace
import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features, rules
from ForeSplat_AutoGaussianCleaner.fast_ply.c2 import (
    C2Config, allocate_features, fill_features, decide_c2, cleanup_c2,
)


def fixture_rows(n=1024):
    names = ['x','y','z','opacity','filter_3D'] + [f'scale_{i}' for i in range(3)]
    names += [f'rot_{i}' for i in range(4)] + [f'f_dc_{i}' for i in range(3)]
    a = np.zeros(n, dtype=[(k,'f4') for k in names]+[('native_id','i4')])
    rng = np.random.default_rng(124)
    for k in ['x','y','z']: a[k] = rng.normal(0, .1, n)
    for k in ['x','y','z']: a[k][:100] = 0  # More than 65 coincident centers.
    for i in range(3):
        a[f'scale_{i}'] = rng.uniform(-9, 0, n)
        a[f'f_dc_{i}'] = rng.uniform(-2.5, 1, n)
    a['opacity'] = rng.uniform(-8, 8, n)
    a['filter_3D'] = .002; a['rot_0'] = 1; a['native_id'] = np.arange(n)
    return a


@pytest.mark.parametrize('n', [1, 21, 22, 1024])
def test_exact_legacy_features_masks_bits_and_parameters(n):
    a = fixture_rows(n)
    old = features(a)
    new = allocate_features(n, a.dtype)
    for start in range(0, n, 17): fill_features(new, start, a[start:start+17])
    expected = rules(old, version=3, normalized=False)
    actual = decide_c2(new)
    for key in ['xyz','scale_max','raw_alpha','brightness','color_range','knn_original']:
        assert np.array_equal(old[key], new[key]), key
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])
    assert expected[2] == actual[2]


@pytest.mark.parametrize('byte_order', ['<', '>'])
def test_one_and_two_read_exact_native_bytes_and_no_overwrite(tmp_path, byte_order):
    a = fixture_rows(); source = tmp_path/'source.ply'
    PlyData([PlyElement.describe(a,'vertex')], text=False, byte_order=byte_order,
            comments=['preserve native unknown fields']).write(source)
    original_bytes = source.read_bytes()
    original = PlyData.read(source)['vertex'].data
    keep = ~rules(features(original), version=3, normalized=False)[0]
    cfg = C2Config(io_limit_mib=0, block_rows=64)
    compact = len(a)*256 + cfg.block_rows*a.dtype.itemsize*4 + 160*2**20
    low = replace(cfg, memory_budget_gib=(compact + len(a)*a.dtype.itemsize/2)/2**30)
    one = cleanup_c2(source, tmp_path/'one', cfg)
    two = cleanup_c2(source, tmp_path/'two', low)
    assert one['io']['complete_model_reads'] == 1
    assert two['io']['complete_model_reads'] == 2
    assert two['io']['logical_read_bytes'] == 2*source.stat().st_size
    assert one['parameters'] == two['parameters']
    result = tmp_path/'one/cleaned_gaussians.ply'
    assert result.read_bytes() == (tmp_path/'two/cleaned_gaussians.ply').read_bytes()
    assert PlyData.read(result)['vertex'].data.tobytes() == original[keep].tobytes()
    assert PlyData.read(result).comments == ['preserve native unknown fields']
    assert source.read_bytes() == original_bytes
    assert sorted(p.name for p in result.parent.iterdir()) == ['cleaned_gaussians.ply','cleanup_report.json']
    with pytest.raises(FileExistsError): cleanup_c2(source, tmp_path/'one', cfg)
    with pytest.raises(MemoryError): cleanup_c2(source, tmp_path/'insufficient', replace(cfg, memory_budget_gib=.001))
