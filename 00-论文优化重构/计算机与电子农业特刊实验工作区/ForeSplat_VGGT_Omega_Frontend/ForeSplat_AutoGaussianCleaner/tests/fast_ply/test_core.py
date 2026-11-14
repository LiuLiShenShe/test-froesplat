import hashlib
import json
from dataclasses import replace
import numpy as np
import pytest
from plyfile import PlyData,PlyElement
from fast_ply import CleanupConfig,cleanup_ply,extract_features,decide_keep
from fast_ply.io import Meter,read_header


def rows(n=128):
    names=[('x','f4'),('y','f4'),('z','f4'),('opacity','f4'),('filter_3D','f4'),
           *[(f'scale_{i}','f4') for i in range(3)],*[(f'rot_{i}','f4') for i in range(4)],
           *[(f'f_dc_{i}','f4') for i in range(3)],('payload','i4')]
    a=np.zeros(n,dtype=names);rng=np.random.default_rng(3)
    for k in ['x','y','z']: a[k]=rng.normal(0,.02,n)
    a['opacity']=2;a['filter_3D']=.001
    for k in range(3): a[f'scale_{k}']=-5;a[f'f_dc_{k}']=.5
    a['rot_0']=1;a['payload']=np.arange(n)
    a[-1]['opacity']=-20
    for k in range(3):a[-1][f'f_dc_{k}']=-3
    a[-1]['x']=a[-1]['y']=a[-1]['z']=100
    return a


def test_formula_and_native_record_bytes(tmp_path):
    a=rows();src=tmp_path/'source.ply';out=tmp_path/'clean.ply';report=tmp_path/'report.json'
    PlyData([PlyElement.describe(a,'vertex')],text=False,byte_order='>',comments=['native test']).write(src)
    before=hashlib.sha256(src.read_bytes()).digest()
    f=extract_features(a);expected=np.hypot(np.exp(-5),.001)
    assert np.allclose(f['effective_max'],expected)
    assert np.isclose(f['alpha_eff'][0],1/(1+np.exp(-2))*(np.exp(-5)/expected)**3)
    r=cleanup_ply(src,out,report,CleanupConfig(io_limit_mib=0,block_rows=17))
    assert r['io']['complete_model_reads']==1
    assert r['io']['logical_read_bytes']==src.stat().st_size
    assert r['io']['logical_write_bytes']==out.stat().st_size
    cleaned=PlyData.read(out)['vertex'].data
    assert len(cleaned)>0 and 127 not in cleaned['payload']
    original=PlyData.read(src)['vertex'].data
    assert cleaned.tobytes()==original[cleaned['payload']].tobytes()
    assert hashlib.sha256(src.read_bytes()).digest()==before
    assert PlyData.read(out).comments==['native test']
    assert json.loads(report.read_text())['resource']['gpu_used'] is False


def test_compact_path_matches_full_path(tmp_path):
    a=rows(20000);src=tmp_path/'source.ply'
    # Additional unknown native properties increase the row width without
    # changing the decision, making the budget select the compact path.
    wide=np.zeros(len(a),dtype=a.dtype.descr+[(f'extra_{i}','f4') for i in range(100)])
    for k in a.dtype.names: wide[k]=a[k]
    PlyData([PlyElement.describe(wide,'vertex')],text=False).write(src)
    cfg=CleanupConfig(io_limit_mib=0,block_rows=128)
    full=cleanup_ply(src,tmp_path/'full.ply',config=cfg)
    compact=cleanup_ply(src,tmp_path/'compact.ply',config=replace(cfg,memory_budget_gib=.105))
    assert compact['execution_path']=='two_sequential_reads_compact_features'
    assert compact['io']['logical_read_bytes']==2*src.stat().st_size
    assert full['decision']==compact['decision']
    assert (tmp_path/'full.ply').read_bytes()==(tmp_path/'compact.ply').read_bytes()


def test_supported_dark_thin_sheet_and_stem_are_kept():
    a=rows(1000)
    # A dark sheet and a line have no minimum thickness/density or anisotropy
    # deletion rule. There is no relative appearance or scale discontinuity.
    a['x']=np.linspace(-.1,.1,len(a));a['y']=0;a['z']=0
    a['opacity']=5
    for k in range(3):a[f'f_dc_{k}']=-1.7
    a['scale_0']=-4;a['scale_1']=-8;a['scale_2']=-8
    keep,stats=decide_keep(extract_features(a))
    assert keep.all()


def test_rate_limiter_counts_and_bounds_transfers(tmp_path,monkeypatch):
    import fast_ply.io as io
    now=[0.0]
    monkeypatch.setattr(io.time,'perf_counter',lambda:now[0])
    monkeypatch.setattr(io.time,'sleep',lambda x:now.__setitem__(0,now[0]+x))
    p=tmp_path/'data';data=np.arange(200000,dtype='u4')
    meter=Meter(1,65536)
    with p.open('wb',buffering=0) as f:meter.write(f,data)
    other=np.empty_like(data)
    with p.open('rb',buffering=0) as f:meter.readinto(f,other)
    assert np.array_equal(data,other)
    assert meter.read_bytes==meter.write_bytes==data.nbytes
    assert meter.wait_s==pytest.approx(2*data.nbytes/2**20)
    assert meter.max_transfer_bytes<=65536
    assert meter.peak_window_bytes<=2**20+65536


def test_refuse_overwrite_schema_and_bad_budget(tmp_path):
    src=tmp_path/'source.ply';out=tmp_path/'out.ply'
    PlyData([PlyElement.describe(rows(),'vertex')],text=False).write(src)
    with pytest.raises(ValueError):cleanup_ply(src,src)
    with pytest.raises(ValueError):cleanup_ply(src,out,src)
    with pytest.raises(MemoryError):cleanup_ply(src,out,config=CleanupConfig(memory_budget_gib=.001))
    assert not out.exists() and not out.with_name(out.name+'.partial').exists()
    bad=rows();bad['filter_3D']=-1
    with pytest.raises(ValueError):extract_features(bad)


def test_separate_viewer_export_matches_native_effective_quantities(tmp_path):
    from fast_ply.export_viewer import export
    src=tmp_path/'native.ply';out=tmp_path/'viewer.ply';a=rows()
    PlyData([PlyElement.describe(a,'vertex')],text=False).write(src)
    before=extract_features(a);export(src,out,0)
    after_rows=PlyData.read(out)['vertex'].data;after=extract_features(after_rows)
    assert np.allclose(before['effective_max'],after['effective_max'],rtol=2e-6)
    assert np.allclose(before['alpha_eff'],after['alpha_eff'],rtol=2e-6)
    assert np.all(after_rows['filter_3D']==0)
    assert np.array_equal(after_rows['payload'],a['payload'])


def test_header_whitespace_and_truncated_payload(tmp_path):
    src=tmp_path/'source.ply';a=rows()
    PlyData([PlyElement.describe(a,'vertex')],text=False).write(src)
    src.write_bytes(src.read_bytes().replace(b'element vertex 128',b'element  vertex\t128'))
    out=tmp_path/'out.ply';cleanup_ply(src,out,config=CleanupConfig(io_limit_mib=0))
    assert len(PlyData.read(out)['vertex'])<128
    src.write_bytes(src.read_bytes()[:-2])
    with pytest.raises(ValueError):cleanup_ply(src,tmp_path/'bad.ply',config=CleanupConfig(io_limit_mib=0))
    assert not (tmp_path/'bad.ply').exists()


def test_relative_geometry_decisions_survive_world_unit_change():
    a=rows(1000);original,_=decide_keep(extract_features(a))
    scaled=a.copy();factor=100
    for k in ['x','y','z']:scaled[k]*=factor
    scaled['filter_3D']*=factor
    for i in range(3):scaled[f'scale_{i}']+=np.log(factor)
    converted,_=decide_keep(extract_features(scaled))
    assert np.array_equal(original,converted)
