"""The shared exporter must preserve both historical decisions and native bytes."""
from dataclasses import replace
import numpy as np
from plyfile import PlyData,PlyElement
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features,rules
from ForeSplat_AutoGaussianCleaner.fast_ply.c1_c2 import cleanup_pair,C2Config
from test_c2 import fixture_rows

def test_shared_pair_and_compact_native_parity(tmp_path):
    a=fixture_rows();src=tmp_path/'input.ply'
    PlyData([PlyElement.describe(a,'vertex')],text=False,byte_order='>').write(src)
    raw=PlyData.read(src)['vertex'].data;f=features(raw)
    cfg=C2Config(io_limit_mib=0,block_rows=64)
    compact=len(a)*272+64*a.dtype.itemsize*6+192*2**20
    low=replace(cfg,memory_budget_gib=(compact+len(a)*a.dtype.itemsize/2)/2**30)
    for name,c in [('full',cfg),('compact',low)]:
        result=cleanup_pair(src,tmp_path/name,c,progress=lambda _:None)
        assert result['io']['complete_source_model_reads']==(1 if name=='full' else 2)
        assert result['io']['complete_model_writes']==2
        for method,version in [('C1_raw',2),('C2_raw',3)]:
            expected=raw[~rules(f,version=version,normalized=False)[0]]
            actual=PlyData.read(tmp_path/name/method/'cleaned_gaussians.ply')['vertex'].data
            assert actual.tobytes()==expected.tobytes()
    for method in ['C1_raw','C2_raw']:
        assert (tmp_path/'full'/method/'cleaned_gaussians.ply').read_bytes()==(tmp_path/'compact'/method/'cleaned_gaussians.ply').read_bytes()
