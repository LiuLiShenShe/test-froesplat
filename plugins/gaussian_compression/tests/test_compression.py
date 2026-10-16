"""Mathematical contracts and PLY preservation; no training data needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData,PlyElement
from plugins.gaussian_compression import compress_model
from plugins.gaussian_compression.io import read_model,sha256,write_subset
from plugins.gaussian_compression.pruning import rank_and_select,removal_budget
from plugins.gaussian_compression.scoring import AdaptationBlocked


def make_model(path,n=17,degree=2,text=False,byte_order='<'):
    fields=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]
    fields += [f'f_rest_{i}' for i in range(3*((degree+1)**2-1))]
    fields += ['opacity','scale_0','scale_1']+[f'rot_{i}' for i in range(4)]
    values=np.zeros(n,dtype=[(k,'f4') for k in fields]+[('extra_origin_id','u4')])
    rng=np.random.default_rng(13)
    for key in fields:
        values[key]=rng.normal(0,.2,n)
    values['rot_0']=1
    values['extra_origin_id']=np.arange(n)
    values['opacity']=np.linspace(-7,6,n)
    values['scale_0']=-2;values['scale_1']=-3
    PlyData([PlyElement.describe(values,'vertex',comments=['vertex metadata'])],
            text=text,byte_order=byte_order,comments=['frozen synthetic model'],obj_info=['identity']).write(str(path))
    return values


class RankingTests(unittest.TestCase):
    def test_decimal_budget_is_exact(self):
        for ratio,expected in [('0',0),('.2',25689),('.4',51378),('.6',77068)]:
            self.assertEqual(removal_budget(128447,ratio),expected)
        self.assertEqual(removal_budget(100,'0.29'),29)

    def test_monotonic_topk_and_ties(self):
        scores=np.array([2.,5.,5.,1.,5.])
        keep,remove=rank_and_select(scores,'.4')
        np.testing.assert_array_equal(keep,[1,2,4])
        np.testing.assert_array_equal(remove,[0,3])
        keep,_=rank_and_select(np.ones(7),'.5')
        np.testing.assert_array_equal(keep,[0,1,2,3])

    def test_capacity_partition_and_nested_ratios(self):
        scores=np.random.default_rng(9).normal(size=101)
        previous=set(range(101))
        for ratio in [0,.2,.4,.6,.9]:
            keep,remove=rank_and_select(scores,ratio)
            self.assertEqual(len(remove),removal_budget(101,ratio))
            self.assertEqual(set(keep)|set(remove),set(range(101)))
            self.assertFalse(set(keep)&set(remove))
            self.assertTrue(set(keep)<=previous)
            if len(remove):self.assertGreaterEqual(scores[keep].min(),scores[remove].max())
            previous=set(keep)

    def test_invalid_budget_or_scores(self):
        for ratio in [-.1,1,2,'NaN',float('inf'),True,'bad']:
            with self.subTest(ratio=ratio),self.assertRaises(ValueError):removal_budget(20,ratio)
        for scores in [[1,np.nan],[1,np.inf],[[1,2]],np.array([1j,2j]),[]]:
            with self.subTest(scores=str(scores)),self.assertRaises(ValueError):rank_and_select(scores,.2)

    def test_extreme_integer_scores_preserve_order(self):
        for scores in [np.array([2**63+1,2**63+2,2**63],dtype=np.uint64),
                       np.array([-2**63,2**63-1,2**63-2],dtype=np.int64)]:
            keep,_=rank_and_select(scores,'.67')
            np.testing.assert_array_equal(keep,[1])


class PLYTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.source=self.base/'input'/'point_cloud.ply';self.source.parent.mkdir()
        make_model(self.source)
        self.hash=sha256(self.source)

    def tearDown(self):self.temp.cleanup()

    def test_opacity_subset_every_property_and_index(self):
        result=compress_model(self.source,self.base/'out','opacity','.4',{'seed':19})
        keep=np.load(result.retained_indices_path);removed=np.load(result.removed_indices_path)
        original=read_model(self.source);pruned=read_model(result.model_path)
        np.testing.assert_array_equal(keep,np.arange(6,17))
        np.testing.assert_array_equal(removed,np.arange(6))
        np.testing.assert_array_equal(pruned.vertices,original.vertices[keep])
        self.assertEqual(pruned.schema,original.schema)
        self.assertEqual(pruned.sh_degree,2)
        self.assertEqual(sha256(self.source),self.hash)
        self.assertLess(result.model_path.stat().st_size,self.source.stat().st_size)
        metadata=json.loads(result.manifest_path.read_text())
        self.assertEqual(metadata['status'],'PASS')
        self.assertTrue(metadata['all_retained_values_identical'])
        self.assertEqual(metadata['removed_count'],6)

    def test_disabled_is_byte_identical_and_ignores_blocked_scorer(self):
        result=compress_model(self.source,self.base/'off','PUP_2DGS_ADAPTED',.6,{'enabled':False})
        self.assertEqual(self.source.read_bytes(),result.model_path.read_bytes())
        np.testing.assert_array_equal(np.load(result.retained_indices_path),np.arange(17))
        self.assertEqual(np.load(result.removed_indices_path).size,0)
        self.assertFalse((result.output_dir/'importance_scores.npy').exists())

    def test_zero_ratio_byte_identity_for_both_baselines(self):
        for method in ['random','opacity']:
            result=compress_model(self.source,self.base/method,method,0,{'seed':4})
            self.assertEqual(result.output_model_sha256,self.hash)

    def test_random_repeatable_and_seed_changes_selection(self):
        a=compress_model(self.source,self.base/'a','random',.6,{'seed':40})
        b=compress_model(self.source,self.base/'b','random',.6,{'seed':40})
        c=compress_model(self.source,self.base/'c','random',.6,{'seed':41})
        self.assertEqual(a.output_model_sha256,b.output_model_sha256)
        self.assertNotEqual(a.output_model_sha256,c.output_model_sha256)

    def test_external_scores_bind_to_source_and_file(self):
        path=self.base/'scores.npy';np.save(path,np.arange(17,dtype=np.float64))
        config={'scores_path':str(path),'score_provenance':{'scorer':'synthetic_test_only',
            'source_model_sha256':self.hash,'scores_sha256':sha256(path)}}
        result=compress_model(self.source,self.base/'external','external',.4,config)
        np.testing.assert_array_equal(np.load(result.retained_indices_path),np.arange(6,17))
        config['score_provenance']['source_model_sha256']='0'*64
        with self.assertRaises(ValueError):compress_model(self.source,self.base/'bad','external',.4,config)

    def test_output_safety_and_explicit_boolean(self):
        with self.assertRaises(ValueError):compress_model(self.source,self.source.parent/'out','opacity',.2)
        for config in [{'enabled':'false'},{'recovery':True},{'expected_source_sha256':'0'*64}]:
            with self.subTest(config=config),self.assertRaises(ValueError):
                compress_model(self.source,self.base/'bad', 'opacity',.2,config)
        compress_model(self.source,self.base/'exists','opacity',.2)
        with self.assertRaises(FileExistsError):compress_model(self.source,self.base/'exists','opacity',.2)

    def test_blocked_methods_leave_truthful_failure(self):
        for method in ['PUP_2DGS_ADAPTED','LIGHTGAUSSIAN_2DGS_ADAPTED']:
            out=self.base/method
            with self.assertRaises(AdaptationBlocked):compress_model(self.source,out,method,.2)
            self.assertEqual(json.loads((out/'compression_config.json').read_text())['status'],'FAILED')
            self.assertFalse((out/'pruned_model.ply').exists())
        self.assertEqual(sha256(self.source),self.hash)

    def test_directory_iteration_resolution(self):
        directory=self.base/'native';p=directory/'point_cloud/iteration_30000/point_cloud.ply';p.parent.mkdir(parents=True)
        p.write_bytes(self.source.read_bytes())
        result=compress_model(directory,self.base/'resolved','opacity',.2)
        self.assertEqual(result.input_count,17)
        with self.assertRaises(ValueError):compress_model(directory,directory/'child','opacity',.2)

    def test_ascii_big_endian_and_sh_zero(self):
        for idx,(text,order,degree) in enumerate([(True,'<',0),(False,'>',3)]):
            p=self.source.parent/f'special_{idx}.ply';make_model(p,13,degree,text,order)
            result=compress_model(p,self.base/f'special_{idx}_out','opacity',.4)
            self.assertEqual(read_model(result.model_path).sh_degree,degree)
            self.assertEqual(read_model(result.model_path).schema,read_model(p).schema)

    def test_nonfinite_zero_quaternion_and_overflow_rejected(self):
        for field,value in [('x',np.nan),('scale_0',1000),('scale_1',-1000)]:
            p=self.base/'invalid.ply';a=make_model(p);a[field][0]=value
            PlyData([PlyElement.describe(a,'vertex')]).write(str(p))
            with self.assertRaises(ValueError):read_model(p)
        a=make_model(p)
        for i in range(4):a[f'rot_{i}'][0]=0
        PlyData([PlyElement.describe(a,'vertex')]).write(str(p))
        with self.assertRaises(ValueError):read_model(p)

    def test_invalid_indices_and_3dgs_schema_rejected(self):
        model=read_model(self.source)
        for idx in [[1,1],[-1,2],[17],[1.5],[]]:
            with self.subTest(idx=idx),self.assertRaises(ValueError):write_subset(model,idx,self.base/'bad.ply')
        a=model.vertices
        extra=np.empty(17,dtype=list(a.dtype.descr)+[('scale_2','f4')])
        for key in a.dtype.names:extra[key]=a[key]
        extra['scale_2']=-2
        p=self.base/'3dgs.ply';PlyData([PlyElement.describe(extra,'vertex')]).write(str(p))
        with self.assertRaises(ValueError):read_model(p)


if __name__=='__main__':unittest.main()
