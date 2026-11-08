"""CPU counterexample using the exact current official GGGS selector AST.

This does not load a model, sample new geometry, or reproduce a historical
training state. CUDA device literals in the selector prefix become CPU; no
comparison, padding, quantile, or pruning expression is changed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import ast
import copy
import hashlib
import json
from pathlib import Path
import torch

AUDIT = Path(__file__).resolve().parents[1]
SOURCE = AUDIT.parents[2] / 'gggs_community/scene/gaussian_model.py'

class CPU(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 'cuda': return ast.copy_location(ast.Constant('cpu'), node)
        return node

def isolated_methods():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GaussianModel')
    methods = {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    namespace = {'torch':torch}
    exact = copy.deepcopy(methods['densify_and_prune'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[exact],type_ignores=[])),str(SOURCE),'exec'),namespace)
    selectors = {}
    for name, stop in [('densify_and_clone','new_xyz'),('densify_and_split','stds')]:
        fn = copy.deepcopy(methods[name]); body=[]
        for node in fn.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==stop for t in node.targets): break
            body.append(node)
        fn.body=body+[ast.Return(ast.Name(id='selected_pts_mask',ctx=ast.Load()))]
        fn.name=name+'_selector'
        fn=CPU().visit(fn)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),namespace)
        selectors[name]=namespace[fn.name]
    return namespace['densify_and_prune'],selectors

EXACT, SELECTORS = isolated_methods()

class Synthetic:
    percent_dense=.01
    def __init__(self, accum, absolute, denom, scales):
        self._xyz=torch.zeros((len(accum),3))
        self.get_scaling=torch.tensor(scales,dtype=torch.float32).reshape(-1,1).repeat(1,3)
        self.get_opacity=torch.full((len(accum),1),.1)
        self.xyz_gradient_accum=torch.tensor(accum).reshape(-1,1)
        self.xyz_gradient_accum_abs=torch.tensor(absolute).reshape(-1,1)
        self.denom=torch.tensor(denom).reshape(-1,1)
        self.records={}
    @property
    def get_xyz(self): return self._xyz
    def densify_and_clone(self,*args):
        mask=SELECTORS['densify_and_clone'](self,*args)
        self.records['clone_selected']=mask.tolist()
        # No sampling: only represent the cardinality of the exact predicate.
        k=int(mask.sum());self._xyz=torch.zeros((len(self._xyz)+k,3))
        self.get_scaling=torch.cat((self.get_scaling,self.get_scaling[mask]))
        self.get_opacity=torch.cat((self.get_opacity,self.get_opacity[mask]))
    def densify_and_split(self,*args):
        self.records['quantile_threshold']=float(args[3])
        mask=SELECTORS['densify_and_split'](self,*args)
        self.records['split_selected']=mask.tolist()
        k=int(mask.sum());self._xyz=torch.zeros((len(self._xyz)+k,3))
        self.get_opacity=torch.full((len(self._xyz),1),.1)
    def prune_points(self,mask):
        self.records['raw_opacity_pruned']=int(mask.sum())
        self._xyz=self._xyz[~mask]

def main():
    cases=[
        ('all_zero_including_unobserved', [0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,1.],[.001,.1,.001,.1]),
        ('one_nonzero_gradient_sanity',[0.,0.,0.,.001],[0.,0.,0.,.001],[0.,1.,1.,1.],[.001,.1,.001,.1]),
    ]
    results=[]
    for name,a,b,d,s in cases:
        model=Synthetic(a,b,d,s)
        cardinality=EXACT(model,.0002,.05,1.0,20)
        results.append(dict(case=name,accum=a,absolute_accum=b,denom=d,scale=s,records=model.records,
                            exact_source_return_counts=list(cardinality),final_synthetic_count=len(model._xyz)))
    output=dict(schema='gggs.zero_selection.counterexample.v1',source=str(SOURCE),
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),torch_version=torch.__version__,device='cpu',
                source_extraction='Full densify_and_prune; clone/split exact AST prefix until before sampling/allocation. Only selector device cuda→cpu.',
                interpretation='Q=0 and >=Q select zero absolute gradients, including denom=0 replaced NaNs. This is a synthetic boundary counterexample, NOT evidence that E5 reached Q=0.',
                historical_E5_trigger_confirmed=False,historical_accumulators_available=False,training_executed=False,
                stochastic_operations_executed=False,results=results)
    path=AUDIT/'gaussian_analysis/zero_selection_counterexample.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
