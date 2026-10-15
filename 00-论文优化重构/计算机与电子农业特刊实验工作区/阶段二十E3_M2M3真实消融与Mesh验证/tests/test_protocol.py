import ast, copy, sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *

def function(path,name):
    return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)

class DisabledPath(ast.NodeTransformer):
    """Delete only statements provably inactive under the frozen M0 switches."""
    def visit_Assign(self,node):
        if any(isinstance(x,ast.Name) and x.id in ['capacity_state','pruning_report'] for x in node.targets):return None
        return self.generic_visit(node)
    def visit_If(self,node):
        names={x.id for x in ast.walk(node.test) if isinstance(x,ast.Name)}
        if names&{'capacity_state','pruning_report'}:return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='update_capacity_state':return None
        return self.generic_visit(node)

class ProtocolTest(unittest.TestCase):
    def test_disabled_training_ast_identical(self):
        left=DisabledPath().visit(function(GS/'train.py','training'))
        right=DisabledPath().visit(function(BRANCH/'train.py','training'))
        self.assertEqual(ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False))
    def test_native_seed_path_identical(self):
        self.assertEqual(ast.dump(function(GS/'utils/general_utils.py','safe_state'),include_attributes=False),
                         ast.dump(function(BRANCH/'utils/general_utils.py','safe_state'),include_attributes=False))
    def test_frozen_pair_only_capacity_mode_differs(self):
        for scene in SCENES:
            a=read_json(STAGE/f'configs/{scene}_M1_TRAIN.json');b=read_json(STAGE/f'configs/{scene}_M2_TRAIN.json')
            for c in [a,b]:
                c.pop('e3');c['model'].pop('model_path');c['all_parameters'].pop('model_path')
                for section in ['optimization','all_parameters']:c[section].pop('capacity_control_mode')
            self.assertEqual(a,b)
    def test_floor_and_budget_native_functions(self):
        import torch
        names=['resolve_capacity_floor_count','select_top_prune_candidates','capacity_floor_is_active']
        body=[function(BRANCH/'train.py',x) for x in names]
        ns={'torch':torch};exec(compile(ast.Module(body=body,type_ignores=[]),'<native helpers>','exec'),ns)
        opt=SimpleNamespace(capacity_floor_count=0,capacity_floor_reference='max_seen',capacity_floor_ratio=.4,
                            capacity_floor_start_iter=18000,capacity_floor_end_iter=0,pruning_start_iter=18000,iterations=30000)
        floor=ns['resolve_capacity_floor_count'](opt,{'max_seen_count':1000},500)
        self.assertEqual(floor,400)
        candidates=torch.ones(500,dtype=torch.bool);score=torch.arange(500,dtype=torch.float32)
        budget=ns['select_top_prune_candidates'](candidates,score,150)
        actual=ns['select_top_prune_candidates'](budget,score,500-floor)
        self.assertEqual(int(actual.sum()),100);self.assertEqual(int(budget.sum()-actual.sum()),50)
        self.assertFalse(ns['capacity_floor_is_active'](opt,17999));self.assertTrue(ns['capacity_floor_is_active'](opt,18000))
    def test_mesh_known_open_square(self):
        import open3d as o3d,numpy as np
        from mesh_metrics import structure
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)),
                                     o3d.utility.Vector3iVector(np.array([[0,1,2],[0,2,3]])))
        result,_=structure(mesh)
        self.assertEqual(result['boundary_edge_count'],4);self.assertEqual(result['connected_components'],1)
        self.assertEqual(result['non_manifold_edge_count'],0);self.assertEqual(result['surface_area_COLMAP_units_squared'],1.)
    def test_foreground_normalization(self):
        import torch
        core=metric_module();roi=torch.tensor([[True,False],[False,False]])
        rgb=torch.ones(3,2,2);ref=torch.zeros_like(rgb);alpha=torch.ones(2,2)
        fake_ssim=lambda a,b:torch.tensor(0.);fake_lpips=lambda *a,**kw:torch.tensor(0.)
        result=core.foreground_metrics(rgb,ref,alpha,roi,fake_ssim,fake_lpips)
        self.assertEqual(result['foreground_MSE'],1.);self.assertEqual(result['foreground_channel_samples'],3)
    def test_raycast_camera_axis_depth(self):
        import open3d as o3d,numpy as np,torch,math
        from mesh_metrics import raycast
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array([[-10,-10,3],[10,-10,3],[10,10,3],[-10,10,3]],float)),
                                     o3d.utility.Vector3iVector(np.array([[0,1,2],[0,2,3]])))
        camera=SimpleNamespace(image_width=4,image_height=4,FoVx=math.pi/2,FoVy=math.pi/2,world_view_transform=torch.eye(4))
        depth,_=raycast(mesh,camera)
        self.assertTrue(np.allclose(depth,3.,atol=1e-6))

if __name__=='__main__':unittest.main()
