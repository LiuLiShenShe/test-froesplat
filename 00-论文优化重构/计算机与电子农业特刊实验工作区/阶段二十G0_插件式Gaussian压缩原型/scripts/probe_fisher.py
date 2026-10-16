"""Tiny exact 2DGS spatial Jacobian probe, NOT a registered compression scorer.

No model is trained or changed. Each RGB scalar has its own backward pass;
outer products are taken before observation reduction. This is a correctness
oracle for a future CUDA port and does not produce real-scene PUP scores.
"""
from common import *
from argparse import Namespace
import inspect
import time
import traceback


def main():
    assert_gpu();p=plan();import_paths()
    import numpy as np
    import torch
    from torch import nn
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.sh_utils import RGB2SH
    from diff_surfel_rasterization import GaussianRasterizer,GaussianRasterizationSettings
    torch.set_num_threads(4)
    output=DATA/'fisher_tiny_probe';output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','started_at':now(),'is_PUP_2DGS_ADAPTED':False,
            'purpose':'small exact per-output Jacobian correctness oracle, no scene-scale scoring or pruning',
            'GPU_uuid':GPU_UUID,'parameters':['x','y','z','log_scale_0','log_scale_1'],'rotation_scored':False}
    started=time.perf_counter()
    try:
        unsupported=[]
        for kw in ['fishers','f_count']:
            try:inspect.signature(GaussianRasterizer.forward).bind_partial(None,**{kw:None})
            except TypeError as e:unsupported.append({'keyword':kw,'error':str(e)})
            else:raise AssertionError('Native ABI unexpectedly accepts official extension keyword')
        result.update(native_forward_signature=str(inspect.signature(GaussianRasterizer.forward)),
                      native_settings_fields=list(GaussianRasterizationSettings._fields),unsupported_upstream_keywords=unsupported)
        camera=Camera(0,np.eye(3),np.zeros(3),1.0,1.0,torch.zeros(3,16,16),None,'synthetic_oracle',0,data_device='cpu')
        g=GaussianModel(0)
        g._xyz=nn.Parameter(torch.tensor([[-.14,-.05,2.0],[.16,.1,2.25],[.0,-.2,2.5]],device='cuda'))
        g._scaling=nn.Parameter(torch.log(torch.tensor([[.24,.18],[.2,.27],[.15,.2]],device='cuda')))
        g._rotation=nn.Parameter(torch.tensor([[1.,0,0,0]]*3,device='cuda'))
        g._opacity=nn.Parameter(torch.logit(torch.tensor([[.7],[.65],[.6]],device='cuda')))
        colors=torch.tensor([[.8,.2,.1],[.1,.75,.25],[.15,.2,.85]],device='cuda')
        g._features_dc=nn.Parameter(RGB2SH(colors)[:,None,:])
        g._features_rest=nn.Parameter(torch.zeros(3,0,3,device='cuda'))
        g.active_sh_degree=0
        bg=torch.zeros(3,device='cuda');pipe=Namespace(**p['pipeline'])
        package=render(camera,g,pipe,bg);image=package['render'];flat=image.reshape(-1)
        assert package['rend_alpha'].max()>.2
        jac=[];tick=time.perf_counter()
        for i in range(flat.numel()):
            xyz,scale=torch.autograd.grad(flat[i],(g._xyz,g._scaling),retain_graph=True)
            jac.append(torch.cat((xyz,scale),dim=1).detach())
        torch.cuda.synchronize();oracle_seconds=time.perf_counter()-tick
        jac=torch.stack(jac).double();assert torch.isfinite(jac).all() and jac.abs().max()>1e-3
        fisher=torch.einsum('pni,pnj->nij',jac,jac)
        eig=torch.linalg.eigvalsh(fisher)
        assert eig.min()>-1e-7 and torch.allclose(fisher,fisher.transpose(-1,-2))
        summed=jac.sum(0);incorrect=torch.einsum('ni,nj->nij',summed,summed)
        difference=torch.linalg.vector_norm(fisher-incorrect)/torch.linalg.vector_norm(fisher)
        assert difference>.01
        # Check every spatial parameter with a centered finite difference.
        comparisons=[]
        with torch.no_grad():
            for n in range(3):
                for param in range(5):
                    tensor=g._xyz if param<3 else g._scaling;col=param if param<3 else param-3
                    original=tensor[n,col].item();eps=0.0005
                    tensor[n,col]=original+eps;plus=render(camera,g,pipe,bg)['render'].reshape(-1).detach().clone()
                    tensor[n,col]=original-eps;minus=render(camera,g,pipe,bg)['render'].reshape(-1).detach().clone()
                    tensor[n,col]=original
                    numeric=((plus-minus)/(2*eps)).double();analytic=jac[:,n,param]
                    # Color values at support thresholds are piecewise differentiable;
                    # compare nontrivial analytic entries, reporting the full error too.
                    mask=analytic.abs()>.05*analytic.abs().max()
                    rel=(numeric[mask]-analytic[mask]).abs().max()/analytic[mask].abs().max()
                    comparisons.append({'gaussian':n,'parameter':result['parameters'][param],
                        'max_error_over_max_analytic_on_nontrivial_entries':rel.item(),
                        'full_max_abs_error':(numeric-analytic).abs().max().item()})
            assert max(x['max_error_over_max_analytic_on_nontrivial_entries'] for x in comparisons)<.03
        torch.save({'jacobian':jac.cpu(),'fisher':fisher.cpu(),'summed_gradient_outer_product':incorrect.cpu()},output/'oracle.pt')
        result.update(status='PASS',completed_at=now(),width=16,height=16,Gaussians=3,RGB_observations=flat.numel(),
            backward_calls=flat.numel(),oracle_seconds=oracle_seconds,wall_seconds=time.perf_counter()-started,
            fisher_shape=list(fisher.shape),fisher_eigenvalues=eig.cpu().tolist(),
            square_after_sum_relative_error=difference.item(),finite_difference_checks=comparisons,
            evidence_path=str(output/'oracle.pt'),evidence_sha256=sha(output/'oracle.pt'),
            scene_scale_status='PUP_ADAPTATION_BLOCKED: no efficient native per-pixel 5D/15-entry CUDA accumulator or complete scene score',
            active_extension_modified=False,production_training_modified=False)
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        write_json(output/'RUN.json',result);write_json(STAGE/'reports/FISHER_TINY_PROBE.json',result)
    print('FISHER_TINY_PROBE',result['status'],oracle_seconds,difference.item(),flush=True)

if __name__=='__main__':main()
