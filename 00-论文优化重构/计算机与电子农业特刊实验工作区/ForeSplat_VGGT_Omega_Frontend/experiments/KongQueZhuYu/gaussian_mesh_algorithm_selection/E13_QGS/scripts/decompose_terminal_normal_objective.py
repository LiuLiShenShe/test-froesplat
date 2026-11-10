"""Reconstruct the native normal scalar on actual terminal maps, CPU only."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def main():
    training=json.loads((ROOT/'training_complete.json').read_text());assert training['iterations']==30000
    params=json.loads((ROOT/'actual_training_parameters.json').read_text());opt=params['config']['optimizer']
    assert opt['wo_image_weight'] is True
    index_path=ROOT/'surface_inputs/render_index.json';index=json.loads(index_path.read_text());assert len(index)==208
    output=ROOT/'native_normal_objective_decomposition';output.mkdir(exist_ok=True)
    rows=[]
    for item in index:
        with np.load(item['path']) as archive:
            normal=torch.from_numpy(archive['native_normal'].copy())
            depth_normal=torch.from_numpy(archive['depth_normal_camera'].copy())
            curvature=torch.from_numpy(archive['curvature'].copy())
            mask=archive['foreground_mask'].copy().astype(bool)
        assert normal.shape==depth_normal.shape and normal.shape[-1]==3
        assert torch.isfinite(normal).all() and torch.isfinite(depth_normal).all()
        log_curve=torch.clamp_max(torch.log(torch.clamp_min(torch.nan_to_num(curvature,0.0001),0.0001)),opt['curvature_clamp_threshold'])
        weight=(1-torch.sigmoid(log_curve)).detach()
        dot=(normal*depth_normal).sum(dim=-1)
        original=opt['lambda_normal']*(weight*(1-normal*depth_normal).sum(dim=-1)).mean()
        one_dot=opt['lambda_normal']*(weight*(1-dot)).mean()
        extra=2*opt['lambda_normal']*weight.mean()
        residual=float(original-one_dot-extra);assert abs(residual)<1e-6
        rows.append(dict(source_id=item['source_id'],native_surface_map=str(Path(item['path']).resolve()),
            native_normal_scalar_reconstructed_CPU=float(original),one_minus_dot_scalar_same_detached_weight=float(one_dot),
            extra_three_channel_scalar=float(extra),decomposition_abs_residual=abs(residual),detached_weight_mean=float(weight.mean()),
            native_rendered_normal_length_FG_mean=float(torch.linalg.vector_norm(normal,dim=-1)[mask].mean()),
            native_rendered_normal_length_BG_mean=float(torch.linalg.vector_norm(normal,dim=-1)[~mask].mean()),
            native_depth_normal_length_FG_mean=float(torch.linalg.vector_norm(depth_normal,dim=-1)[mask].mean()),
            native_depth_normal_length_BG_mean=float(torch.linalg.vector_norm(depth_normal,dim=-1)[~mask].mean())))
    with (output/'per_view.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    numeric=[key for key in rows[0] if key not in ['source_id','native_surface_map']]
    result=dict(status='complete',views=208,iteration=30000,native_preset=params['native_preset'],
        reconstructed_208_view_means={key:float(np.mean([row[key] for row in rows])) for key in numeric},
        native_model_path=str(Path(training['native_quadratic_ply']).resolve()),
        native_render_index_path=str(index_path.resolve()),native_render_index_SHA256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        CPU_float32=True,no_CUDA_bitwise_scalar_equivalence_claim=True,original_models_and_maps_unchanged=True,
        scope='Uniform208 terminal-model forward-map reconstruction, not the stochastic training trace. The original three-channel normal scalar contains an extra detached-weight term; raw direction lengths/alpha differ from an independently normalized geometric angle.',
        formula='lambda_normal*mean(detach(w)*(3-dot(native_normal,depth_normal))); additional same-step term=2*lambda_normal*mean(detach(w))')
    (output/'complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Actual terminal normal objective decomposition',result['reconstructed_208_view_means'],flush=True)

if __name__=='__main__':main()
