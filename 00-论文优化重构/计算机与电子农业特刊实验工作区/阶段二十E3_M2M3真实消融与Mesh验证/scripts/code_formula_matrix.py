"""Generate a traceable implementation matrix without changing production code."""
from common import *
import ast

def main():
    source=BRANCH/'train.py';tree=ast.parse(source.read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    data=[]
    def row(concept,status,function,formula,note):
        n=functions[function]
        data.append({'concept':concept,'implementation_status':status,'function':function,'source_path':str(source),'source_line':n.lineno,
                     'source_sha256':sha(source),'mathematical_definition':formula,'scope_and_limitation':note,
                     'runtime_evidence':'reports/*_INITIALIZATION.json; metrics/*_PRUNING.csv; external capacity_control/*.json'})
    row('Mask-to-model support','IMPLEMENTED_CENTER_PROXY','mask_consistency_scores','s_i=sum_v I(in_frustum_iv)*I(nearest_alpha_iv>0.5)/sum_v I(in_frustum_iv); unseen=1',
        'Up to12 fixed-stride train cameras; Gaussian center, nearest grid_sample align_corners=True. Positive z and strict NDC bounds; no occlusion test, no surface-area support.')
    row('Maximum normalization','IMPLEMENTED','normalized_positive','N(x)_i=max(x_i,0)/max(max_j max(x_j,0),1e-6)','Over all current Gaussians, not only candidates. Detached float32.')
    row('Opacity penalty','IMPLEMENTED_SCORE','m2m3_prune_scores','o_i=max(0.005-sigmoid(opacity_i),0)','Weight1; opacity does not create candidates in frozen pruning_mode=mask.')
    row('Brightness penalty','IMPLEMENTED_SCORE','m2m3_prune_scores','b_i=max(0.01-mean_rgb(clamp(0.28209479177387814*f_dc_i+0.5,0,1)),0)','Weight1; SH DC brightness, not rendered photometric residual.')
    row('Mask penalty','IMPLEMENTED_SCORE','m2m3_prune_scores','m_i=max(0.45-s_i,0)','Weight3, only when region_mode=foreground and masks available.')
    row('Scale penalty','IMPLEMENTED_SCORE','m2m3_prune_scores','l_i=max(max_axis(exp(scaling_i))-torch_median_j(max_axis(exp(scaling_j))),0)','Weight0.5; torch median uses lower middle for even counts; no mesh curvature term.')
    row('Screen radius penalty','IMPLEMENTED_SCORE','m2m3_prune_scores','v_i=max(max_radii2D_i,0)','Weight0.5; accumulated native screen radius, not bidirectional visibility matching.')
    row('Weighted pruning score','IMPLEMENTED','m2m3_prune_scores','q_i=N(o)_i+N(b)_i+3*N(m)_i+0.5*N(l)_i+0.5*N(v)_i','Larger score is pruned first inside candidate set. No loss term or gradient update from score.')
    row('Candidate set','IMPLEMENTED_MASK_ONLY','maybe_prune_gaussians','C={i:s_i<0.45}','Frozen pruning_mode=mask; opacity/brightness union applies only to other modes and is not this experiment.')
    row('Pruning budget','IMPLEMENTED','maybe_prune_gaussians','B=floor(0.03*N); C_B=top_q(C,min(|C|,B)) when B>0','At B=0 native path skips budget limiting; actual models are larger. Gate18000,21000,24000,27000,30000.')
    row('Global Floor40','IMPLEMENTED_EXTRA_PRUNE_ONLY','resolve_capacity_floor_count','F=min(N,max(0,round(0.4*N_max_seen))); allowed=max(0,N-F)','Python round; peak tracks native update calls. M2 applies this bound only to extra pruning; M1 does not.')
    row('Keep/prune','IMPLEMENTED','maybe_prune_gaussians','M1 removes C_B; M2 removes top_q(C_B,min(|C_B|,allowed))','Native prune_points removes Gaussian and optimizer tensors; ordinary densify_and_prune remains separately active.')
    row('Model-to-mask deficit','NOT_IMPLEMENTED','training','NONE','No independent deficit map, deficit objective, or target-driven densification path.')
    row('Boundary-aware utility','NOT_IMPLEMENTED','m2m3_prune_scores','NONE','No boundary distance or utility term in this actual score.')
    row('Bidirectional matching','NOT_IMPLEMENTED','training','NONE','Do not describe center support plus pruning as bidirectional mask/model optimization.')
    row('Local Floor40','NOT_IMPLEMENTED','resolve_capacity_floor_count','NONE','No local region/leaf/voxel capacity floor; existing bound is one global scalar.')
    row('Extra deficit densification','NOT_IMPLEMENTED','training','NONE','Only standard 2DGS densification; no M2M3-specific new Gaussians.')
    write_csv(STAGE/'M2M3_CODE_FORMULA_MATRIX.csv',data)

if __name__=='__main__':main()
