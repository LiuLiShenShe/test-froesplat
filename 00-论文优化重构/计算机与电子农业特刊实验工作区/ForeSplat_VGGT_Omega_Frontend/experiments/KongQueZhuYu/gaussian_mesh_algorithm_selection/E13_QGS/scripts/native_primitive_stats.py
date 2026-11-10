"""Export native QGS curvature diagnostics without equating them to3DGS scales."""
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1]
def quantiles(a):
 a=np.asarray(a);finite=np.isfinite(a)
 values=np.quantile(a[finite],[0,.01,.1,.5,.9,.99,1]).tolist() if finite.any() else []
 return dict(quantile_levels=[0,.01,.1,.5,.9,.99,1],values=values,finite_count=int(finite.sum()),nonfinite_count=int((~finite).sum()))
def main():
 ply=ROOT/'checkpoints/point_cloud/iteration_30000/point_cloud.ply';assert ply.is_file()
 v=PlyData.read(ply,mmap='r')['vertex'].data
 raw=np.column_stack([v[f'scale_{i}'] for i in range(6)]).astype(np.float64)
 signed=np.exp(raw[:,:3])*np.tanh(raw[:,3:])
 sx,sy,sz=signed.T
 with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
  a=sz*np.sign(sx)/sx**2;b=sz*np.sign(sy)/sy**2
  ratio=np.maximum(abs(sx),abs(sy))/np.minimum(abs(sx),abs(sy))
  K=4*a*b;H=a+b
 xyz=np.column_stack([v[k] for k in ['x','y','z']])
 quaternion=np.column_stack([v[f'rot_{i}'] for i in range(4)])
 qnorm=np.linalg.norm(quaternion,axis=-1,keepdims=True)
 # Match native normalize's epsilon while retaining NaN rotations explicitly.
 quaternion_unit=quaternion/np.maximum(qnorm,1e-12)
 opacity=1/(1+np.exp(-np.clip(v['opacity'].astype(np.float64),-700,700)))
 out=ROOT/'native_geometry';out.mkdir(exist_ok=True)
 np.savez_compressed(out/'quadratic_primitive_parameters.npz',xyz=xyz,raw_scale_parameters=raw,signed_native_scaling=signed,
  quaternion=quaternion,quaternion_unit=quaternion_unit,opacity=opacity,tangential_anisotropy=ratio,curvature_a=a,curvature_b=b,
  Gaussian_curvature_at_patch_origin=K,mean_curvature_at_patch_origin=H)
 raw_finite=np.isfinite(raw).all(axis=1)&np.isfinite(xyz).all(axis=1)&np.isfinite(quaternion).all(axis=1)&np.isfinite(v['opacity'])
 singular=(signed==0).any(axis=1)
 valid=raw_finite&np.isfinite(signed).all(axis=1)&(~singular)
 sign=np.sign(signed);saddle=valid&(sign[:,0]*sign[:,1]<0)
 convex=valid&(~saddle)&(sign[:,0]*sign[:,2]>=0);concave=valid&(~saddle)&(~convex)
 j=dict(status='complete',primitive_count=len(v),source_native_ply=str(ply.resolve()),coordinate_system='standard VGGT original world gauge',
  diagnosis='native quadratic parameters; not ordinary3DGS Gaussian covariance/thickness',
  decoder='s=exp(scale0..2)*tanh(scale3..5), computed in float64 for diagnostic export',
  rotation='quaternion stores native raw WXYZ; quaternion_unit matches official normalized get_rotation',
  native_equation='sign(sx)*x²/sx² + sign(sy)*y²/sy² − z/sz =0 in primitive local frame',
  curvature='z=a*x²+b*y²; a=sz*sign(sx)/sx²,b=sz*sign(sy)/sy²; origin principal curvatures2a/2b, K4ab,H=a+b',
  units='a,b,H inverse predicted gauge unit; K inverse predicted gauge unit²; not calibrated metric distances',
  shape_labels='local positive-Z convention; sign/rotation changes must be considered before interpreting plant surface orientation',
  native_nonfinite_primitives=int((~raw_finite).sum()),native_nonfinite_fraction=float((~raw_finite).mean()),
  native_zero_signed_scale_primitives=int(singular.sum()),
  invalid_native_parameters_retained_unchanged=True,
  native_shape_counts=dict(saddle=int(saddle.sum()),convex=int(convex.sum()),concave=int(concave.sum()),undefined=int((~valid).sum())),
  statistics=dict(tangential_anisotropy=quantiles(ratio),abs_sx=quantiles(abs(sx)),abs_sy=quantiles(abs(sy)),abs_sz_curvature_scale=quantiles(abs(sz)),
   curvature_a=quantiles(a),curvature_b=quantiles(b),mean_curvature_origin=quantiles(H),Gaussian_curvature_origin=quantiles(K)),
  parameter_archive=str((out/'quadratic_primitive_parameters.npz').resolve()),
  interpretation='Center projection alone does not establish per-pixel primitive visibility; spatial highlight comparisons should use actual rendered depth/normal/alpha support',
  official_source='https://github.com/will-zzy/QGS/blob/74d05c945e99fcaef7afe5a8831903be71ad9b55/submodules/diff-quadratic-rasterization/cuda_rasterizer/forward.cu')
 (out/'statistics.json').write_text(json.dumps(j,indent=2)+'\n');print('Native QGS curvature diagnostics exported',len(v),flush=True)
if __name__=='__main__':main()

