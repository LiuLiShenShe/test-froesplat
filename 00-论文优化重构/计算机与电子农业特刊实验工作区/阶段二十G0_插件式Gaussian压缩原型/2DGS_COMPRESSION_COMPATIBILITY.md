# 2DGS Compression Compatibility

## Native parameter contract

The frozen local 2DGS loader uses XYZ `(N,3)`, opacity logits `(N,1)`, log-scales `(N,2)`, quaternion `(N,4)`, DC SH `(N,1,3)`, and rest SH `(N,15,3)` for degree 3. Scaling is activated with `exp`; opacity with `sigmoid`; rotation is normalized. The native covariance path constructs a 4x4 homogeneous surfel transform from two in-plane scales plus a unit homogeneous axis. This is not the 3DGS three-scale ellipsoid covariance.

The PLY adapter preserves the original scalar property order and dtype, including normal fields and unknown scalar fields. It rejects list properties, non-finite values, zero quaternions, activated scale overflow, and a three-scale 3DGS schema. The original native `GaussianModel.load_ply` was used to reload every real output.

## Compatibility matrix

| Component | Classification | Evidence and required work |
|---|---|---|
| XYZ, opacity, SH storage | DIRECTLY_REUSABLE | Native PLY fields and row-wise deletion retain values exactly. |
| Stable score ranking and exact capacity | DIRECTLY_REUSABLE | Generic adapter; independent of rasterizer. |
| Model directory/PLY loading | ADAPTER_REQUIRED | 2DGS iteration directory and two-scale schema are handled by `io.py` and native validation. |
| 2DGS two-scale parameterization | ADAPTER_REQUIRED | PUP's `(x,y,z,sx,sy,sz)` must become `(x,y,z,log_scale_0,log_scale_1)`; the surface covariance chain is different. |
| PUP Python Fisher | ADAPTER_REQUIRED + CUDA_CHANGE_REQUIRED for production | Official Python path differentiates each RGB patch sum with respect to XYZ plus three scales. A faithful 2DGS version needs five spatial parameters and per-observation outer products before reduction. |
| PUP Fisher CUDA ABI | CUDA_CHANGE_REQUIRED | Official extension accepts `fishers` and writes 21 symmetric entries per Gaussian; native 2DGS `GaussianRasterizer.forward` accepts neither `fishers` nor `f_count`. |
| PUP 3D covariance backward | INCOMPATIBLE without port | Official CUDA uses 3D ellipsoid covariance and three scales; native rasterizer uses surfel `transMat` and two scales. |
| LightGaussian count renderer | ADAPTER_REQUIRED + CUDA_CHANGE_REQUIRED | Official `count_render` requires `f_count`; its pinned kernel counts accepted pixel hits and adds `con_o.w` opacity. Native 2DGS has no such output. |
| LightGaussian volume multiplier | INCOMPATIBLE as-is | Official `calculate_v_imp_score` uses product of three activated scales. A 2D surface needs an explicitly defined area or footprint metric, not an unannounced volume substitution. |
| SH distillation/quantization | OUT OF SCOPE | Explicitly excluded in Phase 20G-0. |
| Official optimizer recovery | ADAPTER_REQUIRED | Both upstream workflows are training/recovery programs. This plugin is post-hoc and currently performs no optimization. |

## Correctness probe

`scripts/probe_fisher.py` computes an exact small 2DGS 5-parameter Jacobian by separate output backward calls and validates it against centered finite differences. Its Fisher is not registered as a scene-scale score. On the probe, taking an outer product after summing pixel gradients had relative error about 8.79, so the native ordinary screen-space gradient cannot stand in for PUP's per-observation Fisher.

## Conclusion

PUP and LightGaussian cannot be claimed as direct 2DGS adaptations. The independent row-wise compressor is real and native-loadable; conference scoring remains blocked until a dedicated 2D surfel CUDA extension is implemented and independently validated.
