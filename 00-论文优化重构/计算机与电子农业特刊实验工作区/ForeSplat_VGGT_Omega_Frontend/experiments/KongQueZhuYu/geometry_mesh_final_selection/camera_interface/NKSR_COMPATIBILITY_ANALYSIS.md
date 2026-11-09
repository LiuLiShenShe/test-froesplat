# NKSR native compatibility validation

Official NKSR commit e40336845e67761343a756788e5a98b827d4a143, frozen ks pretrained weights and all reconstruction/input parameters retained. Original M1 stopped SIGFPE in libcublas Sdot at ~19,398MiB, not a demonstrated OOM. A known 2-by-2 SPD and independent private cuBLAS handle also failed. Superseded private-handle patch and direct failure log remain available.

Active `nksr_aten_vector_compat.patch` replaces only cuBLAS Level1 dot/norm/copy/scale/axpy with mathematically equivalent float32 ATen operations. Official sparse matrix construction, Jacobi PCG equations, CUDA sparse matvec, tolerance1e-5, maxiterations2000 and extraction parameters remain. No new model/loss/fusion algorithm is introduced. Parallel reduction order may produce ordinary float32 numerical differences, so it is recorded as runtime compatibility, not a bit-identical binary.

Known-system validation: N=2/10,000/1,000,000 diagonal SPD max absolute error0; unequal-block257 dense SPD converged17iterations, max solution error1.159906e-5 and relative residual5.880694e-6 against independent CPU float64 solve. JSON at `nksr_pcg_validation.json`; build and native module SHA at `../configs/nksr_compatibility_module.json`. Module atomically replaced only in independent uv environment, leaving shared uv cache untouched. Full plant reconstructions use this same binary for all inputs.

Sources: [official NKSR](https://github.com/nv-tlabs/nksr), [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html), actual source/headers and reproduced native backtrace. Context7 fetched NVIDIA docs; Bocha search quota unavailable.
