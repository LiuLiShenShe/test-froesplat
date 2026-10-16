# PUP 2DGS Adaptation Status

`PUP_2DGS_ADAPTED`: **BLOCKED for scene-scale pruning**.

The official implementation assumes 3DGS XYZ plus three activated scales, a 6x6 spatial Fisher, and a CUDA rasterizer that accepts a 21-entry symmetric Fisher buffer. The current 2DGS rasterizer has a two-scale surfel covariance path and no `fishers` keyword or `f_count` output. Passing an N x 6 tensor into the native 2D path would be an ABI and memory-layout error, not an adaptation.

The phase includes a small exact 5-parameter probe (`reports/FISHER_TINY_PROBE.json`). It computes per-output Jacobians, forms 5x5 outer products before reduction, and checks centered finite differences. It proves the required mathematical direction and also demonstrates that an ordinary summed screen-space gradient is not equivalent. It does not claim a production Fisher scorer.

Required G1 work is a dedicated 2D surfel CUDA extension that accumulates 15 symmetric entries for XYZ and the two log-scales after the full 2DGS covariance chain, plus tests against the Python oracle and a camera-level reference. Only after that should a method name such as `PUP_2DGS_ADAPTED` be enabled.
