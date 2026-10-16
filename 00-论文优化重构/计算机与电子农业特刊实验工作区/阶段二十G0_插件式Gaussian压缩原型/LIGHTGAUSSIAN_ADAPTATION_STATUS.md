# LightGaussian 2DGS Adaptation Status

`LIGHTGAUSSIAN_2DGS_ADAPTED`: **BLOCKED**.

The pinned implementation requires a compression rasterizer with `f_count=True`. Its kernel records accepted visible Gaussian/pixel hits and adds `con_o.w` opacity to `important_score`. The native 2DGS rasterizer exposes no equivalent output. The second upstream adjustment multiplies importance by a power of the product of three activated scales, which is a 3D volume convention and is not automatically valid for 2D surfels.

The plugin therefore does not substitute alpha, projected area, opacity, or a guessed surface-area multiplier. The source-level entry points and exact submodule commit are recorded in `TOP_CONFERENCE_CODE_AUDIT.md`; an attempted enabled run is retained under the external asset directory with `status=FAILED` and no PLY output.

The reusable parts are the post-hoc score-to-mask control flow, source-index pruning idea, and optional recovery workflow concept. The CUDA count output, two-scale surface support metric, and recovery optimizer state require an explicit adapter before implementation.
