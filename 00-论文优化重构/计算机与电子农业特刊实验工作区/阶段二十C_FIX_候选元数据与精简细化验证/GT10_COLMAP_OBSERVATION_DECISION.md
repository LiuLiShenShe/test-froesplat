# GT10 COLMAP observation decision

DouBanLv2 0037、0039、0041 在冻结原始 OPENCV distorted sparse 模型和
PINHOLE undistorted sparse 模型中均未注册（两个模型各 210 registered images）。
逐条核对真实 image.name / image_id / camera_id，而非只依赖生产前缀匹配。
0038、0040 能找到真实 image_id 35、36 及非负 point3D_id；不得借给缺失帧。
原始输入 RGB 存在，并与冻结候选输入 SHA256 匹配；不是输入路径丢失或前缀错误。

缺失帧保留 `GEOMETRY_NOT_AVAILABLE`。不生成伪稀疏点、不用 GT 反馈、
不借邻帧观测、不重建 COLMAP、不改 COLMAP/2DGS 核心算法。

额外坐标审计：DouBanLv2 undistorted 模型 PINHOLE 为 2119×3779，
distorted OPENCV 为 2160×3840，冻结 RGB 为 2160×3840。
旧生产 sparse 优先路径的 undistorted xys 不能仅凭名称直接认定与原始 RGB
同坐标。完整表 `COLMAP_MODEL_IMAGE_AUDIT.csv` 记录所有 Easy21/GT10 注册、
camera dimensions、RGB identity 和模型 hash。历史 V3 可作同帧保存掩膜的
描述性比较；在坐标映射未验证前，不作为 GCFC 机制收益或最终默认依据。

SPNP-only 未启用 evidence_guidance、geometry_feedback、corrective_geometry、
A6/A7；仅用当前 RGB、已选 mask、冻结 SAM3 checkpoint 和 mask 派生正负框。
因此上述几何缺失不阻断 V4/V5。

来源：原始 `04-COLMAP/DouBanLv2/{sparse/0,distorted/sparse/0}/{images,cameras}.bin`；
只读官方格式解析器 `2d-gaussian-splatting-main/scene/colmap_loader.py`；
本阶段 `COLMAP_MODEL_IMAGE_AUDIT.csv`。未修改历史几何。
