# Open3D 多交点接口的实际文档核对

本轮已通过 Context7 CLI 先解析 Open3D，再读取 `/websites/open3d` 的 `RaycastingScene.list_intersections` / `cast_rays` 文档。问题针对实际 Mesh 同一射线的全部交点、非单位方向与面 ID；没有把一次 `cast_rays` 的首交点冒充全部表面层。

官方来源：[RaycastingScene](https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html)。本机实际版本为 Open3D 0.19.0，实际调用及返回类型另外由实验校准确认；在线 latest 不冒称本地版本。

- 输入射线为 Float32 的原点与方向六元组；`t_hit` 随方向向量长度而改变，方向不要求单位化。
- `list_intersections` 返回各射线的全部面交点及 `ray_splits`、`ray_ids`、`primitive_ids`、`primitive_uvs`、`geometry_ids`、`t_hit`；本实验按每条射线排序并合并共享边上的重复交点。
- 本实验使用修正 K/W2C 生成相机 Z 分量为 1 的世界方向，因此 `t_hit` 是 camera-Z，两个交点的世界间距为 `delta_t * norm(direction)`。
- 三个明确标记为合成校准的平行平面已验证全部交点、返回类型与首交点一致性。校准不是植物实验结果。

[实际调用、校准及测量协议](../comparison/same_ray_mesh_layers/PROTOCOL.json)和原生 `actual_intersections.npz` 保留。多交点不能单独认证错误双层或真实叶片厚度。
