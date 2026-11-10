# 精修Mesh的纹理导入与拓扑适配

实际有效精修几何已交付：refinement/mesh_post/raw.ply与post.ply均为836,610顶点、1,588,062面。原生OBJ/MTL/PNG不变；OBJ为836,610顶点、4,773,984 UV和1,588,063面。两项导入差异均已定位，不能将它们当作精修算法补洞的收益。

**UV顶点复制。** Open3D带纹理导入生成4,764,186顶点，按索引连接的组件统计把大量面分开，最初共同小组件过滤没有有效post。恢复仅按完全相同XYZ合并顶点到836,610，没有距离阈值、坐标平均、平滑或补洞。全部1,588,062个已导入三角面的空间坐标逐位相同，前后SHA256一致；共同<50面组件规则保留同样的面数。证明见[精确合并验证](refined_mesh_exact_weld_validation.json)。

**末面遗漏。** 原生OBJ缺少末尾换行，当前Open3D跳过最后一面（0-based 1,588,062）。临时副本只加一个换行即可导入全部1,588,063面；源OBJ字节不变。该面面积约1.4284226e-6未标定VGGT gauge²，非退化面。最近顶点一一对应最大坐标舍入差8.021e-8，无多余面或重复面。依据：[实际面对应](native_OBJ_import_face_correspondence.json)、[EOF复核](native_OBJ_terminal_newline_audit.json)。

交付完整原生OBJ与已按1,588,062面完成统一208评价的几何导入PLY，各自实际计数单独索引。为保持评价对象一致，没有在评价结束后替换PLY。优先用Blender查看完整OBJ；在Open3D查看时可对单独副本补末尾换行。

refinement/refined_raw.ply保留未合并原导入；refinement/uv_seam_unwelded_post_attempt/保留原失败尝试记录。有效共同raw/post是refinement/mesh_post/中的实际文件。全部版本真实路径见[Mesh索引](../artifact_index/MESH_FILE_INDEX.md)。

来源：官方OBJ实际记录数、原生纹理导出源码、真实Open3D导入与上述数值/逐位核验文件。
