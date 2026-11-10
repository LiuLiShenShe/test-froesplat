# P3 定点修复

source157漂浮环 **Resolved（固定已定位几何范围）**；整株Mesh **Improved**；内部错误连接 **Unresolved**。

原P3后处理Mesh最大实际组件是32，source157独立组件28/29共829三角面。208视角面中心可见/Mask/GGGS median/stereo证据：28组件visible73250、MaskFG39977，但GGGS深度支持0、Stereo仅5次（每面最多1视角）；29 visible17808、FG10464、两类深度支持0。前景Mask覆盖不能证明真实叶片。结合原RGB与已存世界定位，v1仅删除这829面，保留原Triangle ID映射与被删单独Mesh。

v2在v1上移除独立蓝标记组件97的10072面，总删10901；盆形主体保留。灰色、法线、线框、色彩、深度、三角面/组件图已生成并直接看过。source157固定ROI额外投影约1600→159，原定位1441个环像素已去除；叶尖保留，缺面仍221px。该Resolved不涵盖整叶或全Mesh。

| model | stage | iou | boundary_f1 | boundary_chamfer_px | fg_coverage | bg_geometry | hole_px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M4_legacy | post | 0.882547 | 0.122550 | 35.816651 | 0.991885 | 0.060210 | 545.052885 |
| G0 | post | 0.915556 | 0.151903 | 22.294921 | 0.988083 | 0.036240 | 358.312500 |
| G1 | post | 0.914219 | 0.150062 | 22.937884 | 0.988187 | 0.037201 | 326.975962 |
| G2 | post | 0.918008 | 0.157959 | 19.565034 | 0.986937 | 0.033728 | 397.783654 |
| P3_local | v1 | 0.916275 | 0.153032 | 22.226719 | 0.988079 | 0.035925 | 358.740385 |
| P3_local | v2 | 0.913268 | 0.148717 | 22.928656 | 0.982363 | 0.034821 | 358.620192 |
| G2_targeted | post | 0.921293 | 0.163733 | 19.207052 | 0.982709 | 0.030164 | 347.894231 |

54个主组件细长连接候选有≥15可见视角/≥5Mask前景却缺乏深度支撑；candidate_review原照片显示部分与真实叶柄/卷叶重叠。未直接切除全部候选，也未作整株Laplacian smoothing或填叶间隙。Triangle ID及候选单独Mesh保留在wrong_connections；wrong_connections_edited=false。尚不能确认并修复叶内部全部桥接。

局部黑色区拆分为几何未命中、已有面但颜色偏黑、原照片本身深色。G2 source131局部2368个缺面像素、其中2193有有效source Stereo；原P3v2同ROI2362，故清理/重融合未解决此局部缺损。surface roughness/相邻法线角含真实卷叶，不称作扫描误差。

直接P3最佳版本为 `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task1_p3_local_repair/mesh_versions/P3_best_cleaned_mesh.ply`，解析到v2。v1保留历史含标记评价条件；v2为目标Plant+Pot。原始raw/post是保留引用，不覆盖源数据。
