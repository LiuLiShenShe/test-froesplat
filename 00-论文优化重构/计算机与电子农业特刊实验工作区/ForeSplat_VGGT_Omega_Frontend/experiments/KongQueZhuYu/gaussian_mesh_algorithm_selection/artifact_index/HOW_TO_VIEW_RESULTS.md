# 实际模型查看指南

E12 GGGS+EO 30K、E13 原生QGS Mip 30K、E14 原生2DGS 30K及SuGaR 15K精修均已实际完成。先打开[Gaussian索引](GAUSSIAN_PLY_INDEX.md)、[Mesh索引](MESH_FILE_INDEX.md)或[机器索引](ALL_ARTIFACTS.csv)：每项记录实际解析后的绝对文件路径、字节数、类型与基元/顶点/面数。大模型保留在原数据盘，不重复复制。

推荐按 **E10 → P2 Cleaned → E11 → E12 → E13 → E14 原生30K → E14 Refined** 查看Gaussian，再按 **M4 → P1 → P2 TSDF/Tetrahedra → P3 native/shared → P4六组 → E12 TSDF/Tetrahedra → E13 raw/共同post/官方top10 → E14清理前后/初始/Refined** 查看Mesh。最终选择见[算法选择报告](../FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)。

## Gaussian和原生模型

|文件类型|适合的查看方式|实际解释|
|---|---|---|
|E1–E5、E7–E10、P2、E12：三尺度Opacity/SH Gaussian PLY|[SuperSplat Editor](https://superspl.at/editor)导入本机已下载的PLY，或原生GGGS渲染器|普通Gaussian外观与GGGS stochastic-solid原生表面/深度渲染不完全相同|
|E6/E11、E14原生/Pruning：两尺度surfel PLY|原生2DGS或本工作区原生Viewer|面内两尺度，不能当作标准三尺度3DGS直接导入；普通XYZ点云不能代替surfel表面|
|E13：六尺度/二次曲面原生PLY及PTH|[QGS原生Viewer指南](../E13_QGS/NATIVE_VIEWER_GUIDE.md)|原生Quadratic Primitive，不是标准3DGS|
|E14 Refined：三尺度Mesh绑定Gaussian PLY及PT|SuperSplat或E14原生Viewer|标准三尺度外观可查看；Mesh绑定优化状态保留在15000.pt中|
|初始化XYZ/颜色点云、P2 Removed PLY|MeshLab/点云查看器；标准Gaussian字段仍存在时也可原生渲染|Removed用于检查实际删除位置；普通点云只代表中心位置|

E13在E13_QGS目录执行已交付本地入口：

```bash
uv run --no-project --python .venv/bin/python python scripts/native_qgs_viewer.py --iteration 30000 --port 8767 --width 540
```

同一主机打开 http://127.0.0.1:8767 。此入口加载实际Mip 30K；归档base失败/中间模型必须按其Viewer指南显式选PLY与对应配置。终态532,735个基元中3,456个包含原生非有限参数，索引和数值审计如实保留；Checkpoint/PLY来源逐位一致并不表示参数质量全部有效。

E14在E14_2D_SuGaR目录运行：

```bash
uv run --no-project --python .venv/bin/python python scripts/view_native_web.py --stage final --port 8864
```

同一主机打开 http://127.0.0.1:8864 。--stage before、--stage pruned查看实际7K清理前/后，--stage refined查看实际15K精修导出。Viewer使用物理GPU1；训练队列结束后可启动。E14 Refined的Gaussian深度/法线页面是**Alpha归一化中心Z/最小尺度轴辅助图**，原生3DGS没有2DGS表面深度通道；实际Mesh raycast Depth/Normal另存，不混为同一数据。

E12逐视角3×4曝光矩阵独立于PLY，实际文件在 E12_GGGS_EO/exposure_params/exposure_matrices.npz，绝对路径列在Gaussian/机器索引。直接打开PLY得到raw颜色，明显较暗；208视角EO-corrected图另应用该相机矩阵。自由相机没有唯一已训练曝光参数，不能把raw自由视角宣称为corrected结果。E2/E4历史曝光参数同样另存并已索引。

## Mesh、颜色和纹理

Mesh PLY可用Blender、MeshLab或Open3D打开。先看纯灰与法线，再看颜色与Wireframe。raw、Binary Search、共同post、官方post、refined分别保留。顶点着色PLY是有颜色的几何Mesh，不是带UV纹理的OBJ。P1/P2/E12另有顶点着色post，其几何顶点/面哈希与无色post一致。

E14精修OBJ需要一起下载同目录MTL与其引用PNG；实际依赖路径与字节数见Mesh索引。原生OBJ含836,610个顶点和1,588,063个面。它没有末尾换行，当前Open3D导入遗漏最后一个非退化面；临时副本只补一个换行可导入全部面。原OBJ字节未改。交付几何PLY保留实际导入并完成208评价的1,588,062面，不能把差一面解释为算法补洞/孔洞。建议用Blender查看完整原OBJ；需要Open3D时对**单独查看副本**补末尾换行。详见[导入与精确合并证明](../E14_2D_SuGaR/MESH_TOPOLOGY_RECOVERY.md)。

UV导入还会复制纹理缝顶点；已提供仅按完全相同XYZ合并的 refinement/mesh_post/raw.ply与共同post.ply。全部已导入三角形坐标逐位相同，没有平滑、补洞或顶点平均。未合并raw和失败组件过滤尝试保留且在索引标明。

## 相机、方向和图像数据

各文件保留原VGGT预测尺度和世界坐标，未标定毫米/厘米。标准VGGT新增路线共用修正K/W2C/像素中心；VGGT-Ω另有自己的gauge，不能未经配准直接叠加XYZ。固定照片ID相同仅表示同一照片。历史E1–E7/E9保留各自训练时的相机接口；统一208评价不改写历史模型。

需要倒置显示时，在查看器中对视图/对象施加绕X轴180°显示旋转：diag(1,-1,-1,1)。不要保存回原始几何，也不要将显示旋转当作尺度/精度校准。固定八视角为0、26、53、79、105、131、157、183；对照图使用相同相机、显示尺度及RGB像素窗口，另保留共享俯视/斜俯视和自由相机协议。

机器索引的一行可能是NPZ中的一个实际数组。stage注明NPZ array名称，字节数指整个归档；按记录数组名读取。Alpha、Depth、Normal共用文件不代表相同数据。Depth为相机像素网格上的相机轴Z；Normal逐项区分相机/世界向量以及Alpha加权累积或单位法线。EO是图像颜色空间仿射参数，不是空间变换。

E14原始Metric3Dv2深度/法线保留额外先验原输出。深度已做焦距去规范化，但尚未乘逐图轨迹中位尺度；实际尺度保存于 input_colmap/prior_depth_scale.json。原始法线尚未转换到世界方向。权重SHA、RGB来源和预处理记录均保留，先验不等同于同株扫描真值。

来源：实际PLY头、原生源码/Viewer脚本、OBJ/MTL依赖、机器索引及[SuperSplat官方导入文档](https://github.com/playcanvas/supersplat/wiki/Importing-3DGS-Scenes-into-your-Projects)和[场景变换文档](https://github.com/playcanvas/supersplat/wiki/SuperSplat-Editor-Interface)（已通过Context7读取）。官方固定源码README直接联网核验成功；博查搜索HTTP403余额/额度不足，没有把未取得的搜索结果作为依据。
