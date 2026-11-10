# E6 历史相机字段恢复

旧camera_runtime.json保留208相机的FoV、fov_derived_fx/fy、W2C、P、完整投影矩阵与相机中心，但Fx/Fy/Cx/Cy为null，导致统一适配器在进入渲染前失败。

本次仅在新比较目录生成运行元数据：焦距取真实已保存的FoV-derived字段；主点由原始P和历史ndc2pix整数中心公式恢复。全部208份W2C/P/fullP/center逐项保持原值，旧文件没有改动。射线探针投影与保存float32 P的最大差异为1.2836e-5像素。

[原字段与恢复证据](CAMERA_FIELD_RECOVERY_AUDIT.json) · [恢复后的实际运行记录](recovered_camera_runtime.json)。

E6继续使用自己的历史中心投影及原生2DGS实现；它不是E10的修正相机对照。统一光度GT/RAP与208IDs保持相同，历史前端/训练条件差异单独保留。来源：真实旧相机文件及实际矩阵计算，无新前端推理。
