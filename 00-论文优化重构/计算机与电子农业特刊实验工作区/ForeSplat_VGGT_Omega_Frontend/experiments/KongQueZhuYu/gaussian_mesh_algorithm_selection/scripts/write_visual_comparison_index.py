"""Link only actual current overview images and their source protocols."""
import json
from pathlib import Path
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1];VIS=OUT/'comparison/visualizations'
def link(path,label=None):
    actual=path.resolve(strict=True)
    return f'[{label or path.name}](<{actual}>)'
lines=['# 实际统一视觉对照索引','',
    '以下图片均已生成并存在，原生30K与E14实际15K精修各自标明来源阶段。灰色图、相机法线、真实顶点颜色和线框分别观察，颜色图中的黑色不直接解释为几何孔洞。E14 Refined Gaussian中心Z/尺度轴图明确为AUX，Mesh表面图另存。','']
records=[]
groups=[('Gaussian固定208输入中的八个视角','gaussian','source_*.png'),('Mesh固定八视角与统一叶缘/内部缺失窗口','mesh_comparison','*.png'),('终态七Mesh路线（含QGS官方post）的八个实际叶缘与整株复核图','scientific_review','*.png'),('七终态Gaussian同相机原生自由RGB与Alpha','scientific_review/free','*.png'),('固定RGB匹配的原图/训练RGB及原生图','highlight_correspondence','*_all_native_maps.png'),('P4原生Gaussian与Expected/Median raw/post阶段图','P4_stage_visuals','*.png'),('原始背景与真实灰色/颜色叶缘图','leaf_surface_continuation','*.png'),('实际同射线交点与前后表面间距','same_ray_layers','*_actual_layers.png'),('剩余固定视角实际叶缘灰色/法线核查','fixed_view_continuation','*.png'),('紧凑直接查看图','compact_continuation','*.png')]
for title,folder,pattern in groups:
    root=VIS/folder;files=sorted(root.glob(pattern))
    if not files:continue
    lines+=['## '+title,'',f'实际已有{len(files)}张总览／同窗口图。','']
    protocol=root/'PROTOCOL.json'
    if protocol.is_file():lines+=[link(protocol,'相机、像素窗口与来源协议'),'']
    lines+=['|图片|实际文件大小|','|---|---:|']
    for path in files:
        lines.append('|'+link(path)+'|'+str(path.stat().st_size)+' bytes|')
        records.append(dict(group=folder,path=str(path.resolve()),file_size_bytes=path.stat().st_size))
    lines+=['']
lines+=['## 自由视角与俯视','', '自由视角用同一协议相机观察残留暗云与漂浮 Gaussian。E12自由视角是原生raw外观，逐帧EO矩阵只对应其已定义训练相机。','']
for folder in sorted(VIS.iterdir()):
    if not folder.is_dir() or not any(key in folder.name for key in ['_free','top_oblique','_top']):continue
    complete=folder/'complete.json'
    if not complete.is_file():continue
    lines.append('- '+link(complete,folder.name+'：实际渲染清单'))
    for path in sorted(folder.glob('*.png')):
        lines.append('  '+link(path))
        records.append(dict(group=folder.name,path=str(path.resolve()),file_size_bytes=path.stat().st_size))
review=VIS/'VISUAL_REVIEW.json'
paired=OUT/'comparison/paired_view_changes'
if (paired/'PROTOCOL.json').is_file():
    lines+=['','## 同一208视角的实际指标变化分布','',link(paired/'PROTOCOL.json','配对协议：相关视角，仅描述性分布，无独立重复试验推断'),'']
    for path in sorted(paired.glob('*ECDF.png')):
        lines.append(link(path));records.append(dict(group='paired_view_changes',path=str(path.resolve()),file_size_bytes=path.stat().st_size))
        pdf=path.with_suffix('.pdf')
        if pdf.is_file():lines.append(link(pdf,'对应可导出PDF'))
stability=OUT/'E13_QGS/native_stability/native_training_stability.png'
if stability.is_file():
    lines+=['','## QGS 原生训练与数值状态','',link(stability,'实际训练轨迹及保存Checkpoint中的非有限参数'),'']
    records.append(dict(group='E13_native_stability',path=str(stability.resolve()),file_size_bytes=stability.stat().st_size))
    protocol=stability.parent/'PROTOCOL.json'
    if protocol.is_file():lines+=[link(protocol,'真实采样范围与模型保存阶段'),'']
lines+=['', '## 直接视觉核验与空间证据','']
if review.is_file():lines+=[link(review,'当前模型直接看图记录'),'']
for relative,label in [('comparison/P4_local_failure/complete.json','P4黑色区域：Alpha、真实深度与Mesh同像素空间审计'),('comparison/P4_post_depth/complete.json','P4全部208视角post遮挡深度改变'),('comparison/REFLECTION_MESH_CORRESPONDENCE_INDEX.json','反光候选与真实Mesh三角面的世界坐标对应'),('comparison/P3_leaf_tip_spatial/complete.json','P3叶尖额外表面：原图、世界坐标与原生连接组件'),('comparison/SURFACE_AND_REFLECTION_AUDIT.md','实际原始RGB、前景拟合、起伏与法线口径审计')]:
    path=OUT/relative
    if path.is_file():lines+=[link(path,label),'']
lines+=['来源：真实本地Gaussian/Mesh渲染、固定原始RGB像素窗口及对应协议。不是AI生成的植物或Mesh示意图。']
(VIS/'VISUAL_COMPARISON_INDEX.md').write_text('\n'.join(lines)+'\n')
write_status(VIS/'VISUAL_COMPARISON_FILES.json',dict(status='actual_available_visual_index',records=records,all_paths_exist=all(Path(r['path']).is_file() for r in records)))
print('Actual visual overview links',len(records),flush=True)
