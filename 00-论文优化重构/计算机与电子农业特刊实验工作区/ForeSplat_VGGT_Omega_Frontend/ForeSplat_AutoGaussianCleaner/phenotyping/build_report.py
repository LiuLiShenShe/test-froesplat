"""Publish actual local geometry, provisional metrics and the empty-pair audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
from PIL import ImageDraw,ImageFont
from extract_instances import PALETTE
from manual_errors import rows
import html

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(size):return ImageFont.truetype(FONT,size)
def text_at(draw,xy,text,size=20,fill='#183827'):draw.text(xy,text,font=font(size),fill=fill)
def line(draw,xyz,name,color,width=3):
 if len(xyz)<2:return
 xy,z=project(np.array(xyz),name);good=z>0
 if np.all(good):draw.line([tuple(p) for p in xy],fill=color,width=width)

def figures(meter,source,leaves,traits,plant,frame):
 vertices=np.asarray(source.vertices);faces=np.asarray(source.triangles);sheet=Image.new('RGB',(4*480,3*420),'#f4f7f4');mesh_sheet=Image.new('RGB',(4*480,3*420),'#f4f7f4')
 for index,record in enumerate(leaves):
  leaf=record['leaf_id'];folder=OUT/'leaf_instances'/leaf;landmarks=read_json(folder/'landmarks.json');name=landmarks['reference_view'];row=next(r for r in traits if r['leaf_id']==leaf)
  # Original RGB plus actual projected mesh paths; no generated imagery.
  image=rgb(name,meter);d=ImageDraw.Draw(image);line(d,landmarks['centerline_xyz_units'],name,'#ffff00' if landmarks['path_valid'] else '#ff3344',3)
  if landmarks['width_cross_sections']:
   cross=max(landmarks['width_cross_sections'],key=lambda r:r['length_units']);line(d,vertices[cross['path_source_vertices']],name,'#00ffff',3)
  for kind,color in [('base','#ffbb00'),('tip','#00ddff')]:
   xyz=landmarks[kind+'_xyz_units']
   if xyz:
    xy,_=project(np.array([xyz]),name);x,y=xy[0];d.ellipse((x-5,y-5,x+5,y+5),fill=color);d.text((x+6,y),kind,fill=color,stroke_width=1,stroke_fill='black')
  mask=np.array(Image.open(folder/'masks'/f'{name}.png'))>0;yy,xx=np.where(mask)
  projected,_=project(np.array(landmarks['centerline_xyz_units']),name)
  if len(projected):xx=np.r_[xx,projected[:,0]];yy=np.r_[yy,projected[:,1]]
  box=(max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(530,int(xx.max())+25),min(948,int(yy.max())+25))
  tile=Image.new('RGB',(480,420),'white');draw=ImageDraw.Draw(tile);text_at(draw,(12,8),leaf+' · '+name,20);crop=image.crop(box);crop.thumbnail((450,300));tile.paste(crop,((480-crop.width)//2,42));text_at(draw,(12,350),f"曲面路径 {float(row['leaf_length_cm']):.2f} cm  宽 {float(row['leaf_width_cm']):.2f} cm" if row['leaf_length_cm'] and row['leaf_width_cm'] else '路径不可用，见 CSV',17);text_at(draw,(12,378),f"单侧观测面积 {float(row['leaf_area_cm2']):.2f} cm² · 局部曲面",17)
  sheet.paste(tile,((index%4)*480,(index//4)*420))
  with np.load(io.BytesIO(meter.read(folder/'source_indices.npz'))) as archive:ids=archive['original_face_ids']
  mesh,vid=subset_mesh(source,ids);hit,z,uv=cast(cast_scene(mesh),name);rendered=np.full((948,530,3),245,np.uint8);visible=hit>=0;normals=np.asarray(mesh.triangle_normals)
  # Per-face lighting from exact source normals; a face ID is never replaced
  # by a filled silhouette when a hole exists.
  mesh.compute_triangle_normals();normal=np.asarray(mesh.triangle_normals);direction=CAMERAS[name]['c2w'][:3,3]-np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(1);direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9);shade=.45+.55*np.abs((normal*direction).sum(1));color=PALETTE[index]
  rendered[visible]=(shade[hit[visible],None]*color).astype('uint8');pic=Image.fromarray(rendered).crop(box);pic.thumbnail((450,300));tile=Image.new('RGB',(480,420),'white');dr=ImageDraw.Draw(tile);text_at(dr,(12,8),leaf+' · 原始三角面子集',20);tile.paste(pic,((480-pic.width)//2,42));text_at(dr,(12,350),f"{record['faces']:,} 面 · {record['components']} 连通片",18);text_at(dr,(12,380),'保留孔洞及原始坐标；未补面',17);mesh_sheet.paste(tile,((index%4)*480,(index//4)*420))
 png(OUT/'summary/measurement_visualization.png',np.array(sheet),meter);png(OUT/'leaf_instances/leaf_meshes_overview.png',np.array(mesh_sheet),meter)
 # Whole plant measurement overlay in the original camera.
 image=rgb('0000',meter);d=ImageDraw.Draw(image);base=np.array(frame['base_point']);up=np.array(frame['vertical_unit_vector']);end=base+up*float(plant['plant_height_cm'])/float(plant['scale_cm_per_unit']);line(d,[base,end],'0000','#ff004d',4)
 for p in [base,end]:
  xy,_=project(p[None],'0000');x,y=xy[0];d.ellipse((x-5,y-5,x+5,y+5),fill='#ff004d')
 overview=Image.new('RGB',(1590,1100),'#eef3ef');draw=ImageDraw.Draw(overview);text_at(draw,(25,15),'孔雀竹芋 · 2 cm 标定与器官表型实际结果',30)
 for i,(pic,label) in enumerate([(image,'株高线：茎—土基点尚待人工确认'),(Image.open(OUT/'plant_only/0000_plant_only.png'),'Plant-only：原始三角面语义子集'),(Image.open(OUT/'leaf_instances/0000_RGB_ID.png'),'12 个候选 Leaf ID，非整株叶片总数')]):
  overview.paste(pic,(i*530,65));text_at(draw,(i*530+8,1020),label,18)
 text_at(draw,(15,1060),f"尺度 {float(plant['scale_cm_per_unit']):.6f} cm/单位；株高 {float(plant['plant_height_cm']):.2f} cm；冠幅 {float(plant['canopy_width_cm']):.2f} cm。均保留复核状态。",20)
 png(OUT/'summary/phenotype_overview.png',np.array(overview),meter)

def interactive(meter,source,leaves,frame,plant):
 import plotly.graph_objects as go
 rotation=np.array(frame['measurement_rotation_rows']);base=np.array(frame['base_point']);scale=float(plant['scale_cm_per_unit']);xyz=(np.asarray(source.vertices)-base)@rotation.T*scale;f=np.asarray(source.triangles);fig=go.Figure()
 for i,leaf in enumerate(leaves):
  folder=OUT/'leaf_instances'/leaf['leaf_id']
  with np.load(io.BytesIO(meter.read(folder/'source_indices.npz'))) as a:ff=f[a['original_face_ids']]
  vi,inv=np.unique(ff,return_inverse=True);tri=inv.reshape(-1,3);v=xyz[vi];color='rgb('+','.join(map(str,PALETTE[i]))+')'
  fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=tri[:,0],j=tri[:,1],k=tri[:,2],color=color,name=leaf['leaf_id'],showlegend=True,opacity=.75,hovertemplate=leaf['leaf_id']+'<br>%{x:.2f}, %{y:.2f}, %{z:.2f} cm'))
  landmarks=read_json(folder/'landmarks.json');p=np.array(landmarks['centerline_xyz_units'])
  if len(p):
   p=(p-base)@rotation.T*scale;fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',line=dict(color='gold' if landmarks['path_valid'] else 'red',width=5),name=leaf['leaf_id']+(' observed path' if landmarks['path_valid'] else ' rejected path'),showlegend=False))
 fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[0,float(plant['plant_height_cm'])],mode='lines+markers',name='Height: provisional base',line=dict(color='red',width=7)))
 fig.update_layout(title='实际单叶三角面与曲面路径（cm）；叶片残缺，基点待复核',scene=dict(aspectmode='data',xaxis_title='Horizontal A / cm',yaxis_title='Horizontal B / cm',zaxis_title='Support normal / cm'),margin=dict(l=0,r=0,b=0,t=45))
 meter.save(OUT/'summary/measurement_3d.html',fig.to_html(include_plotlyjs=True).encode())

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--text-only',action='store_true');args=parser.parse_args()
 meter=Meter();source=read_mesh(meter) if not args.text_only else None;cal=read_json(OUT/'calibration/scale_calibration.json');frame=read_json(OUT/'calibration/reference_frame.json');catalog=read_json(OUT/'leaf_instances/leaf_instances.json');leaves=catalog['instances'];traits=rows(OUT/'phenotype/leaf_measurements.csv');plant=rows(OUT/'phenotype/plant_measurements.csv')[0];sem=read_json(OUT/'plant_only/triangle_semantics.json');video=read_json(OUT/'leaf_instances/video_geometry_validation.json');ledger=read_json(OUT/'annotations/leaf_prompt_ledger.json')
 if not args.text_only:figures(meter,source,leaves,traits,plant,frame);interactive(meter,source,leaves,frame,plant)
 review_count=sum(bool(prompt.get('requires_review')) for name in sem['views'] for prompt in read_json(OUT/'semantic_masks'/name/'corrected_complete.json')['prompts'])
 def number(value):return f'{float(value):.2f}' if value else '—'
 table='\n'.join(f"| {r['leaf_id']} | {number(r['leaf_length_cm'])} | {number(r['leaf_width_cm'])} | {number(r['leaf_area_cm2'])} | {'局部曲面、自动标志待复核' if r['leaf_length_cm'] else '曲面路径不可靠，叶长留空'} |" for r in traits)
 markdown=f'''# 孔雀竹芋：器官三维表型首轮实施记录

本轮实际运行了标定、语义分离、12 个单叶实例、原始三角面提取和厘米单位曲面测量。全部单叶均存在孔洞或碎片，当前数值描述已观测到的局部曲面，不能作为完整叶片精度结论。基部与竖直方向仍保留复核状态。真实人工配对为 0，误差指标留空。

## 冻结输入与运行范围

只处理 KongQueZhuYu。使用真实 COLMAP 的 210 个已注册相机，以及既有 GGGS 30K Original → bounded TSDF Post。源 Mesh：`{mesh_path()}`。保留其全部 264,700 顶点、484,124 面及既有相机；新模型均为原始三角面子集。没有重训，没有 C1/C2，没有启动其他 18 株。

## 2 cm 标定

RGB 中蓝块、盆和植物位于同一固定支撑板，视角绕目标变化；多视角联合静态立方体在植物 COLMAP 相机下得到低残差，支持共同坐标系假设。不是将外部静止参照套用到转台等效相机。

5 个有效标注视角：{', '.join(cal['reference_views'])}。拟合相邻边长 **{cal['reconstructed_edge_length_units']:.9f} 模型单位**，物理边长 2.0 cm，比例 **{cal['scale_cm_per_unit']:.9f} cm/模型单位**。8 个角点拟合后独立三角化，12 条相邻边均值 {cal['edge_length_units_mean']:.9f}、标准差 {cal['edge_length_units_std']:.9f} 单位。角点重投影平均误差 **{cal['reprojection_error_px']:.4f} px（530×948）**，原图近似 {cal['corner_error_px_original_approx']:.4f} px。去掉任一视角后的比例范围 {min(r['scale_cm_per_unit'] for r in cal['leave_one_view_out']):.6f}–{max(r['scale_cm_per_unit'] for r in cal['leave_one_view_out']):.6f}；这是内部稳定性检查，不是物理精度置信区间。

本次角点的数值约束来自 5 份历史人工蓝块轮廓和刚性立方体拟合，不是人工新点击的精密角点。SAM3 蓝块掩膜另存用于区域检查；其 ROI 使用了拟合几何，不能视为独立标定验证。源 TSDF 没有受支持的蓝块三角面，因此未生成虚构的 reference_cube.ply。

测量竖直为 RGB 中放在板面上的立方体底面的法向；与相机轨迹拟合轴相差 {frame['vertical_verification']['orbit_axis_angle_deg']:.2f}°。轨迹并非严格平面，盆口高处样本过少，盆口平面检查不成立，不能宣称重力方向经过独立测量。原图 0000 的 (267,452) 是智能体目视提出的茎—土连接点，尚无用户确认。参考平面为标定块底面所在支撑板，不等同于土面。变换只应用于测量坐标：`cm = scale × R × (COLMAP_xyz − base)`。

## 语义分离与实例

先做 15 个均匀方位视角，目视发现文本 flowerpot 会包含整株；随后用历史盆体几何生成框及正负点，由 SAM3 重新预测独立花盆，扩展到 {len(sem['views'])} 个视角。植株候选还受既有 RAP 目标包络约束。保存首轮文本预测、修正提示与分数；同一历史标注参与引导，其 IoU 只是引导一致性，不能报告为独立测试精度。仍有 {review_count} 处提示触发待复核；0100 的盆体与历史区域 IoU 约 0.43，不能宣称所有二维盆边界正确。源 Mesh 的叶柄/茎缺失与盆口语义误差尚未完全解决。

三角面仅接收该相机射线首个交点的可见标签，并受原有 GGGS 深度一致性约束；多视角投票、未知面的邻接一致性与多相机目标包络共同排除背景。面中心至少位于所选相机一半的画面中，避免远处碎片凭少量投影误收。没有按整株最大连通块代替语义。Plant-only **{sem['outputs']['plant_only']['faces']:,} 面**，花盆 **{sem['outputs']['pot']['faces']:,} 面**；不确定面仍留在完整源 Mesh 中。

8 个关键视角：0000、0028、0056、0084、0112、0140、0168、0196。自动候选经过智能体直接看图选择了 12 组，并以原始三角面可见重合及空间位置关联。有效存档的框/点提示 {ledger['calls']} 次，对应 79 个不同 LeafID—视角观测。初始有效运行 {ledger.get('initial_prompt_calls',ledger['calls']-ledger.get('nested_proposal_repair_calls',0))} 次提示；原候选含 6 个同叶同视角的嵌套候选，overlap/min 为 0.89–0.98，最后为较完整候选重新提示 {ledger.get('nested_proposal_repair_calls',0)} 次。重复同视角只算一次投票，初始提示记录另存归档。另一次早期运行在结束时失败，其提示输出也记入日志；包括该运行的已输出提示累计 {ledger.get('completed_prompt_outputs_including_failed_runs','见日志')} 次，界面后端另有 1 次智能体验证调用。新用户点击 0，历史语义轮廓 5 视角，智能体目视实例选择 12 次，人工手画新叶片掩膜 0。不能描述为完全无人干预的自动器官分割。

实际 SAM3 短视频传播：0084–0090，3 个实例，7 帧，{video['accepted']}/{video['total']} 个输出通过首个可见三角面的三维身份检查。该检查使用当前实例几何，是一致性检查，不能代替独立跟踪精度。输出完成后旧 reset_state 清理接口报错；已核验 21 张保存掩膜完整并修正代码。视频掩膜只用于验证，没有绕过几何验证扩张到整组相机。

每片 leaf.ply 保留源顶点坐标与法向，source_indices.npz 保存原始顶点/面 ID。像素来源由各视角掩膜和共享投影缓存中的原始 triangle_id、barycentric_uv 精确追溯。边界按原始三角面赋值，当前没有沿像素边界切分，也没有用局部重融补齐孔洞。

## 实际测量值与限制

株高 **{float(plant['plant_height_cm']):.2f} cm**；固定水平轴冠幅 **{float(plant['canopy_span_a_cm']):.2f} × {float(plant['canopy_span_b_cm']):.2f} cm**；最大水平 Feret 冠幅 **{float(plant['canopy_width_cm']):.2f} cm**。均为基点与支撑法向待复核的暂定值。

12 是选取的代表叶数量，**不是整株叶片总数**。完整叶片有效测量数目前为 **0**；整株总有效叶面积留空。12 个实例观测单侧面积之和 {float(plant['total_observed_selected_leaf_area_cm2']):.2f} cm²，不能称为全株叶面积。

| Leaf ID | 局部曲面路径 cm | 局部宽 cm | 观测单侧面积 cm² | 状态 |
|---|---:|---:|---:|---|
{table}

叶长以叶片主轴分层的几何中心为候选路径，逐段沿原始单侧网格图最短路计算；没有用端点欧氏距离冒充曲面路径。对破碎叶片只在一个连通的观测片上测路径，同时保留其余碎片，因而端点可能不是完整解剖叶基/叶尖。宽度是在主轴分层中的左右观测叶缘间求曲面图路径，不保证等同于人工中脉正交横截。面积依据叶片 PCA 参考平面对实际三角面法向分侧，取观测面积较多的一侧，排除相反一侧，未简单除以二；这避免参考视角改变时误选极小的背面碎片，但明显折叠、法向混乱、双层重叠仍会影响结果。未填补孔洞，未虚增面积。leaf_surface_coverage 是所选叶面在多视角掩膜内的可见像素覆盖，不能当作完整解剖曲面覆盖率。

## 人工配对与交互复核

历史表格仅有物种或叶片序号、虚拟派生列，无法确认同株、同叶、同一采集时刻，全部排除。manual_reference.csv 为可补录模板，需填写固定 LeafID、capture_session（确认为本组 210 帧对应植株与时刻）、measured_at、measurement_tool、operator、pair_confirmed=true。叶长沿中脉/中心线，宽度取最大有效宽，面积用扫描或面积仪，株高和冠幅沿相同基准/轴。manual_errors.py 只对已确认配对计算 MAE/RMSE/Bias/平均绝对相对误差；R² 需至少 3 个有方差的真值。当前配对表只有表头，散点图明确 n=0，误差 CSV 无伪造数值。

交互入口默认 http://127.0.0.1:8765，启动见 README.md。支持视角、固定 LeafID、框与正负点、按需 SAM3、同 ID 其他视角、叶基/叶尖/边界/整株基点复核。修正保存在 annotations/user_edits.json 和 user_masks；勾选确认后运行 apply_corrections.py 更新派生实例及 CSV。已测试 HTTP 输入校验、实际 SAM3 后端掩膜和本机 Chrome 页面渲染，并直接查看截图；验证记录见 summary/ui_validation.json。智能体测试提示和用户提示分开记录。边界点保存供复核，当前不自动切分三角面。

## 多物种迁移判断

相机/尺度变换、四语义引导、可见三角面投票、固定实例 ID、曲面图路径、配对审计可复用，物种字段可以仅作为数据标签。本次没有提供跨物种精度证据。主要失败包括遮挡导致掩膜合并、同叶多候选、接触处三维粘连、深色背面纹理不足、薄片正反面融合、叶柄或主茎缺失，以及缺失三角面导致路径中断。

厚叶需明确上表皮还是完整器官表面；薄叶需防止正反面重复；重叠叶需使用可见性和接触处实例边界；卷曲/复叶还需可配置的器官定义。这些可以作为器官/测量协议配置，不必每物种重写网络，但不能假设当前主轴中心线覆盖所有叶形。下一步需要少量人工确认的固定实例和解剖标志、必要的同类 TSDF 单叶局部融合、独立的同株实测配对、尺度与姿态不确定度传播，再做多物种批量验证。

## 来源与资源

- SAM3 官方接口： https://github.com/facebookresearch/sam3 ，本地既有 sam3.pt，无重训。
- Open3D RaycastingScene： https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html 。
- 用户指定论文 DOI 元数据已从 Crossref 核实： https://doi.org/10.1016/j.compag.2026.112467 。本次未声称复现该论文所有方法或精度。
- 博查搜索尝试返回额度不足；接口文档使用 Context7 官方仓库文档与本地源代码核实。测量数字全部来自本轮本地文件。
- GPU 仅物理 GPU1 串行；已完成阶段资源 JSON 逐阶段记录实际时间与显存，权重读入计入时间。早期失败重试另有日志，现有记录不是包含全部失败成本的总耗时。应用层大文件 I/O 每进程计量峰值 19.5 MiB/s；这不是全服务器或多进程合计吞吐，共享缓存、系统页缓存和外部进程 I/O 不属于该计量。
'''
 meter.save(OUT/'summary/conclusions.md',markdown.encode())
 headers=['Leaf ID','局部曲面路径 cm','局部宽 cm','观测单侧面积 cm²','可见覆盖','状态'];tr=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['leaf_id'],f"{float(r['leaf_length_cm']):.2f}" if r['leaf_length_cm'] else '不可用',f"{float(r['leaf_width_cm']):.2f}" if r['leaf_width_cm'] else '不可用',f"{float(r['leaf_area_cm2']):.2f}",f"{float(r['leaf_surface_coverage']):.1%}",'局部曲面，标志待复核' if r['leaf_length_cm'] else '曲面路径不可靠，叶长留空'])+'</tr>' for r in traits)
 links=''.join(f'<li><a href="../leaf_instances/{r["leaf_id"]}/leaf.ply">{r["leaf_id"]} PLY</a> · <a href="../leaf_instances/{r["leaf_id"]}/landmarks.json">标志及曲面路径</a> · <a href="../leaf_instances/{r["leaf_id"]}/instance.json">视角/完整性</a></li>' for r in traits)
 page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>孔雀竹芋 · 器官三维表型</title><style>body{{font:16px/1.65 system-ui;color:#183729;background:#f1f4f0;margin:0}}main{{max-width:1280px;margin:auto;padding:30px}}h1{{font-size:34px}}h2{{margin-top:36px}}.notice{{background:#fff1d9;padding:18px;border-left:5px solid #c5801c}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{padding:20px;background:white;border-radius:10px}}.number{{font-size:32px}}img{{width:100%;height:auto;background:white}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px;border-bottom:1px solid #d5dfd5;text-align:left}}a{{color:#126345}}iframe{{width:100%;height:680px;border:0}}code{{background:#dde7dd;padding:2px}}small{{color:#526554}}</style><main>
<h1>孔雀竹芋 · 器官三维表型</h1><p>真实 COLMAP → 既有 GGGS / bounded TSDF → SAM3 语义与单叶 → 原始三角面测量</p>
<div class="notice"><b>本轮为局部曲面链路验证。</b>12 片候选均有孔洞或碎片，完整叶片有效测量数为 0；株高基点与竖直方向仍待复核。人工配对为 0，不能报告测量精度。</div>
<div class="grid"><div class="card">2 cm 标定比例<div class="number">{cal['scale_cm_per_unit']:.6f}</div>cm / 模型单位<br>5 视角，角点残差 {cal['reprojection_error_px']:.3f} px</div><div class="card">暂定株高 / 最大冠幅<div class="number">{float(plant['plant_height_cm']):.2f} / {float(plant['canopy_width_cm']):.2f}</div>cm；基点与姿态待复核</div><div class="card">独立实例 / 人工真值配对<div class="number">12 / 0</div>8 个关键视角；{ledger['calls']} 次自动几何提示</div></div>
<p><a href="conclusions.md">完整结论与限制</a> · <a href="../phenotype/leaf_measurements.csv">逐叶 CSV</a> · <a href="../phenotype/plant_measurements.csv">整株 CSV</a> · <a href="../phenotype/manual_reference.csv">人工模板</a> · <a href="../phenotype/phenotype_errors.csv">误差 CSV</a> · <a href="http://127.0.0.1:8765">交互复核入口</a> · <a href="../README.md">运行说明</a></p>
<img src="phenotype_overview.png" alt="真实 RGB、植物网格及固定叶片身份">
<h2>标定与语义分离</h2><p>相邻边长 {cal['reconstructed_edge_length_units']:.9f} 模型单位。历史轮廓拟合角点，SAM3 区域核对；不是独立精度测试。蓝块未进入原网格，无虚构蓝块 Mesh。</p><img src="../calibration/calibration_overview.png" alt="五个标定视角投影"><p><a href="../calibration/scale_calibration.json">标定 JSON</a> · <a href="../calibration/reference_frame.json">物理坐标基准</a> · <a href="../plant_only/plant_only.ply">Plant-only PLY</a> · <a href="../plant_only/pot.ply">花盆 PLY</a> · <a href="{html.escape(os.path.relpath(mesh_path(),OUT/'summary'))}">冻结的原始 TSDF PLY</a></p><img src="../plant_only/plant_only_overview.png" alt="四方位原网格与植物子网格对比">
<h2>固定 Leaf ID 与实际曲面</h2><p>智能体直接看图选择 12 组，SAM3 框/正负点与三维可见性关联；新用户点击 0。不能称为完全无交互自动分割。下图保留网格缺口。</p><img src="../leaf_instances/leaf_instances_overview.png" alt="同一 LeafID 的四方位 RGB 与三角面"><img src="../leaf_instances/leaf_meshes_overview.png" alt="十二片独立叶片真实网格"><ul>{links}</ul>
<h2>厘米单位测量</h2><p>黄色为原始单侧网格上的中心路径，青色为曲面宽度路径。端点为局部观测片的自动候选，尚非已确认的完整解剖叶基/叶尖。可见覆盖只表示掩膜像素得到该叶片三角面解释的比例。</p><table><thead><tr>{''.join('<th>'+h+'</th>' for h in headers)}</tr></thead><tbody>{tr}</tbody></table><img src="measurement_visualization.png" alt="逐叶中心路径与横向曲面路径"><iframe src="measurement_3d.html" title="实际三维曲面、路径与厘米坐标"></iframe><p><a href="measurement_3d.html">单独打开可旋转三维视图</a>。观测单侧面积之和 {float(plant['total_observed_selected_leaf_area_cm2']):.2f} cm²，不是全株叶面积；总有效叶面积留空。</p>
<h2>短序列追踪与实测配对</h2><p>SAM3 3 叶 × 7 帧，三维身份一致性 {video['accepted']}/{video['total']}；不是独立跟踪精度。</p><img src="../leaf_instances/video_geometry_review.png" alt="七帧三维身份核验"><img src="../phenotype/paired_scatter.png" alt="当前无真实配对数据，不绘制虚构散点"><p>补录同株同叶同采集时刻参考后运行 <code>manual_errors.py</code>。当前 MAE、RMSE、Bias、相对误差、R² 均留空。</p>
<h2>多物种迁移</h2><p>相机、尺度、可见性融合、实例索引及误差审计可复用。物种可以作为数据标签；厚叶、薄叶、卷曲叶和复叶需要明确测量协议与器官边界。遮挡、粘连、薄片双面混合、缺失叶柄和孔洞仍需解决。下一步是标志确认、必要的局部同类 TSDF 融合以及独立多物种实测。</p><p><small>来源：本轮本地结果；<a href="https://github.com/facebookresearch/sam3">SAM3 官方仓库</a>、<a href="https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html">Open3D 官方文档</a>、<a href="https://doi.org/10.1016/j.compag.2026.112467">用户指定论文 DOI</a>。博查额度不足，接口以 Context7 与本地源码核实。</small></p></main></html>'''
 page=page.replace('黄色为原始单侧网格上的中心路径，青色为曲面宽度路径。','黄色为原始单侧网格上的中心路径，青色为曲面宽度路径，红色为被拒绝的候选路径。')
 page=page.replace('<h2>多物种迁移</h2>','<h2>交互复核</h2><p>已完成本机 Chrome 渲染及真实 SAM3 后端请求测试。需要确认身份、解剖标志和残缺曲面的可测范围；边界点保存供复核，当前不自动切分三角面。</p><img src="annotation_ui.png" alt="实际交互入口截图"><h2>多物种迁移</h2>')
 meter.save(OUT/'summary/index.html',page.encode());write(OUT/'summary/report_resources.json',dict(io=meter.stats()))
 print('REPORT_COMPLETE',OUT/'summary/index.html',flush=True)

if __name__=='__main__':main()
