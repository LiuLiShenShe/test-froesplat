"""Live manifest audit and offline mesh index. Pending rows never count as success."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
from batch_measure import csvwrite,csvread
from batch_manual import TRAITS
from PIL import ImageDraw,ImageFont,ImageOps
import html,colorsys,fcntl

def optional(path,default=None):return read(path) if Path(path).exists() else (default or {})
def rows(path):return csvread(path) if Path(path).exists() else []
def color(i):return np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.75,.95))*255
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def viewer(ctx,inst,mesh=None,*,catalog_href=None,organ_dir='leaf_instances'):
 folder=ctx.out/organ_dir/inst['leaf_id'];metric=inst['metric'];mesh=ctx.mesh() if mesh is None else mesh
 with np.load(io.BytesIO(ctx.meter.read(folder/'source_indices.npz'))) as d:fid=d['original_face_ids'];vi=d['original_vertex_ids']
 sub,_=subset(mesh,fid);v=np.asarray(sub.vertices);f=np.asarray(sub.triangles);colors=(np.asarray(sub.vertex_colors)*255).clip(0,255).astype(int).tolist() if sub.has_vertex_colors() else ['#64ad67']*len(v)
 payload=dict(plant=ctx.plant,leaf=inst['leaf_id'],unit='cm' if metric else 'model units',ply='leaf_cm.ply' if metric else 'leaf_units.ply',vertices=v.tolist(),faces=f.tolist(),colors=colors,source_vertex_ids=vi.tolist())
 template=(CODE/'leaf_viewer.html').read_text();text=template.replace('__PAYLOAD__',json.dumps(payload,separators=(',',':')).replace('</','<\\/')).replace('../../summary/assets/','../../../summary/assets/')
 if catalog_href is not None:text=text.replace('../../../summary/leaf_mesh_index.html',html.escape(catalog_href,quote=True))
 ctx.meter.save(folder/'viewer.html',text.encode())

def preview_view(ctx,mesh,metric):
 """Choose a shared display camera that retains the observed plant in frame."""
 data,cameras=ctx.cameras(metric);faces=np.asarray(mesh.triangles);sem=np.load(ctx.out/'plant_only/triangle_semantics.npy');indices=np.unique(faces[sem==1]);points=np.asarray(mesh.vertices)[indices]
 if len(points)>8192:points=points[np.linspace(0,len(points)-1,8192,dtype=int)]
 candidates=[];width,height=data['width'],data['height']
 for name in sorted(cameras):
  if not(ctx.out/'mesh/surface/frames'/f'{name}.npz').exists():continue
  xy,z=ctx.xyz_project(points,name,metric)
  if not len(xy):continue
  inside=(z>0)&(xy[:,0]>=4)&(xy[:,0]<width-4)&(xy[:,1]>=4)&(xy[:,1]<height-4)
  coverage=float(inside.mean());visible=xy[inside]
  extent=np.ptp(visible,axis=0) if len(visible) else np.zeros(2);area=float(extent.prod()/(width*height));candidates.append(dict(view=name,in_frame_fraction=coverage,projected_bbox_fraction=area))
 if not candidates:return sorted(cameras)[len(cameras)//4]
 best=max(r['in_frame_fraction'] for r in candidates);chosen=max((r for r in candidates if r['in_frame_fraction']>=best-.003),key=lambda r:r['projected_bbox_fraction']);write(ctx.out/'summary/preview_view.json',dict(**chosen,policy='Display only: maximize observed plant in-frame fraction, then projected size within0.3percentage points; all six panels share this actual camera. Geometry and measurements unchanged.'))
 return chosen['view']

def panels(ctx):
 info=read(ctx.out/'mesh/complete.json');metric=info['metric'];mesh=ctx.mesh();data,cameras=ctx.cameras(metric);name=preview_view(ctx,mesh,metric);ids,z,uv,scene=cast(ctx,mesh,name,metric);f=np.asarray(mesh.triangles);good=ids>=0;rgb=np.full((*ids.shape,3),245,np.uint8)
 if mesh.has_vertex_colors():
  c=np.asarray(mesh.vertex_colors)[f[ids[good]]];u=uv[good];bary=np.c_[1-u.sum(1),u];rgb[good]=(np.einsum('ni,nij->nj',bary,c)*255).clip(0,255).astype('uint8')
 else:rgb[good]=[105,165,116]
 folder=ctx.out/'summary';folder.mkdir(exist_ok=True);png(folder/'rgb.png',np.array(ctx.image(name)),ctx.meter);png(folder/'mesh.png',rgb,ctx.meter)
 sem=np.load(ctx.out/'plant_only/triangle_semantics.npy');plant=rgb.copy();keep=np.zeros(ids.shape,bool);keep[good]=sem[ids[good]]==1;plant[~keep]=245;png(folder/'plant_only.png',plant,ctx.meter)
 lid=np.load(ctx.out/'leaf_instances/triangle_leaf_ids.npy');instance=plant.copy();labels=np.zeros(ids.shape,int);labels[good]=lid[ids[good]]
 for leaf in np.unique(labels):
  if leaf:instance[labels==leaf]=color(leaf)
 png(folder/'instances.png',instance,ctx.meter);measurement=Image.fromarray(instance);draw=ImageDraw.Draw(measurement)
 for inst in read(ctx.out/'leaf_instances/leaf_instances.json')['instances']:
  landmarks=optional(ctx.out/'leaf_instances'/inst['leaf_id']/'landmarks.json');points=np.array(landmarks.get('centerline_xyz',[]))
  if len(points)>1:
   xy,zz=ctx.xyz_project(points,name,metric);projected=xy[zz>0]
   if len(projected)>1:draw.line([tuple(a) for a in projected],fill=(255,40,30),width=3)
 p=rows(ctx.out/'phenotype/plant_measurements.csv')[0];label='UNCALIBRATED' if not metric else f"Height candidate {float(p['plant_height_cm']):.2f} cm";draw.rectangle((0,0,data['width'],32),fill='white');draw.text((5,5),label,font=font(17),fill='black');png(folder/'measurement.png',np.array(measurement),ctx.meter)
 frame=ctx.out/'mesh/surface/frames'/f'{name}.npz'
 with np.load(io.BytesIO(ctx.meter.read(frame))) as d:png(folder/'gaussian.png',d['rgb'],ctx.meter)

def summarize():
 folder=OUT/'summary';folder.mkdir(parents=True,exist_ok=True)
 with (folder/'summary.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);return _summarize()

def _summarize():
 folder=OUT/'summary';folder.mkdir(parents=True,exist_ok=True);records=[];phenotypes=[];leaves=[];pairs=[];index=[]
 progress=optional(folder/'progress.json');manifest=read(MANIFEST)
 for r in manifest['plants']:
  plant=r['plant'];base=OUT/plant;status=optional(base/'status.json');stages=status.get('stages',{});sfm=optional(base/'common_input/complete.json');gggs=optional(base/'gaussian/original/complete.json');cal=optional(base/'calibration/scale_calibration.json');metric=optional(base/'gaussian/metric_conversion.json');mesh=optional(base/'mesh/complete.json');sem=optional(base/'plant_only/complete.json');inst=optional(base/'leaf_instances/complete.json');traits=optional(base/'phenotype/complete.json');validation=optional(base/'summary/validation.json');manual=optional(base/'phenotype/manual_pairing_audit.json');pp=rows(base/'phenotype/plant_measurements.csv');ll=rows(base/'phenotype/leaf_measurements.csv');phenotypes+=pp;leaves+=ll;pairs+=rows(base/'phenotype/paired_measurements.csv')
  complete=bool(validation.get('delivery_complete'));failures=[k+':'+str(v.get('error','')) for k,v in stages.items() if v.get('status')=='failed'];running=[k for k,v in stages.items() if v.get('status')=='running']
  row=dict(plant_id=plant,group=r['group'],status='validated_metric_complete' if complete and mesh.get('metric') else 'validated_uncalibrated_complete' if complete else 'failed' if failures else 'running' if running or (progress.get('plant')==plant and progress.get('status')=='running') else 'pending',true_colmap=bool(sfm),registered_views=sfm.get('registered_images',''),gggs_30k=gggs.get('iterations')==30000 or bool(gggs.get('reused_true_colmap_30k')),calibration_status=cal.get('status','pending'),calibration_mode=cal.get('mode',cal.get('calibration_mode','')),cube_edge_units=cal.get('reconstructed_edge_length_units',''),scale_cm_per_unit=cal.get('scale_cm_per_unit',''),metric_gaussian=bool(metric) and (base/'gaussian/metric_gaussians.ply').exists(),bounded_tsdf=bool(mesh),metric_mesh=bool(mesh.get('metric')),plant_only_faces=sem.get('plant_faces',''),pot_faces=sem.get('pot_faces',''),stable_leaf_count=inst.get('stable_leaf_count',''),curved_length_candidates=traits.get('length_candidates',''),confirmed_complete_leaf_measurements=traits.get('whole_leaf_valid',''),verified_manual_pairs=manual.get('verified_pairs',0),validation_passed=complete,problem='; '.join(failures) or ('Insufficient cube observations; centimetres omitted' if cal and not cal.get('scale_cm_per_unit') else ''),active_stage=','.join(running))
  row['plant_type']=plant.rstrip('0123456789')
  row['raw_views']=sfm.get('raw_images',210 if plant=='KongQueZhuYu' and sfm else '')
  row['sfm_source']=sfm.get('source','')
  row['sfm_action']=('reused_native_sfm_with_new_undistortion' if (base/'colmap/reused_native_sfm.json').exists() else 'new_native_feature_track_sfm' if sfm.get('source')==str(base/'colmap') else 'reused_existing_native_sfm') if sfm else 'pending'
  row['gggs_action']=('reused_existing_true_colmap_30k' if gggs.get('reused_true_colmap_30k') else 'trained_native_30k') if row['gggs_30k'] else 'pending'
  row['sfm_registered_fraction']=row['registered_views']/row['raw_views'] if row['raw_views'] else ''
  row['fragmented_leaf_candidates']=sum(i.get('largest_component_fraction',1)<.8 for i in inst.get('instances',[]))
  row['unassigned_plant_faces']=inst.get('unassigned_plant_faces','')
  row['calibration_rejected_views']=len(cal.get('rejected_views',[]))
  row['calibration_support_rms_px']=cal.get('boundary_support_rms_px','')
  row['semantic_unknown_faces']=sem.get('unknown_faces','')
  row['independent_cube_edges']=len(cal.get('independent_adjacent_edges',[]))
  row['independent_edge_cv']=cal['edge_length_units_std']/cal['edge_length_units_mean'] if cal.get('edge_length_units_mean') else ''
  limits=[]
  if row['sfm_registered_fraction'] and row['sfm_registered_fraction']<.9:limits.append(f"Only {row['registered_views']}/{row['raw_views']} RGB views registered/matched")
  if inst and row['fragmented_leaf_candidates']:limits.append(f"{row['fragmented_leaf_candidates']}/{row['stable_leaf_count']} leaf candidates fragmented (largest component<80%)")
  if traits and row['stable_leaf_count']:limits.append(f"{row['curved_length_candidates']}/{row['stable_leaf_count']} curved-length paths pass current checks; anatomical completeness unconfirmed")
  if limits:row['problem']='; '.join(([row['problem']] if row['problem'] else [])+limits)
  if validation.get('passed') and not complete:row['status']='processed_with_limits';row['problem']='; '.join(validation.get('quality_limits',[]))
  records.append(row);cards=[]
  for item in inst.get('instances',[]):
   leaf=item['leaf_id'];path=f'../{plant}/leaf_instances/{leaf}';cards.append(f'<a class="leaf" href="{path}/viewer.html"><img loading="lazy" src="{path}/reference.png"><b>{leaf}</b> · {item["faces"]} faces · {len(item["views"])} views · {item["unit"]}</a> <a href="{path}/leaf_{"cm" if item["metric"] else "units"}.ply" download>PLY</a>')
  models=[]
  for relative,label in [('gaussian/metric_gaussians.ply','厘米 GGGS'),('mesh/mesh_cm.ply','厘米整株场景 Mesh'),('plant_only/plant_only_cm.ply','厘米 Plant-only'),('plant_only/pot_cm.ply','厘米花盆'),('mesh/mesh_units.ply','未标定场景 Mesh'),('plant_only/plant_only_units.ply','未标定 Plant-only'),('plant_only/pot_units.ply','未标定花盆'),('calibration/scale_calibration.json','蓝块尺度记录')]:
   if (base/relative).exists():models.append(f'<a href="../{plant}/{relative}" download>{label}</a>')
  index.append('<section><h2>'+html.escape(plant)+' · '+r['group']+'</h2><p>'+html.escape(row['status'])+'; stable leaves '+str(row['stable_leaf_count'])+'; '+html.escape(str(row['calibration_status']))+'</p><p>'+ ' · '.join(models)+'</p><a href="../'+plant+'/phenotype/manual_reference.csv" download>人工参考表</a> <a href="../'+plant+'/phenotype/plant_measurements.csv">整株测量</a><div class="grid">'+''.join(cards)+'</div></section>')
 csvwrite(folder/'all19_results.csv',records);csvwrite(folder/'phenotype_summary.csv',phenotypes,fields=None if phenotypes else ['plant_id','group','plant_height_cm','canopy_span_a_cm','canopy_span_b_cm','canopy_width_cm']);csvwrite(folder/'all_leaf_measurements.csv',leaves,fields=None if leaves else ['plant_id','leaf_id','leaf_length_cm','leaf_width_cm','leaf_area_cm2']);csvwrite(folder/'all_paired_measurements.csv',pairs,fields=None if pairs else ['plant_id','leaf_id','trait','automatic','manual','error'])
 kinds=[]
 for kind in dict.fromkeys(r['plant_type'] for r in records):
  selected=[r for r in records if r['plant_type']==kind];kinds.append(dict(plant_type=kind,label_basis='Manifest directory-name group; not independently taxonomically verified',plants=len(selected),metric_meshes=sum(r['metric_mesh'] for r in selected),validated_complete=sum(r['validation_passed'] for r in selected),stable_leaf_candidates=sum(r['stable_leaf_count'] or 0 for r in selected),curved_length_candidates=sum(r['curved_length_candidates'] or 0 for r in selected),fragmented_leaf_candidates=sum(r['fragmented_leaf_candidates'] for r in selected),confirmed_complete_leaf_measurements=sum(r['confirmed_complete_leaf_measurements'] or 0 for r in selected),problems='; '.join(r['plant_id']+': '+r['problem'] for r in selected if r['problem'])))
 csvwrite(folder/'plant_type_summary.csv',kinds)
 errorrows=[]
 for trait in TRAITS:
  pp=[p for p in pairs if p['trait']==trait];a=np.array([float(p['automatic']) for p in pp]);b=np.array([float(p['manual']) for p in pp]);e=a-b;ss=float(np.sum((b-b.mean())**2)) if len(b) else 0
  errorrows.append(dict(trait=trait,n_pairs=len(pp),MAE=float(np.mean(abs(e))) if len(e) else '',RMSE=float(np.sqrt(np.mean(e**2))) if len(e) else '',R2=float(1-np.sum(e**2)/ss) if len(e)>=2 and ss else '',status='PROVISIONAL_GEOMETRY_LIMITS_APPLY' if len(e) else 'NO_VERIFIED_MANUAL_PAIRS'))
 csvwrite(folder/'phenotype_errors.csv',errorrows)
 totals=dict(manifest_plants=len(records),manifest_plant_types=len(kinds),true_colmap=sum(r['true_colmap'] for r in records),gggs_30k=sum(r['gggs_30k'] for r in records),metric_gaussians=sum(r['metric_gaussian'] for r in records),metric_meshes=sum(r['metric_mesh'] for r in records),validated_complete=sum(r['validation_passed'] for r in records),stable_leaves=sum(r['stable_leaf_count'] or 0 for r in records),curved_length_candidates=sum(r['curved_length_candidates'] or 0 for r in records),confirmed_complete_leaves=sum(r['confirmed_complete_leaf_measurements'] or 0 for r in records),verified_manual_pairs=len(pairs),updated_unix=time.time());write(folder/'totals.json',totals)
 page='<!doctype html><meta charset="utf-8"><title>ForeSplat 全株单叶索引</title><style>body{font:16px system-ui;background:#f4f7f6;color:#16312b;margin:30px}.grid{display:flex;flex-wrap:wrap;gap:14px}.leaf{display:grid;width:160px;padding:8px;background:white}.leaf img{height:200px;max-width:155px;object-fit:contain}section{border-top:1px solid #ccd8d2;padding:12px}</style><h1>ForeSplat · 19 株公制重建与独立叶片</h1><p>已验证 '+str(totals['validated_complete'])+'/19 株；公制 Mesh '+str(totals['metric_meshes'])+'；独立叶片 '+str(totals['stable_leaves'])+'。状态随实际运行更新。</p><p>可见叶面保留原始孔洞，不代表完整器官。选点工具在每片叶子的三维页面中。未标定模型不提供厘米值。</p><p><a href="/match">57 片标定叶片匹配</a> · <a href="three_leaf_match_overview.html">57 片进度</a> · <a href="all19_results.csv">全流程状态</a> · <a href="all_leaf_measurements.csv">全部叶片测量</a> · <a href="phenotype_summary.csv">整株表型</a> · <a href="phenotype_errors.csv">人工配对误差</a> · <a href="/review">人工参考录入</a></p>'+''.join(index)
 (folder/'leaf_mesh_index.html').write_text(page)
 # Evidence panels only; unfinished cells are explicitly pending.
 cellw,cellh=180,245;canvas=Image.new('RGB',(220+6*cellw,70+len(records)*cellh),'white');draw=ImageDraw.Draw(canvas);head=['RGB','Gaussian','Mesh','Plant-only','Leaf instances','Measurement']
 for j,title in enumerate(head):draw.text((225+j*cellw,20),title,font=font(16),fill='black')
 for i,r in enumerate(records):
  y=65+i*cellh;draw.text((8,y+5),r['plant_id'],font=font(15),fill='black');draw.text((8,y+30),r['group'],font=font(13),fill='black');draw.text((8,y+52),r['status'].replace('_','\n'),font=font(12),fill='#52655f')
  for j,key in enumerate(['rgb','gaussian','mesh','plant_only','instances','measurement']):
   p=OUT/r['plant_id']/'summary'/f'{key}.png'
   if p.exists():
    im=Image.open(p).convert('RGB');im=ImageOps.contain(im,(cellw-6,cellh-8));canvas.paste(im,(220+j*cellw+(cellw-im.width)//2,y))
   elif key=='rgb':
    preview=folder/'input_previews'/f"{r['plant_id']}.png"
    if not preview.exists():
     candidates=sorted((ROOT/'02-FFT'/r['plant_id']).glob('*.jpg'))
     if candidates:
      meter=Meter();im=ImageOps.contain(Image.open(io.BytesIO(meter.read(candidates[0]))).convert('RGB'),(cellw-6,cellh-8));png(preview,np.array(im),meter)
    if preview.exists():
     im=Image.open(preview);canvas.paste(im,(220+j*cellw+(cellw-im.width)//2,y))
   else:draw.text((230+j*cellw,y+100),'PENDING',font=font(13),fill='#7f8b85')
 png(folder/'all19_overview.png',np.array(canvas),Meter())
 lines=['# ForeSplat 19 株实际运行报告','',f"当前已验证全流程 {totals['validated_complete']}/19 株；公制 Gaussian {totals['metric_gaussians']} 株；公制 boundedTSDF {totals['metric_meshes']} 株。",f"独立稳定叶片 {totals['stable_leaves']} 片；可用曲面长度候选 {totals['curved_length_candidates']} 片；经确认完整叶片测量 {totals['confirmed_complete_leaves']} 片；真实人工配对 {totals['verified_manual_pairs']} 条。",'','真实 COLMAP 模型按原始输入 SHA256 对应相机与 RGB；历史黑背景 PNG 则逐像素核对全部保留的非黑 RGB。无法核验的视角排除，记录输入来源；其余18株不复用 Omega/VGGT 训练模型。固定原生 GGGS30K/SH3/SG0 与已选 boundedTSDF。公制只缩放一次，世界轴保持不变。','蓝块由独立 SAM3 检测进行三维多视角联合拟合；历史标注辅助另行标识。缺少尺度的植株保留模型单位，不填写厘米值。','可见独立叶片不等同完整生物学叶片。未经确认的叶基、叶尖、竖直方向和土壤基准仍标记为候选；开放边界和不连通曲面不补造。','', '| Plant | Group | Status | COLMAP views | Calibration | Scale cm/u | Leaves | Curved paths | Problem |','|---|---|---|---:|---|---:|---:|---:|---|']
 for r in records:lines.append('| '+' | '.join(str(r[k]).replace('|','/') for k in ['plant_id','group','status','registered_views','calibration_status','scale_cm_per_unit','stable_leaf_count','curved_length_candidates','problem'])+' |')
 lines+=['','按目录名称分组共 '+str(len(kinds))+' 类植物；这不是19个独立物种的验证。分类汇总见 plant_type_summary.csv。','', '| Plant type | Plants | Metric mesh | Completed | Stable candidates | Curved paths | Fragmented candidates |','|---|---:|---:|---:|---:|---:|---:|']
 for r in kinds:lines.append('| '+' | '.join(str(r[k]) for k in ['plant_type','plants','metric_meshes','validated_complete','stable_leaf_candidates','curved_length_candidates','fragmented_leaf_candidates'])+' |')
 lines+=['','跨物种能力结论以实际完成数、尺度质量与几何范围为限；运行中不宣称全部成功。器官完整性和准确度需真实人工配对验证。','功能颗粒度参照 GC-3DGS-Pheno：https://doi.org/10.1016/j.compag.2026.112467；本项目使用既定 COLMAP + GGGS + boundedTSDF + SAM3 路线，未宣称复现论文算法。','数据来源：本地 manifest、逐阶段日志、原始模型与本次输出。网络搜索服务本次返回额度不足，未借用搜索摘要证明实验结果。']
 (folder/'conclusions.md').write_text('\n'.join(lines)+'\n');return totals

def main(ctx):
 assets=OUT/'summary/assets';assets.mkdir(parents=True,exist_ok=True)
 if not(assets/'plotly.min.js').exists():
  from plotly.offline import get_plotlyjs
  (assets/'plotly.min.js').write_text(get_plotlyjs())
 catalog=read(ctx.out/'leaf_instances/leaf_instances.json')
 mesh=ctx.mesh()
 for inst in catalog['instances']:viewer(ctx,inst,mesh)
 panels(ctx);write(ctx.out/'summary/report_complete.json',dict(status='complete',leaf_viewers=len(catalog['instances'])));summarize()

if __name__=='__main__':
 if '--summary-only' in sys.argv:summarize()
 else:main(Context(sys.argv[1]))
