"""Assemble measured results, immutable lineage and reviewable artifact indices."""
from common import *
from plyfile import PlyData
import csv, hashlib, difflib
from PIL import Image,ImageDraw

def rows(path):return list(csv.DictReader(Path(path).open()))
def means(data):
    out={}
    for k in data[0]:
        try:out[k]=float(np.mean([float(x[k]) for x in data]))
        except (ValueError,TypeError):pass
    return out
def csvout(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in data for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def md(path,text):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.strip()+'\n')
def table(keys,data):
    def val(x):return f'{x:.6f}' if isinstance(x,float) else str(x)
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(val(r.get(k,'')) for k in keys)+' |' for r in data)
def header(path):
    with path.open('rb') as stream:
        lines=[]
        while True:
            line=stream.readline();lines.append(line.decode('ascii').strip())
            if line==b'end_header\n':break
            if not line:raise ValueError(path)
    vertices=next(int(x.split()[-1]) for x in lines if x.startswith('element vertex'))
    faces=next((int(x.split()[-1]) for x in lines if x.startswith('element face')),0)
    return vertices,faces,lines

def main():
    best=ROOT/'task0_gaussian_cleaning/best_cleaned';g1=ROOT/'task2_cleaned_gs2mesh/G1_P2_to_GS2Mesh';g2=ROOT/'task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh';repair=g2/'local_repair'
    assert json.loads((repair/'unified_evaluation/complete.json').read_text())['views_per_stage']==208
    assert (ROOT/'task0_gaussian_cleaning/diagnosis/E10_preview/preview_complete.json').exists()
    ref=ROOT/'task2_cleaned_gs2mesh/G0_original_reference';ref.mkdir(exist_ok=True)
    for name,path in [('original_gaussians.ply',SOURCE),('raw.ply',SEL/'P3_GS2Mesh/meshes_shared_depth_range/raw.ply'),('post.ply',SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply')]:
        if not (ref/name).exists():(ref/name).symlink_to(path)
    write(ref/'REFERENCE.json',dict(source='Historical completed E10 -> P3_shared, no new extraction',gaussian=str(SOURCE),gaussian_sha256=sha(SOURCE),post_sha256=sha(ref/'post.ply'),original_report=str(SEL/'P3_GS2Mesh/RUN_REPORT.md')))
    legacyg=rows(SEL/'comparison/ALL_GAUSSIAN_EXPERIMENTS.csv');legacym=rows(SEL/'comparison/ALL_MESH_EXPERIMENTS.csv')
    previews=[('E10',ROOT/'task0_gaussian_cleaning/diagnosis/E10_preview'),('P2',ROOT/'task0_gaussian_cleaning/diagnosis/P2_preview'),('Round1',ROOT/'task0_gaussian_cleaning/round_01/preview'),('Round2',ROOT/'task0_gaussian_cleaning/round_02/preview'),('Round3_shrink',ROOT/'task0_gaussian_cleaning/round_03/preview'),('Aggressive',ROOT/'task0_gaussian_cleaning/additional_rounds/aggressive_delete/preview'),('Round4',ROOT/'task0_gaussian_cleaning/additional_rounds/round_04/preview'),('Round5_historical',ROOT/'task0_gaussian_cleaning/additional_rounds/round_05/preview'),('Best_PlantPot',best/'preview')]
    gauss=[];free=[]
    for name,folder in previews:
        info=json.loads((folder/'preview_complete.json').read_text());m=means(rows(folder/'training_metrics.csv'));fr=rows(folder/'free_metrics.csv');free.extend([dict(model=name,**x) for x in fr])
        gauss.append(dict(model=name,scope='8 representative source + 8 fixed free views',gaussians=info['gaussian_count'],ply=info['ply'],sha256=info['sha256'],**m,free_black_pixels_sum=sum(int(x['black_cloud_px']) for x in fr),free_highalpha_pixels_sum=sum(int(x['bg_highalpha_px']) for x in fr),frozen_cloud_remaining_px_sum=sum(int(x['frozen_cloud_black_remaining_px']) for x in fr)))
    for name in ['E10','P2']:
        row=next(r for r in legacyg if r['experiment_id']==name)
        gauss.append(dict(model=name,scope='208 historical frozen source cameras',gaussians=int(row['num_gaussians']),full_psnr=float(row['psnr_raw']),fg_psnr=float(row['fg_psnr']),ssim=float(row['ssim']),lpips=float(row['lpips']),bg_alpha=float(row['background_alpha']),fg_alpha=float(row['foreground_alpha']),bg_alpha_gt05=float(row['background_alpha_gt05_fraction']),source=row['metric_source']))
    for name,folder in [('Round5_historical',best/'historical_208'),('Best_PlantPot',best/'scene_208')]:
        data=rows(folder/'per_view_metrics.csv');assert len(data)==208;m=means(data)
        gauss.append(dict(model=name,scope='208 frozen source cameras, original RAP including marker',gaussians=int(m['gaussian_count']),full_psnr=m['full_psnr'],fg_psnr=m['fg_mask_psnr'],ssim=m['ssim'],lpips=m['lpips_alex'],bg_alpha=m['background_alpha_mean'],fg_alpha=m['foreground_alpha_mean'],bg_alpha_gt05=m['background_alpha_gt05_fraction'],silhouette_iou=m['alpha_mask05_iou'],source=str(folder/'per_view_metrics.csv')))
    csvout(ROOT/'comparisons/gaussian_cleaning_metrics.csv',gauss);csvout(ROOT/'comparisons/free_view_images/gaussian_free_metrics.csv',free)
    meshrows=[]
    for ident in ['M4_E10_GGGS_TSDF','P3_shared']:
        for stage in ['raw','post']:
            row=next(r for r in legacym if r['mesh_id']==ident and r['mesh_stage']==stage)
            meshrows.append(dict(model='G0' if ident=='P3_shared' else 'M4_legacy',stage=stage,views=208,vertices=int(row['vertices']),triangles=int(row['faces']),components=int(row['components']),iou=float(row['silhouette_iou']),boundary_f1=float(row['boundary_f1']),boundary_chamfer_px=float(row['boundary_chamfer_px']),fg_coverage=float(row['foreground_coverage']),bg_geometry=float(row['background_geometry_ratio']),hole_px=float(row['hole_area']),source=row['metric_source']))
    for name,folder,stages in [('G1',g1/'meshes_shared_depth_range/unified_evaluation',['raw','post']),('G2',g2/'meshes_shared_depth_range/unified_evaluation',['raw','post']),('P3_local',ROOT/'task1_p3_local_repair/mesh_versions/unified_evaluation',['v1','v2']),('G2_targeted',repair/'unified_evaluation',['post'])]:
        structural=rows(folder/'structure.csv')
        for stage in stages:
            data=rows(folder/(stage+'_per_view.csv'));assert len(data)==208;m=means(data);s=next(x for x in structural if x['stage']==stage)
            meshrows.append(dict(model=name,stage=stage,views=208,vertices=int(s['vertices']),triangles=int(s['faces']),components=int(s['components']),iou=m['silhouette_iou'],boundary_f1=m['boundary_fscore_2px'],boundary_chamfer_px=m['boundary_chamfer_px'],fg_coverage=m['foreground_coverage'],bg_geometry=m['background_geometry_fraction'],hole_px=m['hole_area_px'],surface_roughness=float(s['surface_roughness']),source=str(folder/(stage+'_per_view.csv'))))
    csvout(ROOT/'comparisons/mesh_artifact_metrics.csv',meshrows)
    local=[]
    for name,folder in [('P3_v1',ROOT/'task1_p3_local_repair/source157/v1_views'),('P3_v2',ROOT/'task1_p3_local_repair/source157/v2_views'),('G1',g1/'visualizations'),('G2',g2/'visualizations'),('G2_targeted',repair/'visualizations')]:
        local.extend([dict(model=name,**x) for x in json.loads((folder/'views_complete.json').read_text())['records']])
    csvout(ROOT/'comparisons/local_mesh_roi_metrics.csv',local)
    # Two fixed views, six versions and four native render channels in identical poses/background.
    chosen=[previews[i] for i in [0,1,2,3,4,8]]
    for view,kind in [('free_0053_-15','free'),('0157','training')]:
        atlas=Image.new('RGB',(6*270,4*510),'white');draw=ImageDraw.Draw(atlas)
        for j,(name,folder) in enumerate(chosen):
            for i,channel in enumerate(['RGB','Alpha','Depth','Normal']):
                im=Image.open(folder/kind/view/(channel+'.png'));im.thumbnail((270,480));x=j*270;y=i*510;atlas.paste(im,(x,y+25));draw.text((x+5,y+5),name+' '+channel,fill='black')
        atlas.save(ROOT/'comparisons/before_after_images'/(kind+'_'+view+'_all_rounds.png'))
    # Source provenance, actual fresh caches and fixed official source integrity.
    provenance=json.loads((SEL/'P3_GS2Mesh/PROVENANCE.json').read_text());official=SEL/'P3_GS2Mesh/official';verified={rel:sha(official/rel)==expected for rel,expected in provenance['source_sha256'].items()};assert all(verified.values())
    stages=[]
    for folder,count in [(g1,544074),(g2,510248)]:
        render=json.loads((folder/'render_complete.json').read_text());stereo=json.loads((folder/'stereo_complete.json').read_text());fusion=json.loads((folder/'fusion_shared_depth_range_complete.json').read_text());assert render['model_count']==count and len(render['records'])==len(stereo['records'])==208
        for x in render['records']:assert x['gaussian_sha256']==render['gaussian_sha256'] and x['gaussian_count']==count
        for x in stereo['records']:assert x['gaussian_sha256']==render['gaussian_sha256']
        assert abs(fusion['parameters']['TSDF_voxel']/512-.0017389331048768373)<1e-12
        stages.append(dict(experiment=folder.name,gaussian_sha256=render['gaussian_sha256'],gaussian_count=count,left_right_pairs=208,fresh_stereo_records=208,depth_trunc=fusion['depth_trunc'],stereo_iterations=stereo['native_valid_iters'],render_seconds=render['wall_seconds'],stereo_seconds=stereo['wall_seconds'],fusion_seconds=fusion['wall_seconds']))
    assert stages[0]['gaussian_sha256']!=stages[1]['gaussian_sha256']
    write(ROOT/'PROVENANCE_VERIFICATION.json',dict(status='passed',official_core_unchanged=verified,stages=stages,manager='uv',gpu=1,original_model_untouched_sha256=dict(E10=sha(SOURCE),P2=sha(P2),P3=sha(SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply')),scope='No VGGT/SAM inference, no GGGS training, no new model representation',search='Bocha HTTP403 quota unavailable. Official project and GS2Mesh source retrieved directly; ctx7 GS2Mesh has no indexed match, uv current docs fetched. All quality conclusions measured locally.'))
    for old,new,label in [('render_e10_stereo.py','render_cleaned_stereo.py','render_cleaned_stereo'),('run_stereo_fusion.py','run_cleaned_stereo_fusion.py','run_cleaned_stereo_fusion')]:
        text=''.join(difflib.unified_diff((SEL/'P3_GS2Mesh/scripts'/old).read_text().splitlines(True),(ROOT/'scripts'/new).read_text().splitlines(True),fromfile='original/'+old,tofile='refinement/'+new));(ROOT/'source_diffs'/(label+'.diff')).write_text(text)
    md(ROOT/'source_diffs/cleaning_pipeline.diff',''.join(difflib.unified_diff((SEL/'scripts/clean_gggs.py').read_text().splitlines(True),(ROOT/'scripts/build_rounds.py').read_text().splitlines(True),fromfile='original/clean_gggs.py',tofile='independent/build_rounds.py')))
    write(ROOT/'source_diffs/SOURCE_MANIFEST.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*.py')})
    inventory=[];paths=[SOURCE,P2,SEL/'P2_GGGS_Cleaning/removed_gaussians.ply',SEL/'P3_GS2Mesh/meshes_shared_depth_range/raw.ply',SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply']
    for base,dirs,files in os.walk(ROOT):
        dirs[:]=[x for x in dirs if x not in ['.venv','runtime_checkpoint','official','stereo','__pycache__']]
        paths.extend(Path(base)/name for name in files if name.endswith('.ply'))
    cache={}
    for p in sorted(set(paths)):
        v,f,h=header(p);resolved=p.resolve()
        if resolved not in cache:cache[resolved]=sha(p)
        inventory.append(dict(kind='Mesh' if f else 'Gaussian',path=str(p.absolute()),resolved_path=str(resolved),vertices=v,triangles=f,sha256=cache[resolved],bytes=p.stat().st_size,viewer='Open3D / Blender / MeshLab' if f else 'Native GGGS; baked standard fields for SuperSplat where labeled',status='Exploratory' if 'plant_only' in p.name else 'Evidence/candidate' if 'contributors' in p.name or 'candidate' in p.name else 'Preserved version'))
    csvout(ROOT/'artifact_index/ALL_ARTIFACTS.csv',inventory)
    for kind,file in [('Gaussian','ALL_GAUSSIAN_FILES.md'),('Mesh','ALL_MESH_FILES.md')]:
        items=[r for r in inventory if r['kind']==kind]
        md(ROOT/'artifact_index'/file,'# '+kind+' absolute paths\n\n'+table(['path','vertices','triangles','sha256','status'],items)+'\n\nNative coordinates are unchanged. Aliases list both real resolved path and SHA in ALL_ARTIFACTS.csv.')
    selected=best/'cleaned_gaussians_supersplat.ply';selectedmesh=repair/'G2_targeted_cleaned_mesh.ply'
    md(ROOT/'artifact_index/HOW_TO_VIEW.md',f'''# 最终查看文件

SuperSplat 打开：`{selected}`（510248 Gaussian，filter_3D 已烘焙到标准 scale/opacity）。

Blender / MeshLab / Open3D 打开：`{selectedmesh}`（1492364 triangles，G2 官方融合后另行移除17449个非目标标记三角面）。

原生 GGGS Renderer 和今后 GS2Mesh 输入使用：`{best/'cleaned_gaussians.ply'}`。不要把未烘焙的原生 filter_3D 文件与忽略该字段的查看器画面混为一谈。

所有模型保持修正后的标准 VGGT / E10 OpenCV 世界坐标，−Y 近似为植株向上，尺度未作物理标定。查看器需要向上轴时可仅调整对象显示旋转：Y-up 软件绕X轴180°；Blender Z-up 可绕X轴−90°。不要将查看旋转写回分析模型或沿用旋转后的 Triangle ID。

本次使用原生 GGGS Rasterizer 和 Open3D Raycast 生成真实 RGB/Alpha/Depth/Normal/三角面ID，并由当前模型直接读取比较图片。未启动交互式 SuperSplat、Blender 或 MeshLab GUI。Gaussian PLY 原生字段已加载验证；标准兼容版本与原生有效尺度、不透明度及四视角渲染核对，RGB/Alpha最大差小于1/255（非位级相同）。Mesh 使用 Open3D 实际读取和逐像素 raycast。GUI 自由拖动仍应视为用户最后的视觉验收。

Plant-only 试验文件：`{best/'plant_only_geometry_v2.ply'}`。它已去除花盆主体，但盆沿/基部混合Gaussian和微小标记残留未消失，根与叶基完整性未认证；不用于表型测量。v1裁切保留作失败对照。

同姿态逐轮图：`{ROOT/'comparisons/before_after_images/free_free_0053_-15_all_rounds.png'}`；每轮完整8个冻结自由视角位于该版本 preview/free。Mesh 8自由视角：`{ROOT/'comparisons/mesh_comparison'}`。局部灰色/法线/线框/颜色/组件图和精确三角面ID位于各版本 visualizations；ID在 raycast.npz。
''')
    visual=[('task0_gaussian_cleaning/diagnosis/baseline_free_views.png','P2 retains broad upper/left black clouds'),('task0_gaussian_cleaning/round_02/preview/free_contact.png','Round2 clouds remain large'),('task0_gaussian_cleaning/round_03/preview/free_contact.png','Shrink reduces some cloud, insufficient'),('task0_gaussian_cleaning/additional_rounds/round_04/preview/free_contact.png','Large clouds greatly reduced; streaks remain'),('task0_gaussian_cleaning/additional_rounds/round_05/preview/free_contact.png','Further cloud reduction; leaf edge streaks remain'),('task0_gaussian_cleaning/best_cleaned/preview/free_contact.png','Marker body removed; tiny blue remnant remains'),('comparisons/free_view_images/extended/comparison.png','Far/top/side/oblique improvement'),('comparisons/before_after_images/worst_depth_change_0037.png','Depth/RGB change concentrated at pot rim, brighter incorrect texture'),('comparisons/before_after_images/worst_depth_change_0038.png','Same pot regression, no broad missing leaf in this view'),('task0_gaussian_cleaning/diagnosis/gaussian_spatial_distribution.png','Deleted large peripheral center clusters; residual near leaf boundaries'),('task0_gaussian_cleaning/diagnosis/projected_ellipse_overlay.png','Large mixed/background analytic footprints; native contributor IDs localized'),('task0_gaussian_cleaning/best_cleaned/plant_only_views/free_0000_-15/RGB.png','v1 crop fails to remove much pot'),('task0_gaussian_cleaning/best_cleaned/plant_only_v2_views/free_0000_-15/RGB.png','v2 removes pot body; pot rim/base and tiny marker remain'),('task1_p3_local_repair/source157/v1_views/0157/local_panel.png','Removed ring while retaining tip'),('task1_p3_local_repair/wrong_connections/candidate_review.png','Some unsupported slender faces overlap actual stalk/curled leaf, blanket cut unsafe'),('task2_cleaned_gs2mesh/G1_P2_to_GS2Mesh/visualizations/0157/local_panel.png','G1 ring persists'),('task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/visualizations/0157/local_panel.png','G2 ring disappears without local edit; tip preserved'),('task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/visualizations/0131/local_panel.png','Local missing surface remains; dark appearance not automatically hole'),('comparisons/mesh_comparison/G2_free_contact.png','Stereo regenerates detached marker-area surfaces'),('comparisons/mesh_comparison/G2_targeted_free_contact.png','Matched marker components removed, other detached fragments remain'),('task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/local_repair/visualizations/0157/local_panel.png','Tip/ring result remains after marker-only repair')]
    assert all((ROOT/p).exists() for p,o in visual)
    write(ROOT/'comparisons/VISUAL_REVIEW.json',dict(method='Current model directly viewed local rendered images; no external vision API, no interactive GUI session claimed',records=[dict(path=str(ROOT/p),observation=o,actually_read=True) for p,o in visual],overall='Improved, not yet usable clean 3D asset'))
    brief=table(['model','stage','iou','boundary_f1','boundary_chamfer_px','fg_coverage','bg_geometry','hole_px'],[r for r in meshrows if r['stage'] in ['post','v1','v2']])
    rounds=table(['model','gaussians','free_black_pixels_sum','free_highalpha_pixels_sum'],gauss[:len(previews)])
    photo=table(['model','full_psnr','fg_psnr','ssim','lpips','bg_alpha','fg_alpha'],[r for r in gauss if r['scope'].startswith('208')])
    diagnosis=f'''# GGGS 伪影诊断（真实模型与渲染证据）

结论：原P2未完成清理。中心投影/小范围尺度保护和KDTree邻近前景保护，使少量前景支持的背景大体元保留；前层去除后还会露出原先被遮挡的黑层。故仅一次原始模型贡献评分也不足。

原P2保留544074/576396，保护517673，异常尖刺仅删除5。旧保护点中8312只有≤1视角前景中心支持，42293有≥15视角背景中心支持。保护是候选关系，不是真实表面认证。

新证据以原生GGGS color Jacobian统计每Gaussian的 T×alpha，分别累加Mask内、边缘外背景、黑云ROI。模型参数detach，无优化或训练；各视角贡献总和与原生Alpha独立校验（相对误差低于0.03%）。完整208训练视角加8历史自由相机。高斯ID始终指原E10行号。

投影椭圆几何诊断、filtered scale、opacity、旋转、SH字段、实际raster radius/visible contributions均保留。`projected_ellipse_diagnostic.npz`为协方差一阶投影，不复现screen clamp/low-pass，不拿它代替真实Rasterizer贡献。覆盖大小本身不等同废点。

黑云不是只由5个尖刺组成：原E10测得黑云权重50%由2558个ID贡献，90%由15152个，99%由28835个。top250占21.60%，包括455628、569226、317467、575871等；完整ID/XYZ在dominant_cloud_gaussian_ids.json。P2保留的16510个 harm>.95 候选承载94.98%的原模型测得剩余黑权重，同时只承载0.5957%的前景贡献。后续需要重新归因可见层。

六个固定定位区和来源相机/ID/AABB在artifact_regions.json；95%贡献模型对应*_contributors.ply。叶片中心和盆边ROI同时含有效结构，未按ROI整片删除。每个Gaussian的原生所有字段及多视角贡献在gaussian_artifact_evidence.csv（576396行）；最近VGGT前景PointMap、GGGS可见采样表面、P3距离和k8密度在geometric_auxiliary_evidence.csv。VGGT只是辅助参考，不作为叶片真值。

实际3D残余贡献重新归因后，4011个Gaussian的黑ROI权重>.05；96.27%残余黑权重处于 harm≤.95 的混合贡献点上。示例ID186547、359655、406752、382988、470521、491751，完整列表及每视角关系在 best_cleaned/remaining_artifact_dependencies.csv。继续整点删除会切掉已测前景贡献，不能保证消掉伪影同时保住叶缘。

盆沿变亮回归已定位到3135、290315、316725、3627、124520等真实ID，六视角的95%贡献AABB、像素数量和原始参数在POT_BRIGHT_REGRESSION_DIAGNOSIS.json及pot_bright_artifact_dependencies.csv。source37变化区RGB MSE从0.0639升到0.1353，source38从0.0540升到0.1742；这是清理露出错误颜色，不能写作无副作用。

实际SH方向颜色加真实alpha贡献：8自由视角E10深色体元权重3333370.53→Best54136.36（98.38%下降），深色贡献Gaussian45729→5884；RGB阈值为Rec709 luminance<.12，所有视角并集统计，不是唯一废点认证。

`SPATIAL_COMPONENT_METRICS.json`另外报告原测 harm>.95 候选的半径.015连通图和AABB体积：P2有16510候选点/9个≥10点簇/37孤立中心，Best此旧候选集合为0。该0不表示真正所有漂浮点为0，新露出的混合点仍存在。

源代码：原脚本 `{SEL/'scripts/clean_gggs.py'}`；独立诊断/编辑脚本 `{ROOT/'scripts'}`；所有原始文件未覆盖。
'''
    md(ROOT/'task0_gaussian_cleaning/diagnosis/GAUSSIAN_ARTIFACT_DIAGNOSIS.md',diagnosis)
    md(ROOT/'task0_gaussian_cleaning/CLEANING_ANALYSIS.md',f'''# Gaussian 多轮清理与验收

状态 **Improved**。大面积黑云实质减少，仍有叶外细黑条、叶片交叠黑层、盆沿颜色错误和极小标记残留；没有宣称完全干净。

{rounds}

free_black_pixels_sum是8个固定1080×1920相机中、原P3轮廓外>5px、Alpha>.5、RGB亮度<.12的像素总和。各视角可重叠，不能作为3D物理面积；原P3轮廓只是保守参考。P2→Best 2682781→68362，下降97.45%。各视角/固定黑云ROI的局部Alpha和覆盖在comparisons/free_view_images/gaussian_free_metrics.csv。另4个远距/俯视/侧面/斜视从原P2[65246,243954,312081,171456]→R5[1440,4184,2107,2087]；这组仍含历史标记，独立标注。

Round1保守：原P2基础上实际背景危害比>.99；Round2中等：>.95加空间孤立/needle支撑；高anisotropy使用最大/次大尺度区别针状与薄叶，不按最大/最小比整批删叶。Round3对4831个混合大Gaussian试验原始sigma×.5；filter_3D不变，因此有效opacity系数也受影响，逐点旧/新值已存，收益不足未选。Aggressive(.8)保留作更激进对照，不凭删除数量选中。Round4/5重算当前模型新露出黑层，再删12505/1208；最终保留513840。另独立语义去除3592个标记Gaussian，得到510248。总删66148（其中P2已有32322）；相对P2新增33826。最终不使用Round3参数修改；保留点所有原生字段与原E10逐位一致。

{photo}

历史208条件R5保留标记：FG-PSNR19.47479→19.51278，SSIM.877588→.876838，LPIPS.117015→.116987，背景Alpha.331702→.045932。背景高Alpha比例.33760→.042835。最终排除标记后仍用原RAP评价，历史FG-PSNR降到18.68055；不能隐藏该下降。补充同一Plant+Pot区域指标19.49629→19.53903、叶缘带PSNR13.96383→14.15654，只排除原Mask的明确独立小标记组件，含混合标记的主组件未删。因此照片主体总体保持，语义移除与光度收益需分开。

几何208检查（R5历史场景 vsP2）：薄结构Mask骨架3px覆盖99.929%；公共前景深度变化>2vox约0.786%；每视角中位变化均值0，但P95均值.012677。source37/38等盆沿局部P95分别.33979/.29559，改变12.8%/11.5%公共FG，必须读worst_depth_change图。所查看大变化集中在盆沿，而不是整片叶片消失；局部颜色明显变差。法线中位变化0，P95均值2.64°。这不是扫描误差或物理叶厚测量。

标准SuperSplat版本把原生filter_3D影响烘焙到标准scale/opacity；四视角渲染RGB/Alpha最大差均<1/255，原生有效sigma最大差7.45e-9。原始ID和旧/新值存supersplat_baking_changes.npz。它与原生最终模型是同一有效表示的查看导出，非另一次训练。

Plant-only v1空间裁切未充分删除盆；v2加大已知盆区域删25198点，盆主体去除，但盆沿/叶基混合Gaussian和微标记还在，根部完整性未经认证。保留两种语义产物，不标称完整植物测量资产。

后处理限制：混合Gaussian同时负担纹理/叶缘和背景，整体删除/统一缩放不能独立修正错误SH和遮挡层。若下一步继续修复，优先对已定位ID进行局部前景贡献保护的尺度/颜色约束，并对背景alpha/表面偏离加训练约束；该判断基于剩余贡献与盆沿回归，不保证训练必然解决。本轮未重新训练。
''')
    md(ROOT/'task1_p3_local_repair/P3_LOCAL_REPAIR_ANALYSIS.md',f'''# P3 定点修复

source157漂浮环 **Resolved（固定已定位几何范围）**；整株Mesh **Improved**；内部错误连接 **Unresolved**。

原P3后处理Mesh最大实际组件是32，source157独立组件28/29共829三角面。208视角面中心可见/Mask/GGGS median/stereo证据：28组件visible73250、MaskFG39977，但GGGS深度支持0、Stereo仅5次（每面最多1视角）；29 visible17808、FG10464、两类深度支持0。前景Mask覆盖不能证明真实叶片。结合原RGB与已存世界定位，v1仅删除这829面，保留原Triangle ID映射与被删单独Mesh。

v2在v1上移除独立蓝标记组件97的10072面，总删10901；盆形主体保留。灰色、法线、线框、色彩、深度、三角面/组件图已生成并直接看过。source157固定ROI额外投影约1600→159，原定位1441个环像素已去除；叶尖保留，缺面仍221px。该Resolved不涵盖整叶或全Mesh。

{brief}

54个主组件细长连接候选有≥15可见视角/≥5Mask前景却缺乏深度支撑；candidate_review原照片显示部分与真实叶柄/卷叶重叠。未直接切除全部候选，也未作整株Laplacian smoothing或填叶间隙。Triangle ID及候选单独Mesh保留在wrong_connections；wrong_connections_edited=false。尚不能确认并修复叶内部全部桥接。

局部黑色区拆分为几何未命中、已有面但颜色偏黑、原照片本身深色。G2 source131局部2368个缺面像素、其中2193有有效source Stereo；原P3v2同ROI2362，故清理/重融合未解决此局部缺损。surface roughness/相邻法线角含真实卷叶，不称作扫描误差。

直接P3最佳版本为 `{ROOT/'task1_p3_local_repair/mesh_versions/P3_best_cleaned_mesh.ply'}`，解析到v2。v1保留历史含标记评价条件；v2为目标Plant+Pot。原始raw/post是保留引用，不覆盖源数据。
''')
    stereo=json.loads((ROOT/'comparisons/stereo_depth_comparison_summary.json').read_text())['summaries'];s1=stereo[g1.name];s2=stereo[g2.name]
    md(ROOT/'task2_cleaned_gs2mesh/COMBINATION_ANALYSIS.md',f'''# Cleaned GGGS → 官方GS2Mesh因果对照

G1和G2实际各完成208 Left/Right重新渲染、DLNR双向10迭代、全新深度缓存、共享范围TSDF和Raw/Post208评价。G1加载544074点、SHA406189476cf70e7e4a65c23fd87013140126cef828467b361ff8c25a27f9d7eb；G2加载510248点、SHAdc647363875a782c51c1987f35c9e1032a41a83e7366188c7a3599f6ea3b34f9。缓存逐相机记录RGB SHA和Gaussian SHA，与G0深度不共用。stereo_depth_comparison.csv有416行完整对照。

官方GS2Mesh固定提交560f3e8a2349aaad24fe9444ef1ca2770aeec5ab、DLNR Middlebury权重SHA57e46b8fed66b685f732b730ebffd71998dcabea49f8443ca77e015419f6ae18。baseline.037776436900048636、LR阈值3px、208相机原K及RAP、无Mask erosion；TSDF voxel.0017389331048768373、sdftrunc.008694665524384186、上界1.5692457556724548均未变。官方参数名TSDF_voxel存.8903337497，是源码中再除512的总尺度；不应误读为实际voxel。原官方4个core文件SHA验证未改。

G1相对G0共同FG Stereo深度中位变化均值{s1['depth_delta_median']:.8f}，>2vox变化{s1['depth_changed_gt2voxel_fraction']*100:.3f}%；FG有效支持.921890→{s1['new_fg_stereo_support']:.6f}，叶缘支持.775637→{s1['new_leaf_boundary_support']:.6f}。照片训练视角几乎不变，G1漂浮环仍存在；孔洞均值358.31→326.98有所改善，但IoU/BF/背景几何略变差。结论：小幅混合收益，不解决主伪影。

G2 FG Stereo有效支持.921890→{s2['new_fg_stereo_support']:.6f}、叶缘.775637→{s2['new_leaf_boundary_support']:.6f}、内部缺深度比例.073158→{s2['new_interior_missing_fraction']:.6f}；>2vox变化{s2['depth_changed_gt2voxel_fraction']*100:.3f}%。Mesh轮廓/背景改善，source157环无需局部编辑已消失（固定ROI额外像素163、缺面198），但整体缺面投影397.78比G0更差。Raster深度改善不等于融合面片完整。

G2自由视角又在原标记世界位置产生独立碎片。重新定位后删除新组件44/88/94/129/138共17449面；原P3编号97从未用作G2组件编号。候选71仍临近盆基部、未作为确证标记整片删。局部修复版另行208评价，保留未经修复G2避免因果混淆。G2_targeted孔洞均值347.89下降有语义删除影响，不能宣称补齐了叶片孔洞；source157及source131局部缺面未因标记删除改变。

{brief}

照片与Mesh可以共享同一个原生Gaussian输入，已由实际G2证明接口可用；质量上尚不能宣布共享模型满足三维资产要求。当前观看推荐G2定点标记清除版；若要求历史Mask完全含标记，G1或P3v1的孔洞数更小，但大局部伪影仍在。定点删除有效解决已确证独立环/标记，不解决复杂内部表面、冲突深度或盆沿纹理。

来源：[官方GS2Mesh](https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab)、本地PROVENANCE_VERIFICATION.json及每组render/stereo/fusion/evaluation日志。
''')
    md(ROOT/'FINAL_ARTIFACT_REFINEMENT_DECISION.md',f'''# ForeSplat伪影处理最终判定

**当前模型尚未达到完整、干净三维资产使用目标。** 本轮真实完成多轮Gaussian处理、P3已定位组件编辑、G1/G2完整208视角官方GS2Mesh与统一评价。质量标记逐项如下，Improved不等于Resolved。

| 验收问题 | 结论 | 实际证据 |
|---|---|---|
| 1 原P2为何不干净 | 已诊断 | 中心/邻域保护忽略大footprint背景危害，前层删除又露出黑层 |
| 2 新清理删了哪些 | 可追溯 | 相对E10删66148，含P2已有32322、新背景30234、语义标记3592；全部原ID和原字段保留 |
| 3 大面积黑云 | Improved | 固定8自由视角2682781→68362px，减少97.45%；仍有小云/黑条 |
| 4 叶尖/交叉尖刺 | Improved | 大块周边尖刺显著减，混合叶缘和交叠黑层仍残留 |
| 5 叶片/照片副作用 | Improved，局部回归Unresolved | Plant+Pot FG19.496→19.539；叶片主体保留，但盆沿错误亮纹理和局部深度明显改变 |
| 6 source157漂浮环 | Resolved（指定范围） | P3删除829原三角面；G2直接融合环已消失，叶尖仍在 |
| 7 叶内错误连接 | Unresolved | 54实际ID候选部分覆盖真叶柄/卷叶，未盲目切除 |
| 8 P2→GS2Mesh | 小幅混合收益 | 缺面358.31→326.98；其他轮廓指标略退，环仍在 |
| 9 Best→GS2Mesh | Improved | IoU/BF/背景几何改善；原G2缺面397.78更差、标记又生成；另存局部修复版 |
| 10 当前最佳Gaussian | Improved查看候选 | 510248点原生R5+标记清除；SuperSplat导出与原生有效参数校验 |
| 11 当前最佳Mesh | Improved查看候选 | G2_targeted，1492364面；轮廓/背景最佳，局部面片错误仍在 |
| 12 打开哪两个文件 | 见下方 | 绝对路径及SHA在artifact_index/ALL_ARTIFACTS.csv |

SuperSplat：`{selected}`

Mesh：`{selectedmesh}`

原生Gaussian输入：`{best/'cleaned_gaussians.ply'}`；当前直接P3定点修复：`{ROOT/'task1_p3_local_repair/mesh_versions/P3_best_cleaned_mesh.ply'}`。

{brief}

上述208指标全部沿用历史RAP，仍包含原标记目标；G2_targeted语义去标记导致FG coverage从.986937降至.982709。孔洞统计是>=9px、远离Mask边缘的封闭缺失投影，单位px，不是扫描真值或物理孔洞面积。标记删除后的孔洞减少不能归因叶片补面。

剩余根因可追溯：4011个残余黑贡献Gaussian，96.27%权重在前景/背景混合点；误删会牺牲已测前景。source37/38盆沿黑遮挡去除后由3135/290315/316725等点贡献错误亮纹理。source131固定ROI缺面2368，其中2193有有效source Stereo，说明仅清Gaussian仍不能修复多视角冲突或TSDF局部融合。还有近盆基部独立碎片和叶片表面不合理起伏。

后续若继续，需要对已定位混合Gaussian作局部受约束处理，或在已有GGGS路线训练中约束背景贡献/尺度/表面一致性；叶内部连接及孔洞需基于局部多视角一致性独立修复。本轮未执行任何重新训练、VGGT或RAP推理；不以可能的未来效果充当本轮验收。

真实查看：当前模型直接读取原生渲染与Open3D灰色/法线/线框/颜色图片；已读路径和逐图观察在comparisons/VISUAL_REVIEW.json。没有声称交互式SuperSplat/Blender GUI完成；查看器兼容通过PLY字段、原生渲染及Open3D实际读取验证。

完整绝对模型目录：artifact_index/ALL_GAUSSIAN_FILES.md、ALL_MESH_FILES.md；所有实验比较：comparisons/gaussian_cleaning_metrics.csv、mesh_artifact_metrics.csv、stereo_depth_comparison.csv（416行）、local_mesh_roi_metrics.csv。删除原因和原Triangle/Gaussian ID均在各版本目录。详细分析分task0、task1、task2报告。

资料来源：[真实ForeSplat项目](https://github.com/LiuLiShenShe/test-froesplat)、[固定官方GS2Mesh](https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab)与本地真实文件/实验。Bocha检索返回HTTP403额度不足，未据此编造搜索结果；官方源文件直接获取，ctx7对GS2Mesh无索引，uv官方文档已取。质量结论均来源本地测量和直接看图。
''')
    write(ROOT/'FINAL_STATUS.json',dict(experiment_execution='complete',asset_quality='Improved; not ready as a fully clean 3D asset',gaussian=str(selected),native_gaussian=str(best/'cleaned_gaussians.ply'),mesh=str(selectedmesh),source157_ring='Resolved within fixed located region',wrong_internal_connections='Unresolved',cloud_reduction_8fixed_views=1-68362/2682781,inventory_models=len(inventory),all_required_208_stages_verified=True))
    print('final reports and indices written',len(inventory),flush=True)
if __name__=='__main__':main()
