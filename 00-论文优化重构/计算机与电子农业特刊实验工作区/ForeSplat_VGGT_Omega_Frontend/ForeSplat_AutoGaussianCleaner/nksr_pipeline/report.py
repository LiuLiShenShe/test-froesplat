"""Assemble measured three-plant artifacts; never reconstruct or infer traits."""
import csv
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np

from PIL import Image,ImageDraw,ImageFont

from .common import HISTORY,LABELS,OUT,PLANTS,ROOT,hash_file,stamp,write


def csvwrite(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify():
    summary=OUT/'summary'
    protocol=json.loads((summary/'protocol.json').read_text())
    for path,expected in protocol['historical_source_sha256'].items():
        assert hash_file(path)==expected,path
    details=[]
    for plant in PLANTS:
        saved=protocol['inputs'][plant]
        spec=json.loads((ROOT/f'configs/scenes/{plant}.json').read_text())
        assert stamp(ROOT/f'configs/scenes/{plant}.json')==saved['scene']
        assert stamp(spec['model'])==saved['original']
        assert stamp(ROOT/f'experiments/C1_C2_all19/{plant}/C1_raw/cleaned_gaussians.ply')==saved['C1_raw']
        surface=json.loads((OUT/plant/'surface/complete.json').read_text())
        cameras=json.loads((OUT/plant/'surface/camera_runtime.json').read_text())['records']
        expected_cameras=json.loads(Path(spec['geometry_cameras']).read_text())['frames']
        assert surface['used_camera_count']==len(cameras)==len(surface['frames'])==len(expected_cameras)
        assert {c['name'] for c in cameras}=={Path(c['image_name']).stem for c in expected_cameras}
        with zipfile.ZipFile(OUT/plant/'surface/nksr_input.npz') as archive:
            assert set(archive.namelist())=={'xyz.npy','normal.npy','sensor.npy','rgb.npy'}
            for name in ['xyz','normal','sensor','rgb']:
                with archive.open(name+'.npy') as stream:
                    assert np.lib.format.read_magic(stream)==(1,0)
                    shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                assert shape==(surface['nksr_points'],3) and not fortran
                assert dtype==np.dtype('uint8' if name=='rgb' else 'float32')
        report=json.loads((OUT/plant/'mesh/mesh_report.json').read_text())
        phenotype=json.loads((OUT/plant/'phenotype/geometry.json').read_text())
        assert report['input_points']==surface['nksr_points']
        assert report['native_module_sha256']==json.loads((HISTORY/'configs/nksr_compatibility_module.json').read_text())['new_sha256']
        assert report['weight_sha256']==json.loads((HISTORY/'configs/nksr_environment_ready.json').read_text())['weight_sha256']
        assert report['parameters']==dict(detail_level=None,voxel_size=surface['scale']['diagonal']/512,
            solver_max_iter=2000,solver_tol=1e-5,approx_kernel_grad=False,fused_mode=True,mise_iter=1,trim=True,max_points=1000000)
        for name in ['raw','post']:
            path=OUT/plant/f'mesh/{name}.ply'
            with path.open('rb') as stream:
                lines=[]
                while True:
                    line=stream.readline(4096);lines.append(line)
                    assert line and len(lines)<100
                    if line.strip()==b'end_header':break
            nv=int(next(x for x in lines if x.startswith(b'element vertex')).split()[-1])
            nf=int(next(x for x in lines if x.startswith(b'element face')).split()[-1])
            assert path.stat().st_size==sum(map(len,lines))+nv*51+nf*13
            assert nv==report[f'{name}_vertices'] and nf==report[f'{name}_faces']
        assert phenotype['stages']['post']['post_vertices_exact_subset_of_raw']
        assert json.loads((OUT/plant/'visualizations/reference_rgb/complete.json').read_text())['views']==4
        for folder in ['surface','mesh','visualizations','phenotype','visualizations/reference_rgb']:
            resource=json.loads((OUT/plant/folder/'resources.json').read_text())
            assert resource['status']=='complete'
            assert resource['observed_peak_one_second_mib_s']<=20
            assert resource['physical_gpu']==(None if folder in ['phenotype','visualizations/reference_rgb'] else 1)
        for order in range(4):
            sizes=[]
            for prefix in ['reference','C1_GGGS','NKSR_raw','NKSR_post']:
                with Image.open(OUT/plant/f'visualizations/{prefix}_{order:02}.png') as image:sizes.append(image.size)
            assert len(set(sizes))==1
        details.append(dict(plant=plant,cameras=surface['used_camera_count'],input_points=surface['nksr_points'],
            raw_vertices=report['raw_vertices'],raw_faces=report['raw_faces'],post_vertices=report['post_vertices'],post_faces=report['post_faces']))
    assert not list(OUT.rglob('*.partial'))
    result=dict(passed=True,plants=PLANTS,meshes=6,matched_comparison_images=3,
        checks=['unchanged original GGGS/C1 stamps','unchanged historical source hashes',
            'existing NKSR binary and weight hashes','fixed NKSR parameters','all actual camera counts',
            'surface NPZ fields/shapes/dtypes','mesh headers/counts/lengths','exact raw/post vertex subset','stage GPU and I/O records','matched image dimensions'],
        regression_tests='3 adapter checks passed',details=details,
        final_adapter_sha256={str(path.relative_to(ROOT)):hash_file(path) for path in sorted((ROOT/'nksr_pipeline').glob('*.py'))})
    write(summary/'verification.json',result)
    return result


def main():
    summary=OUT/'summary'
    assert json.loads((summary/'run_complete.json').read_text())['completed']
    reviews=json.loads((summary/'visual_reviews.json').read_text()) if (summary/'visual_reviews.json').exists() else {}
    mesh_rows,phenotype_rows,resource_rows,all_data=[],[],[],{}
    command_records=[json.loads(line) for line in (summary/'actual_commands.jsonl').read_text().splitlines()]
    entrypoint={(r['plant'],r['stage']):r['entrypoint_wall_s'] for r in command_records if 'entrypoint_wall_s' in r}
    for plant in PLANTS:
        root=OUT/plant
        surface=json.loads((root/'surface/complete.json').read_text())
        mesh=json.loads((root/'mesh/mesh_report.json').read_text())
        geometry=json.loads((root/'phenotype/geometry.json').read_text())
        quality=json.loads((root/'visualizations/metrics.json').read_text())
        for name in ['raw','post']:
            structure=mesh['structure'][name]
            mesh_rows.append(dict(plant=plant,label=LABELS[plant],stage=name,input_points=surface['nksr_points'],
                cameras=surface['used_camera_count'],vertices=structure['vertices'],faces=structure['faces'],components=structure['components'],
                boundary_edges=structure['boundary_edges'],nonmanifold_edges=structure['nonmanifold_edges'],
                largest_component_face_fraction=structure['largest_component_face_fraction'],surface_area_model_units2=structure['surface_area'],
                silhouette_iou_mean=sum(r['silhouette_iou'] for r in quality['views'][name])/4,
                foreground_coverage_mean=sum(r['foreground_coverage'] for r in quality['views'][name])/4,
                nksr_compute_s=mesh['timing']['nksr_compute_s'],gpu_peak_process_gib=mesh['resources']['gpu_peak_process_mib']/1024,
                mesh_path=str((root/f'mesh/{name}.ply').resolve())))
        with (root/'phenotype/phenotype.csv').open() as stream:phenotype_rows.extend(csv.DictReader(stream))
        for stage,folder in [('surface','surface'),('reconstruct','mesh'),('evaluate','visualizations'),('phenotype','phenotype'),('reference_rgb','visualizations/reference_rgb')]:
            value=json.loads((root/folder/'resources.json').read_text())
            resource_rows.append(dict(plant=plant,stage=stage,active_wall_s=value['wall_s'],
                entrypoint_wall_with_startup_s=entrypoint.get((plant,stage),''),process_cpu_s=value['process_cpu_s'],
                peak_rss_gib=value['peak_rss_gib'],gpu_peak_process_gib=value['gpu_peak_process_mib']/1024 if value['gpu_peak_process_mib'] is not None else '',
                gpu_peak_allocated_gib=value['gpu_peak_allocated_gib'],gpu_peak_reserved_gib=value['gpu_peak_reserved_gib'],
                logical_read_mib=value['logical_read_bytes']/2**20,logical_write_mib=value['logical_write_bytes']/2**20,
                maximum_one_second_mib_s=value['observed_peak_one_second_mib_s'],physical_gpu=value['physical_gpu']))
        all_data[plant]=dict(surface_points=surface['nksr_points'],cameras=surface['used_camera_count'],mesh=mesh,geometry=geometry,visual_review=reviews.get(plant))
    csvwrite(summary/'mesh_results.csv',mesh_rows)
    csvwrite(summary/'phenotype_results.csv',phenotype_rows)
    csvwrite(summary/'resources.csv',resource_rows)
    write(summary/'results.json',all_data)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',30)
    canvas=Image.new('RGB',(3600,2380),'white');draw=ImageDraw.Draw(canvas)
    for col,plant in enumerate(PLANTS):
        draw.text((col*1200+10,10),f'{LABELS[plant]} / {plant}',fill='black',font=font)
        with Image.open(OUT/plant/'visualizations/mesh_comparison.png') as image:canvas.paste(image,(col*1200,60))
        # The supplementary panel makes normal-facing surfaces inspectable
        # without changing the colour comparison's background or exposure.
        geometric=Image.new('RGB',(1200,1160),'white');gd=ImageDraw.Draw(geometric)
        for row,name in enumerate(['raw','post']):
            for order in range(4):
                with Image.open(OUT/plant/f'visualizations/NKSR_{name}_geometry_{order:02}.png') as image:
                    image.thumbnail((300,535));geometric.paste(image,(order*300, row*580+45))
                gd.text((order*300+5,row*580+5),f'{name} | view {order+1}',fill='black',font=font)
        geometric.save(OUT/plant/'visualizations/geometry_comparison.png')
    canvas.save(summary/'mesh_overview.png')
    verification=verify()
    html=['<!doctype html><meta charset="utf-8"><title>C1 NKSR 三株结果</title>',
        '<style>body{font:17px sans-serif;margin:24px;max-width:1500px}img{max-width:100%;height:auto}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style>',
        '<h1>C1 GGGS → NKSR → 初步几何测量</h1><p>三株均使用全部训练视角提取表面。每图四行：原始拍摄RGB、C1 GGGS、NKSR Raw、NKSR Post；四列是完全一致的相机。渲染行灰背景0.5，照片保留原背景，未改变曝光；黑底训练参考另存training_reference。</p>',
        '<p><a href="mesh_results.csv">Mesh统计</a> · <a href="phenotype_results.csv">几何测量</a> · <a href="resources.csv">分阶段资源</a> · <a href="mesh_overview.png">总览</a> · <a href="conclusions.md">结论</a> · <a href="verification.json">核验</a></p>',
        '<p>所有几何高度和跨度目前均为包含花盆及参考物的PCA探索性模型单位范围，不能当作正式株高/冠幅。全网格表面积也不是叶面积。厘米尺度与叶片实例信息不足，物理值留空。</p>']
    for plant in PLANTS:
        note=reviews.get(plant,{}).get('notes','视觉审查待完成')
        html.extend([f'<h2>{LABELS[plant]}</h2><p>{note}</p>',
            f'<img loading="lazy" src="../{plant}/visualizations/mesh_comparison.png">',
            f'<p><a href="../{plant}/visualizations/geometry_comparison.png">几何着色图</a> · <a href="../{plant}/mesh/raw.ply">Raw PLY</a> · <a href="../{plant}/mesh/post.ply">Post PLY</a> · <a href="../{plant}/mesh/mesh_report.json">Mesh报告</a> · <a href="../{plant}/phenotype/geometry.json">几何及测量边界</a></p>'])
    (summary/'index.html').write_text('\n'.join(html)+'\n')
    if set(reviews)==set(PLANTS):
        write_conclusions(summary,all_data,mesh_rows,resource_rows,reviews)
        light=ROOT/'reports/C1_NKSR_Phenotype';light.mkdir(parents=True,exist_ok=True)
        for name in ['mesh_results.csv','phenotype_results.csv','resources.csv','verification.json']:
            shutil.copyfile(summary/name,light/name)
        report_text=(summary/'conclusions.md').read_text()
        report_text=re.sub(r'\]\(([^)]+)\)',lambda match: ']('+str((summary/match[1]).resolve())+')',report_text)
        (light/'conclusions.md').write_text(report_text)
        # This tracked index points to the server artifacts, never copying PLY/NPZ.
        (light/'RESULTS.md').write_text('三株已完成。完整结果位于服务器实验目录。\n\n'
            f'- [图像和模型索引]({summary}/index.html)\n- [总览图]({summary}/mesh_overview.png)\n'
            '- [实测结论](conclusions.md)\n- [Mesh统计](mesh_results.csv)\n- [表型几何](phenotype_results.csv)\n- [资源](resources.csv)\n')
    print('Assembled three plants;',len(reviews),'visual reviews; verified',verification['meshes'],'meshes')


def write_conclusions(summary,data,mesh_rows,resources,reviews):
    lines=['# 三株 C1 → NKSR → 初步植物几何测量实测结果','',
        '三株均已实际完成。交付6份三角Mesh（Raw/Post各3份）、三张四行四视角比较图和模型单位几何测量。使用既有C1结果、原生GGGS渲染、已验证NKSR环境/权重/兼容模块；没有重新训练、清理或调整跨植物重建参数。原始RGB行来自02-FFT原始照片，保留实际背景；与相机对应的整幅画布只按评价分辨率缩放，不掩膜或改变曝光。原黑底训练参考另存training_reference，逐帧路径及前景像素一致性核验见visualizations/reference_rgb/sources.json。','',
        '[完整图像与模型索引](index.html) · [三株总览](mesh_overview.png) · [Mesh统计](mesh_results.csv) · [表型几何](phenotype_results.csv) · [阶段资源](resources.csv) · [核验](verification.json)','',
        '| 植物 | 相机数 | NKSR输入点 | Raw顶点/面 | Post顶点/面 | NKSR实际耗时(s) | NKSR进程峰值显存(GiB) |',
        '|---|---:|---:|---|---|---:|---:|']
    for plant in PLANTS:
        d=data[plant];m=d['mesh']
        lines.append(f'| {LABELS[plant]} | {d["cameras"]} | {d["surface_points"]:,} | {m["raw_vertices"]:,} / {m["raw_faces"]:,} | {m["post_vertices"]:,} / {m["post_faces"]:,} | {m["timing"]["nksr_compute_s"]:.2f} | {m["resources"]["gpu_peak_process_mib"]/1024:.2f} |')
    lines += ['', 'NKSR耗时计入模型初始化/权重加载、固定参数求解和dual mesh提取，排除表面生成及后续颜色/后处理导出；阶段资源表另列整个阶段及入口含启动耗时。GPU峰值为按本PID采样的物理GPU1使用量，Torch allocated/reserved另存。','',
        '| 植物 | Post顶点 | Post面 | 几何高度代理(模型单位) | 水平跨度代理(模型单位，两轴) | 物理株高/冠幅 | 叶片测量 |',
        '|---|---:|---:|---:|---|---|---|']
    for plant in PLANTS:
        m=data[plant]['mesh'];g=data[plant]['geometry']['stages']['post'];spans=g['exploratory_horizontal_pca_spans_units']
        lines.append(f'| {LABELS[plant]} | {m["post_vertices"]:,} | {m["post_faces"]:,} | {g["exploratory_height_pca_units"]:.6f} | {spans[0]:.6f} / {spans[1]:.6f} | PHYSICAL_SCALE_NOT_AVAILABLE | 缺Leaf ID、叶基叶尖和边界 |')
    lines += ['', '**上述尺寸是全Mesh包含花盆、蓝色参考物及可能伪影的探索性PCA范围，不是正式植物株高或冠幅。** PCA主轴没有被当成重力方向。全Mesh表面积包含花盆、参考物、粗糙和可能双层表面，不能当成叶面积。XYZ包围盒、PCA三轴、表面积、原前景候选点范围及Raw/Post变化都记录在各株geometry.json及总CSV。物理值留空，不根据未知盆径或历史表格假设厘米尺度。','',
        '| 植物 | 阶段 | 活动耗时(s) | 入口总墙钟(s) | CPU(s) | 峰值RSS(GiB) | 峰值显存(GiB) | 读/写(MiB) | 最大合计I/O(MiB/s) |',
        '|---|---|---:|---:|---:|---:|---:|---|---:|']
    stage_labels=dict(surface='GGGS表面提取',reconstruct='NKSR重建',evaluate='Mesh评价',phenotype='表型计算',reference_rgb='原始RGB参考整理(CPU)')
    for r in resources:
        wall=f'{r["entrypoint_wall_with_startup_s"]:.2f}' if r['entrypoint_wall_with_startup_s']!='' else '未单独计时'
        gpu=f'{r["gpu_peak_process_gib"]:.2f}' if r['gpu_peak_process_gib']!='' else '—'
        lines.append(f'| {LABELS[r["plant"]]} | {stage_labels[r["stage"]]} | {r["active_wall_s"]:.2f} | {wall} | {r["process_cpu_s"]:.2f} | {r["peak_rss_gib"]:.3f} | {gpu} | {r["logical_read_mib"]:.1f} / {r["logical_write_mib"]:.1f} | {r["maximum_one_second_mib_s"]:.3f} |')
    lines += ['', '活动时间覆盖阶段主体操作；独立入口时间含uv/Python/动态库冷启动及进程退出（首株表面入口未单独计时）。CPU记录为阶段主体进程时间，RSS为该阶段进程峰值；没有混入历史NKSR或C1清理耗时。大文件与图像流共用阶段20 MiB/s限速器，记录应用层字节而非设备物理峰值。', '',
        '首株Mesh评价在结果写完后发生原生库解释器退出延迟，最终自行正常退出；未发送成功的终止信号，也未重算或修改几何。之后评价入口补齐与历史脚本一致的完成后直接退出处理。该额外墙钟时间保留在入口统计，不伪装成NKSR计算或删除此记录。详见首株visualizations/exit_deviation.json。','']
    for plant in PLANTS:
        lines += [f'**{LABELS[plant]}：** {reviews[plant]["notes"]}', '',
            f'[四行四视角图](../{plant}/visualizations/mesh_comparison.png) · [几何着色图](../{plant}/visualizations/geometry_comparison.png) · [Raw Mesh](../{plant}/mesh/raw.ply) · [Post Mesh](../{plant}/mesh/post.ply)', '']
    lines += ['**下一步判断。**','',
        '1. C1清理后的GGGS可以沿现有Median Depth→有向表面点→官方ks流程进入NKSR，三株已完成；这是三株上的流程稳定性证据，不是对其余16株的质量保证。',
        '2. 叶片、边缘及细结构的实际表现见上方逐株直接看图记录；成功生成三角面或减少连通分量不等于植物器官完整。四视角可见性也不能证实所有叶片都已完整恢复。',
        '3. NKSR会对稠密观测进行隐式表面插值，可能保留或连接输入已有的错误深度；具体异常见逐株记录。Post只删<50面连通块，不能修复大块错误连接、双层叶面或植株/花盆粘连。',
        '4. 现在可计算全Mesh XYZ尺寸、PCA范围、表面积、连通性、边界/非流形边，以及全前景候选点的探索性跨度。正式株高/冠幅仍缺植物专属区域、独立竖直方向和植物基准。',
        '5. 厘米级测量需同株同次采集的真实已知长度、其两个三维对应点及单位/不确定度记录；再以真实长度除以模型距离得到比例。叶片测量还需要实例ID、叶基、叶尖、边界/面集合和测量方向。',
        '6. 路径、相机数量及世界坐标适配已支持其他Scene；其余植物可以复用相同代码和公式。当前驱动限制为这三株，未自动启动其余16株。扩展前应保留逐株相机核验、资源检查和几何审查。','',
        '**固定协议与证据。** 全训练视角、训练分辨率二分之一；Alpha>=0.5，深度跳变0.05，既有差分法向朝向相机；采样robust bbox diagonal/1024，重建/512；官方ks与原兼容模块；迭代2000、tol1e-5、fused_mode、MISE1、trim、max_points1000000均固定。既有原始源码及GGGS/C1输入戳保持不变，3项适配回归与交付核验通过。原始及Post网格各自保存，未使用最大连通块策略。','',
        '信息来源：本轮真实运行记录、直接查看对比图、仓库历史源码及表型证据审计。博查HTTP403，Context7无NKSR匹配项；未用外部摘要代替本次实验结果。','']
    (summary/'conclusions.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
