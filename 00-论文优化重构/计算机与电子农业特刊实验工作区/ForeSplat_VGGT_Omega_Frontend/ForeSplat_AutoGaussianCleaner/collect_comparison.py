#!/usr/bin/env python3
"""Index only real artifacts and completed metrics; visualize shared cameras."""
import sys
import csv
import json
import re
import fcntl
import os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from PIL import Image,ImageDraw,ImageOps
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha


def experiment_files(base, name):
    # Source/vendor copies cannot contain experimental results and can have
    # tens of thousands of irrelevant files. Keep actual output traversal.
    for folder, directories, filenames in os.walk(base, followlinks=False):
        directories[:]=[d for d in directories if d not in ['.git','node_modules','editor_source','source_snapshot','development_variants']]
        if name in filenames:
            yield Path(folder)/name


def main():
    base=Path(__file__).resolve().parent/'experiments/GGGS_Cleaner_Comparison'
    comp=base/'comparisons';index=base/'artifact_index';comp.mkdir(exist_ok=True);index.mkdir(exist_ok=True)
    lock=(index/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    measurements=[];files=[];method_dirs={};selected_priority={};completed_dirs=set();invalidated=[];integrity_failures=[]
    original_report=base/'baselines/C0/cleanup_metrics.json'
    original_mean=json.loads(original_report.read_text())['metrics']['mean'] if original_report.exists() else None
    shared_original_cloud_path=base/'baselines/C0/BLACK_CLOUD_METRICS.json'
    shared_original_cloud=json.loads(shared_original_cloud_path.read_text()) if shared_original_cloud_path.exists() else {}
    awaiting_cli=[]
    browser_audits={}
    for audit_file in sorted((index/'supersplat_browser_audit').glob('*/BROWSER_AUDIT.json')):
        audit=json.loads(audit_file.read_text())
        if audit.get('status')=='actual_import_and_render_verified':
            browser_audits[audit['ply_sha256']]=str(audit_file.resolve())
    for m in sorted(experiment_files(base,'cleanup_metrics.json')):
        out=m.parent
        if (out/'INVALIDATED.json').exists():
            invalidated.append(dict(experiment=str(out.resolve()),reason=json.loads((out/'INVALIDATED.json').read_text())))
            continue
        if not (out/'COMPLETE.json').exists():continue
        d=json.loads(m.read_text())
        batch_parent=next((parent for parent in out.parents if (parent/'batch_status.json').exists() and base in parent.parents),None)
        if batch_parent is not None:
            child=json.loads((batch_parent/'batch_status.json').read_text()).get(d.get('plant'),{})
            if child.get('status')!='complete':
                awaiting_cli.append(dict(experiment=str(out.resolve()),child_status=child.get('status','pending'),
                    reason='Engine export exists, but authoritative isolated CLI post-validation is not complete'))
                continue
        try:
            marker=json.loads((out/'COMPLETE.json').read_text())
            if sha(out/'cleaned_gaussians.ply')!=marker['native_sha256']:
                raise ValueError('Native PLY differs from completion marker')
            for name,expected in marker.get('artifact_hashes',{}).items():
                if sha(out/name)!=expected:raise ValueError(f'Artifact differs from completion marker: {name}')
        except (OSError,ValueError,KeyError) as error:
            integrity_failures.append(dict(experiment=str(out.resolve()),reason=str(error)))
            continue
        completed_dirs.add(out)
        method=d['method']
        version=str(d.get('config',{}).get('method_version','historical_or_baseline'))
        plant=d.get('plant','KongQueZhuYu')
        mean=d.get('metrics',d.get('final'))['mean']
        extra=out/'FG_LOCAL_METRICS.json'
        local_mean=json.loads(extra.read_text())['mean'] if extra.exists() else mean
        row=dict(plant=plant,method=method,experiment_version=version,experiment=str(out.resolve()),
            input_sha256=d.get('input_sha256',marker.get('input_sha256','')),native_sha256=marker['native_sha256'],
            outcome=d.get('outcome','historical_reference' if method.startswith('C3') else
                'no_accepted_modification' if method=='C4' else 'original_baseline' if method=='C0' else 'uncertified_appearance_rule_comparison'),
            original_gaussians=d['original_gaussians'],cleaned_gaussians=d['cleaned_gaussians'],
            removed=d['removed_gaussians'],seconds=d['seconds'],peak_allocated_mib=d['peak_allocated_mib'],
            sampled_process_peak_mib=d.get('gpu_memory',{}).get('sampled_process_peak_mib',''),
            gpu_sampling_seconds=d.get('gpu_memory',{}).get('sampling_interval_seconds',''),
            evaluated_views=len(d.get('metrics',d.get('final'))['views']),
            refined_gaussians=d.get('refined_gaussians',0),
            timing_scope=('reused historical output: evaluation/export/parity/visuals only' if method.startswith('C3') or method=='C4' else
                'baseline evaluation/export/parity/visuals; no cleaning' if method=='C0' else
                'appearance rules plus evaluation/export/parity/visuals' if method.startswith(('C1','C2')) else
                'complete automatic analysis/trials/refinement/evaluation/export/parity/visuals'),
            historical_cleaning_seconds=d.get('historical_seconds',''),
            fg_alpha=mean['fg_alpha'],boundary_coverage=mean['boundary_coverage'],
            dark_foreground_mse=mean['dark_foreground_mse'],depth_changed_fraction=mean['depth_changed_fraction'],
            fg_psnr=mean['fg_psnr'],fg_ssim=mean['fg_ssim'],lpips=mean.get('fg_masked_lpips'),
            bg_alpha=mean['bg_alpha'],bg_alpha_mass_per_fg=mean['bg_alpha_mass_per_fg'],
            high_alpha_background_pixels=mean['bg_high_alpha_pixels'],largest_background_component=mean['largest_bg_component_fraction'],
            dark_background_pixels=mean['dark_bg_high_alpha_pixels'],boundary_f1=mean['boundary_f1'],
            foreground_local_boundary_f1=local_mean.get('fg_local_boundary_f1',''),
            thin_coverage=mean['thin_coverage'],depth_p95=mean['depth_relative_p95'],normal_p95=mean['normal_angle_p95_deg'],
            background_alpha_mass_reduction=(d.get('artifact_reduction') if 'artifact_reduction' in d else
                (original_mean['bg_alpha_mass_per_fg']-mean['bg_alpha_mass_per_fg'])/max(original_mean['bg_alpha_mass_per_fg'],1e-12)
                if plant=='KongQueZhuYu' and original_mean else ''),
            free_artifact_reduction=d.get('free_artifact_reduction',''),
            external_depth_mae=mean.get('external_depth_mae',''),external_depth_p95=mean.get('external_depth_p95',''),
            native_ply=d['native_ply'],supersplat_ply=d['supersplat_ply'],removed_ply=d['removed_ply'],report=str(m.resolve()))
        historical=out/'visualizations/historical_free_metrics.json'
        row['historical_free_black_pixels']=json.loads(historical.read_text())['black_pixels'] if historical.exists() else ''
        row['automatic_free_components']=''
        free=out/'visualizations/automatic_free_views.json'
        if free.exists():row['automatic_free_components']=sum(v['external_components'] for v in json.loads(free.read_text())['metrics'])
        cloud=out/'BLACK_CLOUD_METRICS.json'
        clouds=json.loads(cloud.read_text()) if cloud.exists() else {}
        original_cloud=out/'ORIGINAL_BLACK_CLOUD_METRICS.json'
        before_clouds=json.loads(original_cloud.read_text()) if original_cloud.exists() else {}
        if (not before_clouds and clouds and plant=='KongQueZhuYu'
                and shared_original_cloud.get('native_sha256')==row['input_sha256']):
            before_clouds=shared_original_cloud
            original_cloud=shared_original_cloud_path
        if clouds and clouds['native_sha256']!=marker['native_sha256']:
            raise ValueError(f'Black-cloud evaluation belongs to another model: {out}')
        if before_clouds and before_clouds['native_sha256']!=row['input_sha256']:
            raise ValueError(f'Original black-cloud evaluation belongs to another input: {out}')
        if clouds and before_clouds:
            keys=['resolution_max_side','definition']
            train=lambda data:[(v['index'],v['name'],v['background']) for v in data['training_views']]
            free=lambda data:[(v['name'],v['background']) for v in data['free_views']]
            if (any(clouds[key]!=before_clouds[key] for key in keys)
                    or train(clouds)!=train(before_clouds) or free(clouds)!=free(before_clouds)):
                raise ValueError(f'Paired black-cloud measurements use different camera/measurement protocols: {out}')
        row['original_black_cloud_metrics']=str(original_cloud.resolve()) if before_clouds else ''
        for background in ['black','white']:
            for metric in ['dark_bg_high_alpha_pixels','largest_dark_bg_component_pixels','largest_dark_bg_component_fraction','isolated_small_dark_bg_components','isolated_small_dark_bg_pixels']:
                after=clouds.get('training_mean',{}).get(background,{}).get(metric,'')
                before=before_clouds.get('training_mean',{}).get(background,{}).get(metric,'')
                row[background+'_'+metric]=after
                row[background+'_'+metric+'_original']=before
                row[background+'_'+metric+'_reduction']=(before-after)/before if isinstance(before,(int,float)) and before>0 and isinstance(after,(int,float)) else ''
        review_file=out/'quality_inspection/DIRECT_VISUAL_REVIEW.json'
        review=json.loads(review_file.read_text()) if review_file.exists() else {}
        if review and review.get('native_sha256',marker['native_sha256'])!=marker['native_sha256']:
            raise ValueError(f'Direct visual review belongs to another native model: {out}')
        row['direct_visual_review']=str(review_file.resolve()) if review else ''
        row['conspicuous_leaf_loss_in_reviewed_views']=review.get('conspicuous_leaf_loss_in_reviewed_views','')
        row['residual_black_cloud_visible']=review.get('residual_black_cloud_visible','')
        row['actual_supersplat_browser_audit']=browser_audits.get(marker.get('artifact_hashes',{}).get('cleaned_gaussians_supersplat.ply'),'')
        measurements.append(row)
        if plant=='KongQueZhuYu':
            numbers=tuple(float(x) for x in re.findall(r'\d+(?:\.\d+)?',version))
            priority=(int('final' in out.parts),numbers)
            if method not in selected_priority or priority>selected_priority[method]:
                method_dirs[method]=out;selected_priority[method]=priority
        for k in ['native_ply','supersplat_ply','removed_ply']:
            f=Path(d[k]);files.append(dict(plant=plant,method=method,experiment_version=version,type=k,path=str(f.resolve()),bytes=f.stat().st_size))
        for name in ['original_gaussian_ids.npy','removed_original_gaussian_ids.npy','removed_gaussians_supersplat.ply','cleanup_metrics.json','viewer_baking.json','original_reference.json','execution.json','COMPLETE.json','PIPELINE_STAGE.json','FG_LOCAL_METRICS.json','BLACK_CLOUD_METRICS.json','ORIGINAL_BLACK_CLOUD_METRICS.json','quality_inspection/DIRECT_VISUAL_REVIEW.json','quality_inspection/FREE_VIEW_CONTACT_SHEET.json']:
            f=out/name
            if f.exists():files.append(dict(plant=plant,method=method,experiment_version=version,type=name,path=str(f.resolve()),bytes=f.stat().st_size))
        for folder in ['visualizations','quality_inspection']:
            for f in sorted((out/folder).rglob('*.png')):
                files.append(dict(plant=plant,method=method,experiment_version=version,type=folder,path=str(f.resolve()),bytes=f.stat().st_size))
    for name,data in [('comparisons/ALL_GGGS_CLEANING_RESULTS.csv',measurements),('artifact_index/ALL_ARTIFACTS.csv',files)]:
        if data:
            with (base/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    pending_models=[]
    for m in sorted(experiment_files(base,'MODEL_READY.json')):
        if m.parent in completed_dirs:continue
        d=json.loads(m.read_text());out=m.parent
        interrupted=any((p/'INTERRUPTED.json').exists() for p in [out,*out.parents] if p==base or base in p.parents)
        status=('rejected_candidate_not_selected' if (out/'REJECTED_CANDIDATE.json').exists() else
            'uncertified_rules_complete' if (out/'RULES_COMPLETE.json').exists() else
            'interrupted_development_snapshot' if interrupted else 'model_exported_evaluation_pending')
        removed=d.get('removed',d.get('removed_gaussians'))
        if removed is None:raise ValueError(f'Model export lacks a recorded deletion count: {m}')
        pending_models.append(dict(method=str(out.relative_to(base)),removed=removed,status=status,
            native_ply=str((out/'cleaned_gaussians.ply').resolve()),supersplat_ply=str((out/'cleaned_gaussians_supersplat.ply').resolve()),
            removed_ply=str((out/'removed_gaussians.ply').resolve()),report=str(m.resolve())))
    write_json(index/'PENDING_MODEL_EXPORTS.json',pending_models)
    write_json(index/'PENDING_CLI_VERIFICATION.json',awaiting_cli)
    write_json(index/'INVALIDATED_EXPERIMENTS.json',invalidated)
    write_json(index/'ARTIFACT_INTEGRITY_FAILURES.json',integrity_failures)
    text='# Actual GGGS artifact paths\n\nCompleted results only. Zero deletion is explicitly recorded.\n\n'
    for r in measurements:
        text+=f"## {r['plant']} / {r['method']} / {r['experiment_version']}: {r['original_gaussians']} → {r['cleaned_gaussians']} ({r['removed']} deleted)\n\n"
        for k in ['native_ply','supersplat_ply','removed_ply','report']:text+=f"- [{k}]({r[k]})\n"
        text+='\n'
    (index/'ALL_GGGS_PLY_PATHS.md').write_text(text)
    if pending_models:
        with (index/'ALL_GGGS_PLY_PATHS.md').open('a') as f:
            f.write('\n## Exported models awaiting full evaluation\n\n')
            for r in pending_models:
                f.write(f"### {r['method']}: {r['removed']} removed IDs; {r['status']}\n\n")
                for k in ['native_ply','supersplat_ply','removed_ply','report']:f.write(f"- [{k}]({r[k]})\n")
                f.write('\n')
    if invalidated:
        with (index/'ALL_GGGS_PLY_PATHS.md').open('a') as f:
            f.write('\n## Invalidated experiments excluded from results\n\n')
            for r in invalidated:
                f.write(f"- [{r['experiment']}]({r['experiment']}/INVALIDATED.json)\n")
    if integrity_failures:
        with (index/'ALL_GGGS_PLY_PATHS.md').open('a') as f:
            f.write('\n## Artifact integrity failures excluded from results\n\n')
            for r in integrity_failures:f.write(f"- {r['experiment']}: {r['reason']}\n")
    (index/'HOW_TO_VIEW.md').write_text('''# 查看实际 Gaussian 模型

在 SuperSplat 的 File → Import 中选择 `cleaned_gaussians_supersplat.ply`，也可把该文件拖入编辑器。
当前官方说明：[导入 Gaussian PLY](https://github.com/playcanvas/supersplat/wiki/Importing-3DGS-Scenes-into-your-Projects)。
编辑器的标准三轴尺度、四元数和SH属性来自[官方导入源码](https://github.com/playcanvas/supersplat/blob/main/src/editor-splat-resource.ts)。
文档已通过 Context7 拉取。优先查看 ALL_GGGS_PLY_PATHS.md 中的实际绝对路径。
目前核验的官方 SuperSplat 3.8.0 编辑器使用 WebGPU，见[官方启动源码](https://github.com/playcanvas/supersplat/blob/main/src/index.ts)。
导入后按 F 使用编辑器的 Frame Selection 将模型移入视野，见[官方快捷键源码](https://github.com/playcanvas/supersplat/blob/main/src/shortcut-manager.ts)。
已实际完成 C10-C 查看版的本地官方编辑器导入、526,202 点数核对及截图直接查看：
[实际浏览器核验](supersplat_browser_audit/C10_C_framed/BROWSER_AUDIT.json)。
此核验使用隔离临时浏览器和 CPU 软件 WebGPU，没有上传模型；仅代表该已实测文件，其他结果的实际浏览器核验见 CSV 对应字段。

查看版保留完整 SH，已经烘焙 GGGS Filter 的有效三轴尺度和 Alpha；原生 `cleaned_gaussians.ply`
继续保留有效 `filter_3D`，供同一 GGGS Renderer 和下游流水线使用。
滤波烘焙的实际原生渲染一致性保存在各结果的 viewer parity 报告中；它不等于浏览器截图的逐像素一致性测试。
不得用编辑器重新导出的标准模型覆盖原生 GGGS 文件。

原始 Gaussian ID 是输入 PLY 的零基行号，记录于 `original_gaussian_ids.npy`。
被删点查看版为 `removed_gaussians_supersplat.ply`，原生被删点为 `removed_gaussians.ply`。
被删点文件保存实际被删ID对应的原始输入参数；曾改参后再删除的行，其实际改参过程在参数patch中记录。
零删除结果的被删点文件为空，不能把空文件误认为模型加载失败或清理成功。

各组白底、黑底和 Alpha 图使用相同相机、分辨率；自动自由视角定义在结果文件中。
历史固定八视角仅用于历史参考评价。C3 最终版包含3592个历史人工标记排除点；
另有 `C3_pre_marker` 独立结果，避免把标记处理和背景清理混为一谈。
C1/C2/C5-A 仅外观规则的结果存在实际前景损失；`no_accepted_modification` 表示未清理。
`MODEL_READY.json` 仅表示模型已导出；完整评价通过后才有 `COMPLETE.json`。
''')
    selected_main=comp/'SELECTED_MAIN_MODEL.json'
    chosen='C5-C' if 'C5-C' in method_dirs else 'C5-D'
    if selected_main.exists():
        selection=json.loads(selected_main.read_text());chosen=selection['variant']
        selected_path=Path(selection['output']).resolve()
        if selected_path not in {path.resolve() for path in completed_dirs}:
            raise ValueError('Selected main model is not a valid completed experiment')
        selected_report=json.loads((selected_path/'cleanup_metrics.json').read_text())
        if selected_report['plant']!='KongQueZhuYu' or selected_report['method']!=chosen:
            raise ValueError('Selected main model does not match the primary GGGS plant/variant')
        if sha(selected_path/'cleaned_gaussians.ply')!=selection['native_sha256']:
            raise ValueError('Selected main native model differs from the decision')
        method_dirs[chosen]=selected_path
    labels=['C0','C1_adaptive','C2_adaptive','C3','C4',chosen]
    visual_records=[]
    if all(k in method_dirs for k in labels):
        visual=comp/'visualizations';visual.mkdir(exist_ok=True)
        definitions=[json.loads((method_dirs[k]/'visualizations/automatic_free_views.json').read_text())['definitions'] for k in labels]
        if any(value!=definitions[0] for value in definitions[1:]):
            raise ValueError('Main visual comparison has different automatic free cameras')
        for view in ['top','oblique','far','orbit_00','orbit_01']:
            for background in ['white','black']:
                images=[];source_images=[];resolution=None
                for k in labels:
                    path=method_dirs[k]/f'visualizations/free/{view}_{background}.png'
                    if not path.exists():break
                    with Image.open(path) as im:
                        if resolution is None:resolution=im.size
                        if im.size!=resolution:raise ValueError('Main comparison source renders use different resolutions')
                        fitted=ImageOps.contain(im.convert('RGB'),(360,640))
                        cell=Image.new('RGB',(360,640),'#dedede');cell.paste(fitted,((360-fitted.width)//2,(640-fitted.height)//2));images.append(cell)
                    source_images.append(dict(method=k,experiment=str(method_dirs[k].resolve()),path=str(path.resolve()),sha256=sha(path)))
                if len(images)!=6:continue
                canvas=Image.new('RGB',(1080,1348),'#dedede');draw=ImageDraw.Draw(canvas)
                for i,(k,im) in enumerate(zip(labels,images)):
                    x=(i%3)*360;y=(i//3)*674
                    draw.text((x+10,y+8),k,fill='black');canvas.paste(im,(x,y+32))
                panel=visual/f'{view}_{background}_C0_C5.png';canvas.save(panel)
                visual_records.append(dict(view=view,background=background,source_resolution=list(resolution),panel=str(panel.resolve()),sources=source_images))
    write_json(comp/'VISUAL_COMPARISON_SOURCE_BINDING.json',dict(chosen_variant=chosen,global_selection_recorded=selected_main.exists(),
        scope='Identical recorded automatic free cameras, original native render resolutions and display backgrounds. Panels resize equally for inspection. Historical eight cameras remain separate posthoc evaluations.',panels=visual_records))
    write_json(comp/'COMPARISON_STATUS.json',dict(completed=list(method_dirs),
        invalidated_experiments=invalidated,
        artifact_integrity_failures=integrity_failures,
        selected_completed_experiments={k:str(v.resolve()) for k,v in method_dirs.items()},
        pending=[k for k in ['C0','C1_raw','C1_adaptive','C2_raw','C2_adaptive','C3','C4','C5-A','C5-B','C5-C','C5-D'] if k not in method_dirs],
        warning='Only completed models have measured results; no generalization conclusion until fixed C5 independent tests'))


if __name__=='__main__':main()
