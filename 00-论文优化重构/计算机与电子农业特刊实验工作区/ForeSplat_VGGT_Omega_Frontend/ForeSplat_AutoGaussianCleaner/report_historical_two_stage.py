#!/usr/bin/env python3
"""Summarize measured two-stage evidence against the original19-plant scope."""
from pathlib import Path
import argparse,csv,json,sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.execution_provenance import gaussian_count
from viewer_review_validity import validated_review


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def change(old,new):
    return (old-new)/old*100 if old else 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    inventory=read(BASE/'configs/plant_manifest.json')['plants']
    manifest=read(root/'run_manifest.json') or {'plants':[]}
    requested={i['plant']:Path(i['output']) for i in manifest['plants']}
    historical_assets={}
    historical_rounds=[]
    kong_scene=read(root/'scene_configs/KongQueZhuYu.json')
    if kong_scene and kong_scene.get('baseline_models',{}).get('R5'):
        best=Path(kong_scene['baseline_models']['R5'])
        paths={'historical_best_native':best,
            'historical_best_supersplat':best.with_name('cleaned_gaussians_supersplat.ply'),
            'historical_round5_with_marker':best.parent.parent/'additional_rounds/round_05/gaussians.ply'}
        for label,path in paths.items():
            if path.is_file():historical_assets[label]=dict(path=str(path.resolve()),gaussians=gaussian_count(path),sha256=sha(path))
        source=best.parents[2]/'comparisons/gaussian_cleaning_metrics.csv'
        if source.exists():
            for record in csv.DictReader(source.open()):
                if record['scope']=='8 representative source + 8 fixed free views':
                    historical_rounds.append(dict(model=record['model'],gaussians=int(record['gaussians']),
                        black_pixels=int(record['free_black_pixels_sum']),source=str(source.resolve()),
                        selected_lineage=record['model'] not in ['Round3_shrink','Aggressive']))
    completed={};statuses=[];index={};reviewed={};audited={};selected_count=0
    for item in inventory:
        plant=item['plant'];options=[root/plant,root/'cross_plant'/plant,root/'all_19_plants'/plant]
        out=next((d for d in options if (d/'execution.json').exists()),requested.get(plant,options[0]))
        marker=read(out/'COMPLETE.json');r=read(out/'cleanup_metrics.json')
        status='pending'
        if marker and r and marker.get('source_and_input_verification'):
            completed[plant]=r;status='complete'
            index[plant]={k:str(Path(v).resolve()) for k,v in r['models'].items() if v and Path(v).is_file()}
            index[plant]['stage1_supersplat']=str((out/'stage1_supersplat.ply').resolve())
            index[plant]['stage2_supersplat']=str((out/'stage2_supersplat.ply').resolve())
        elif (out/'FAILED.json').exists():status='failed'
        elif (out/'execution.json').exists():status='incomplete'
        review=validated_review(out)
        if review:
            reviewed[plant]=review
            selected=read(out/'VISUAL_SELECTION.json')
            if selected and selected.get('selected_model')==review.get('recommended_model') and plant in index:
                for suffix,record in selected['models'].items():
                    index[plant]['selected_'+suffix]=record['path']
                selected_count+=1
        audit=read(out/'DELIVERY_AUDIT.json')
        audit_files=['COMPLETE.json','cleanup_metrics.json','comparison/official_supersplat/REVIEW.json',
            'comparison/official_supersplat/DIRECT_IMAGE_REVIEW.json','VISUAL_SELECTION.json']
        audit_current=bool(audit) and all((out/name).is_file() and audit.get('evidence_sha256',{}).get(name)==sha(out/name) for name in audit_files)
        if audit_current and audit.get('passed') and audit.get('verified_direct_browser_review') and audit.get('verified_selected_model'):
            audited[plant]=audit
        statuses.append(dict(plant=plant,status=status,output=str(out),actual_viewer_review=bool(review),
            visual_acceptance=review.get('visual_acceptance') if review else None,
            improved_with_residuals=review.get('eligible_for_batch') if review else None))
    visual_failures=[p for p,r in reviewed.items() if r.get('visual_acceptance') is False
        or not r.get('eligible_for_batch',False)]
    k=completed.get('KongQueZhuYu')
    lines=['# ForeSplat 两阶段 Gaussian 清理实测报告','',
        f'当前完成 {len(completed)}/{len(inventory)} 株；实际编辑器截图直接审阅记录 {len(reviewed)}/{len(inventory)} 株。',
        f'已记录的视觉失败案例：{visual_failures}。固定参数批量评估不等于全部模型视觉验收通过。',
        '此报告按真实完成文件生成。未完成、未审阅和未通过历史复现门槛的结果不计为全部任务完成。','',
        '## 1. 为什么历史 Round5 比 C5-C 更干净？','',
        '历史方法在删除后重新测量原生遮挡贡献，继续删除新暴露背景层。已完成 C5-C 的记录为437次候选试验、400次拒绝；较严格的前景约束使部分高背景危害的混合 Gaussian 留在模型中。历史方法接受了更大的局部光度变化。历史结果仍有已记录的盆沿和暗部光度问题，不能据此声称其所有指标都更优。','',
        '## 2. 哪些历史机制产生主要收益？','',
        '原生 T×Alpha 颜色雅可比归因、前景/背景贡献比、删除后的重新归因是主线。历史Round4/5分别删除12505/1208个新暴露体元；Round3整体缩放未进入最佳模型。独立删除3592个标记物体元与背景清理分开报告。','',
        '## 3. 通用Stage1是否复现历史效果？','']
    if k:
        s1=k['stage1'];s2=k['stage2'];x=s1['final']['free_background'];y=s2['initial']['free_background']
        lines.append(f'自动自由8视角：Stage1黑色伪影像素{x["black_pixels"]}；历史最佳{y["black_pixels"]}。Stage1最终回退={s1["rolled_back"]}。')
        lines.append('两者使用同一自动自由视角协议；历史固定8视角的数值另列，不能跨协议比较。')
        for label,folder in [('Stage1','stage1_generalized/visualizations'),('历史最佳','stage2_residual/visualizations_input'),('Stage2','stage2_residual/visualizations')]:
            m=read(root/'KongQueZhuYu'/folder/'historical_free_metrics.json')
            if m:lines.append(f'历史固定8视角 {label} 黑色像素：{m["black_pixels"]}。')
        lines.extend(['','## 4. 二次清理能否减少剩余黑点？','',
            f'孔雀竹芋Stage2自动自由黑色像素：{y["black_pixels"]} → {s2["final"]["free_background"]["black_pixels"]}；减少{change(y["black_pixels"],s2["final"]["free_background"]["black_pixels"]):.2f}%。',
            '数值收益不能证明所有悬浮黑点或叶旁黑条已消失；实际编辑器审阅见下表记录。','',
            '## 5. Stage2是否伤害叶缘或暗色叶片？',''])
        before,after=s2['initial']['mean'],s2['final']['mean']
        for key in ['fg_psnr','boundary_f1','thin_coverage','skeleton_coverage','dark_foreground_mse','depth_changed_fraction','normal_angle_p95_deg']:
            lines.append(f'- {key}: {before.get(key)} → {after.get(key)}')
        lines.extend(['',f'全视角拒绝原因：{s2["full_view_rejection_reasons"]}；最终回退：{s2["rolled_back"]}。这些指标与截图共同判断，不据平均值宣称不存在任何局部误差。'])
    else:
        lines.extend(['孔雀竹芋完整实验尚未完成，复现结论未成立。','','## 4. 二次清理能否减少剩余黑点？','','完整结果待测。','','## 5. Stage2是否伤害叶缘或暗色叶片？','','完整视角质量和实际截图审阅待完成。'])
    if historical_rounds:
        lines.extend(['','历史固定8自由视角原始记录如下。Round3_shrink和Aggressive属于未采用的分支，不能把表中顺序解释为最佳模型的连续修改链。','',
            '| 历史模型 | Gaussian数 | 固定8黑色像素 | 最优结果采用的谱系 |',
            '|---|---:|---:|---|',
            *[f'| {r["model"]} | {r["gaussians"]} | {r["black_pixels"]} | {r["selected_lineage"]} |' for r in historical_rounds]])
    lines.extend(['','## 6. 两个阶段各有多少清理收益？','',
        '| 植物 | Stage1删除/修改 | Stage2删除/修改 | Stage1黑色像素前→后 | Stage2黑色像素前→后 | Stage1/2秒 |',
        '|---|---:|---:|---:|---:|---:|'])
    for plant,r in completed.items():
        s1,s2=r['stage1'],r['stage2']
        black=lambda s:f'{s["initial"]["free_background"]["black_pixels"]}→{s["final"]["free_background"]["black_pixels"]}'
        lines.append(f'| {plant} | {s1["removed_gaussians"]}/{s1["refined_gaussians"]} | {s2["removed_gaussians"]}/{s2["refined_gaussians"]} | {black(s1)} | {black(s2)} | {s1["seconds"]:.1f}/{s2["seconds"]:.1f} |')
    lines.extend(['','孔雀竹芋Stage2直接从历史最佳开始；其两个阶段收益不能当作同一条串联结果相加。其他植物的Stage2自动使用Stage1输出。','',
        '背景Alpha、外部高Alpha面积和单视角最大悬浮组件前后值见 `metrics/background_before_after.csv`。其中最大组件取八视角中的最大值；底层free_background.largest_component字段是各视角最大组件之和，两种汇总方式不混用。外部区域由独立深度投影代理定义，不是真实自由视角标注。','',
        '原始Thin Coverage和证据支持的Thin Coverage分开报告。第八版对无原始照片/深度支持且RGB未变差的Alpha尾部损失单独计量；它会影响验收判断，不能将这种规则变化表述成原始覆盖率完全不变。','',
        '| 植物/阶段 | FG-PSNR前→后 | 原始Thin前→后 | 支持Thin前→后 | 无支持且RGB未变差的Thin损失 | 回退 |',
        '|---|---:|---:|---:|---:|---|'])
    for plant,r in completed.items():
        for stage in ['stage1','stage2']:
            s=r[stage];before=s['initial']['mean'];after=s['final']['mean']
            pair=lambda key:f'{before.get(key,0):.6f}→{after.get(key,0):.6f}'
            lines.append(f'| {plant}/{stage} | {pair("fg_psnr")} | {pair("thin_coverage")} | {pair("supported_thin_coverage")} | {after.get("unsupported_nonworsening_thin_loss",0):.6f} | {s["rolled_back"]} |')
    lines.extend(['',
        '## 7. 比C5-C快多少？',''])
    if k:
        lines.append(f'已有C5-C记录13754.18秒；本次孔雀竹芋两阶段程序内计时{k["seconds"]:.2f}秒，观察到的耗时比为{13754.1792047/max(k["seconds"],1):.2f}。')
    lines.extend(['启动时间、共享磁盘和运行负载不同；清理质量也须单独核对。因此该比值是实测耗时对照，不能直接表述为同质量加速倍数。','',
        '## 8. 同一流程能否适用于其他植物？','',
        f'已完成其他形态植物{sum(p!="KongQueZhuYu" for p in completed)}株。未完成的跨植株或未参与调参植株不作泛化成功结论。',
        f'冻结协议文件：{root/"FROZEN_HELDOUT_PROTOCOL.json"}；实际存在={(root/"FROZEN_HELDOUT_PROTOCOL.json").exists()}。','',
        '## 9. 哪些植物仍有明显伪影？',''])
    if reviewed:
        for plant,review in reviewed.items():
            lines.append(f'{plant}：'+'；'.join(review.get('observations',[])))
    else:lines.append('本版本尚无已保存的直接截图审阅记录，不能从点数或平均分推断视觉上没有明显伪影。')
    lines.extend(['','## 10. 最终使用哪份PLY？','',
        '所有已完成模型的真实绝对路径见 `model_index/all_models_absolute.json`。原生GGGS使用native路径，SuperSplat打开对应supersplat路径。最佳模型推荐还须通过本版本实际编辑器审阅；未完成版本不替代历史资产。','',
        '本地 `REVIEW.html` 可切换已审阅植株、八个自由视角和白/黑底，并直接链接推荐文件；孔雀竹芋附有历史标记物版本对照。交互打开模型的步骤见 `VIEWING_GUIDE.md`。','',
        *[f'{plant}：选择 `{review.get("recommended_model")}`。原生入口：`{index[plant].get("selected_native")}`；SuperSplat入口：`{index[plant].get("selected_supersplat")}`。' for plant,review in reviewed.items() if plant in index and index[plant].get('selected_native')],
        *[f'{label}（{record["gaussians"]}点，保留历史资产）：`{record["path"]}`。' for label,record in historical_assets.items()],
        '', '本次实际编辑器对照使用记录在REVIEW.json中的独立兼容构建：保留的本地官方3.8.0源码与splat-transform3.11.1组合存在重复sigmoid；单行修复通过已知Alpha实拍测试。模型文件仍使用标准Opacity logit，未通过篡改PLY来补偿编辑器。其他编辑器版本需按其实际导入行为判断。','',
        '## 11. 是否可以一键完成全部植物？','',
        f'统一入口与串行批处理程序已实现；本目录实际完成数为{len(completed)}/{len(inventory)}。只有全部任务的完成文件、全视角评价和视觉审阅齐备，才能宣称全部交付完成。','',
        '```bash','CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python3 python \\',
        '  ForeSplat_AutoGaussianCleaner/run_historical_two_stage.py \\',f'  --output "{root}"','```','',
        '| 植物 | 实测状态 | 编辑器直接审阅 | 输出目录 |','|---|---|---|---|',
        *[f'| {s["plant"]} | {s["status"]} | {s["actual_viewer_review"]} | {s["output"]} |' for s in statuses],
        '', '来源：本地原生GGGS渲染、各轮贡献与实际编辑记录、全视角metrics及实际SuperSplat截图。联网搜索不可用：Bocha返回HTTP403额度不足；以上实验结论来自本地实测，不是基于未核实网页的断言。'])
    write_json(root/'model_index/all_models_absolute.json',index)
    write_json(root/'model_index/historical_assets.json',historical_assets)
    write_json(root/'metrics/historical_rounds_source.json',historical_rounds)
    background_rows=[]
    for plant,result in completed.items():
        for stage in ['stage1','stage2']:
            row=dict(plant=plant,stage=stage)
            for moment in ['initial','final']:
                bg=result[stage][moment]['free_background']
                for key in ['alpha_mass','high_alpha_pixels','black_pixels','aabb_alpha_mass']:
                    row[f'{moment}_{key}']=bg[key]
                row[f'{moment}_max_single_view_component']=max((v['largest_component'] for v in bg['views']),default=0)
                row[f'{moment}_sum_view_largest_components']=bg['largest_component']
                row['protocol']=bg['protocol']
            background_rows.append(row)
    if background_rows:
        with (root/'metrics/background_before_after.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(background_rows[0]));writer.writeheader();writer.writerows(background_rows)
    scope_complete=(len(completed)==len(inventory) and len(reviewed)==len(inventory)
        and len(audited)==len(inventory) and selected_count==len(inventory)
        and (root/'FROZEN_HELDOUT_PROTOCOL.json').is_file())
    write_json(root/'metrics/delivery_status.json',dict(expected_plants=len(inventory),completed_plants=len(completed),
        direct_reviewed_plants=len(reviewed),audited_plants=len(audited),selected_plants=selected_count,
        statuses=statuses,scope_complete=scope_complete,visual_failures=visual_failures,
        universal_visual_success=scope_complete and all(r.get('visual_acceptance') is True for r in reviewed.values())))
    (root/'FINAL_TWO_STAGE_CLEANER_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote measured report: {len(completed)}/{len(inventory)} complete; {len(reviewed)} directly reviewed')


if __name__=='__main__':main()
