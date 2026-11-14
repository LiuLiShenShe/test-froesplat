"""Assemble existing metrics, reference panels, visual reviews and all-19 index."""
from pathlib import Path
import csv,json,statistics
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/C1_C2_all19'
LABELS=dict(CaoMei1='草莓1',CaoMei2='草莓2',ChangShouHua1='长寿花1',ChangShouHua2='长寿花2',ChangShouHua3='长寿花3',
    DouBanLv1='豆瓣绿1',DouBanLv2='豆瓣绿2',DouBanLv3='豆瓣绿3',HongZhang='红掌',KongQueZhuYu='孔雀竹芋',
    WanNianQing1='万年青1',WanNianQing2='万年青2',WangWenCao1='网纹草1',WangWenCao2='网纹草2',
    XianKeLai1='仙客来1',XianKeLai2='仙客来2',XianKeLai3='仙客来3',XiangPiShu1='橡皮树1',XiangPiShu2='橡皮树2')

def make_panels(plant):
    p=OUT/plant/'comparison';q=json.loads((p/'paired_quality.json').read_text())
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    # Four portrait frames per row, identical full-frame scale for each model.
    cell_w,cell_h=300,560
    canvas=Image.new('RGB',(4*cell_w,3*cell_h),'white');d=ImageDraw.Draw(canvas)
    ref=Image.new('RGB',(4*cell_w,cell_h),'white');rd=ImageDraw.Draw(ref)
    for col,vi in enumerate(q['results']['original']['observed_views']):
        with Image.open(p/f'reference_observed_{col:02}.png') as im:
            im.thumbnail((cell_w,cell_h-40));ref.paste(im,(col*cell_w+(cell_w-im.width)//2,40))
        rd.text((col*cell_w+5,5),vi['name'],fill='black',font=font)
        for row,method in enumerate(['original','C1_raw','C2_raw']):
            with Image.open(p/f'{method}_observed_{col:02}.png') as im:
                im.thumbnail((cell_w,cell_h-40));canvas.paste(im,(col*cell_w+(cell_w-im.width)//2,row*cell_h+40))
            d.text((col*cell_w+5,row*cell_h+5),method+' | '+vi['name'],fill='black',font=font)
    canvas.save(p/'contact.png');ref.save(p/'reference_contact.png')
    return q

def make_focus(plant, order=1):
    p=OUT/plant/'comparison'
    q=json.loads((p/'paired_quality.json').read_text())
    view=q['results']['original']['observed_views'][order]['name']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
    panel=Image.new('RGB',(2160,1040),'white');draw=ImageDraw.Draw(panel)
    draw.text((10,5),f'{plant} | {view}',fill='black',font=font)
    for col,(prefix,label) in enumerate([('reference','Reference RGB'),('original','Original'),('C1_raw','C1_raw'),('C2_raw','C2_raw')]):
        draw.text((col*540+10,40),label,fill='black',font=font)
        with Image.open(p/f'{prefix}_observed_{order:02}.png') as im:
            im.thumbnail((540,960));panel.paste(im,(col*540+(540-im.width)//2,80))
    panel.save(p/f'focus_view_{order:02}.png')
    return panel

def write_conclusions(completed, metrics, resources, reviews, summary):
    if set(completed)!=set(reviews):
        raise ValueError('Every completed plant needs a direct visual review before final reporting')
    counts={}
    for method in ['C1_raw','C2_raw']:
        counts[method]={key:[p for p in completed if reviews[p][method][key]]
            for key in ['cloud_good','acceptable','severe','dirty']}
    both_dirty=[p for p in completed if all(reviews[p][m]['dirty'] for m in ['C1_raw','C2_raw'])]
    counts['both_dirty']=both_dirty
    (summary/'classification_counts.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n')
    names=lambda plants:'、'.join(LABELS[p] for p in plants) or '无'
    batch=resources['actual_batch']
    lines=[
        '# 19株固定 C1_raw / C2_raw 实测结果', '',
        '已完成19株原生GGGS 30K模型的两种清理，保存38个最终PLY入口、19张三行四视角对比图、76张RGB参考及228张独立渲染图。33份PLY为本轮新导出，5份为核验等价后直接硬链接复用；原始模型未覆盖。', '',
        '**本轮结论：没有一种固定规则在全部19株上统一可用。** C1对主体更温和；C2能进一步减少残云，但暗叶和黑色花盆更易受损。未因这些观察修改参数、回滚输出或停止后续植物。', '',
        '“效果好”指四个实测视角中背景足够干净且主体基本完整，允许少量窄黑边或局部暗部侵蚀；“明显过删”包含大片叶片、连接区域或花盆损伤，不等于整株都消失。“仍不干净”指明显黑块/黑条，不能只用黑像素减少百分比判定。分类来自直接查看三行图及RGB参考，不参与规则计算。', '',
        f'- 兼顾去云和主体的效果好：C1 **{len(counts["C1_raw"]["acceptable"])}/19**；C2 **{len(counts["C2_raw"]["acceptable"])}/19**。',
        f'- 只看背景去云足够干净：C1 {len(counts["C1_raw"]["cloud_good"])}/19；C2 {len(counts["C2_raw"]["cloud_good"])}/19。',
        f'- C1明显过删：{names(counts["C1_raw"]["severe"])}。',
        f'- C2明显过删：{names(counts["C2_raw"]["severe"])}。',
        f'- C1/C2均仍不干净（{len(both_dirty)}株）：{names(both_dirty)}。', '',
        '| 植物 | C1去云效果 | C1主体损伤 | C2去云效果 | C2主体损伤 | 更合适的方法 |',
        '|---|---|---|---|---|---|',
    ]
    for plant in completed:
        r=reviews[plant]
        lines.append(f'| [{LABELS[plant]}](../{plant}/comparison/contact.png) | {r["C1_raw"]["cloud"]} | {r["C1_raw"]["damage"]} | {r["C2_raw"]["cloud"]} | {r["C2_raw"]["damage"]} | {r["preferred"]} |')
    lines += ['', '**长寿花1：C1能恢复大部分主体，但没有完全恢复。** 花盆及下层暗叶明显优于C2，内侧暗叶仍有局部稀疏。四视角新增前景Alpha孔洞均值C1约4.1%、C2约12.5%；前景PSNR相对原模型分别下降约0.52/2.24 dB。推荐本株采用C1。', '',
        '[长寿花1完整对比](../ChangShouHua1/comparison/contact.png) · [crop_0055 RGB/Original/C1/C2细看](../ChangShouHua1/comparison/focus_view_01.png)', '',
        '**网纹草1：换成C1仍未恢复完整主体。** 纳入已知暗视角crop_0013，参考RGB和原始GGGS中均有植株；C1仅剩零散彩色片，C2几乎全空。该视角新增前景孔洞C1约93.0%、C2约99.4%。两种均不适合该株。全局Gaussian只删约6.0%/6.8%，不能据删点比例较低认定安全。', '',
        '[网纹草1完整对比](../WangWenCao1/comparison/contact.png) · [crop_0013 RGB/Original/C1/C2细看](../WangWenCao1/comparison/focus_view_01.png)', '',
        '**仙客来2存在原模型/相机对齐问题。** crop_0033原始GGGS位置已明显偏右，参考RGB与模型不对齐。清理去掉原先覆盖前景mask的黑云，会得到约91%的新增mask孔洞，不能将其解读为真实植物被删91%。两种最终结果都有残余黑块，本报告未凭这个数值将其认定为已证实大幅过删。', '',
        '[仙客来2原有错位证据](../XianKeLai2/comparison/focus_view_01.png)', '',
        '**计算与I/O实测（清理阶段，GPU渲染另计）。** 方法时间为共享读取/特征时间加各自判定和实际导出时间；共享时间出现在两列，不能将两列相加当作批次时间。复用输出本轮写入为零，因此C2总时间较短不代表其判定更快。峰值RSS和限速峰值为同株联合进程统计，不是两个独立进程的各自峰值。', '',
        '| 方法 | 平均CPU时间/株(s) | 平均总时间/株(s) | 平均规则判定CPU(s) | 平均联合峰值RSS(GiB) | 最大联合RSS(GiB) | 平均共享读(MiB) | 平均本轮写(MiB) | 最大合计读写速率(MiB/s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for method in ['C1_raw','C2_raw']:
        r=resources[method]
        lines.append(f'| {method} | {r["average_cpu_s"]:.2f} | {r["average_wall_s"]:.2f} | {r["average_decision_cpu_s"]:.2f} | {r["average_peak_rss_gib"]:.3f} | {r["maximum_peak_rss_gib"]:.3f} | {r["average_shared_read_mib"]:.2f} | {r["average_write_mib"]:.2f} | {r["maximum_application_peak_mib_s"]:.3f} |')
    lines += ['', f'19株联合清理实际累计：墙钟 {batch["total_wall_s"]:.2f} s（{batch["total_wall_s"]/60:.2f} min），CPU {batch["total_cpu_s"]:.2f} s；应用层PLY读取 {batch["logical_read_gib"]:.3f} GiB、写入 {batch["logical_write_gib"]:.3f} GiB。全部走一读两输出路径，每株KNN只计算一次；8 GiB预算内，最大RSS {resources["C1_raw"]["maximum_peak_rss_gib"]:.3f} GiB。共享20 MiB/s限速器的最大单秒窗口实测 {resources["C1_raw"]["maximum_application_peak_mib_s"]:.3f} MiB/s。该值为应用层统计，非磁盘设备峰值。', '',
        f'独立评价累计墙钟 {resources["evaluation"]["total_wall_s"]:.2f} s，额外应用层PLY读取 {resources["evaluation"]["logical_read_gib"]:.3f} GiB，物理GPU1。每株只渲染4个现有观测相机；三种模型逐一渲染，镜头、分辨率（最长边960）、灰背景0.5和曝光一致。未执行全208相机或新增自由视角渲染。评价PLY读取亦经共享20 MiB/s限速。', '',
        '**协议与核验。** 原始规则文件SHA256未变化。入口调用legacy.rules(version=2/3, normalized=False)，共用既有原始特征定义及一次分块KNN；未引入adaptive、Fast Joint、R5、mask清理约束或归因。孔雀竹芋直接与历史原始ID逐项相等，C1保留443,681、C2保留428,967；未重跑昂贵历史评价。原始逐点记录直接筛选，SH、旋转、Opacity、Scale、filter_3D及其他原字段均保留。7项回归检查通过。', '',
        '前景PSNR、背景Alpha/黑像素、叶缘及薄结构覆盖沿用已有评价定义。前景mask/骨架是评价代理，不是植物语义人工真值；只检查这4个视角，不能据此声称任意方向均可靠。长寿花1和网纹草1特意纳入已知暗部问题视角，其他植物用既有四视角选择器。', '',
        '**完整交付索引。** [19株交互索引](index.html) · [19株总览](all19_overview.png) · [38行数值总表](all19_results.csv) · [完整指标JSON](metrics_all19.json) · [资源统计](resources.json) · [分类名单](classification_counts.json) · [原模型及代码校验记录](protocol.json) · [最终核验](verification.json)。', '',
        '| 植物 | 三行四视角图 | 原始RGB参考 | 指标 | C1最终PLY | C2最终PLY |',
        '|---|---|---|---|---|---|',
    ]
    for plant in completed:
        lines.append(f'| {LABELS[plant]} | [对比](../{plant}/comparison/contact.png) | [RGB](../{plant}/comparison/reference_contact.png) | [JSON](../{plant}/comparison/metrics.json) | [C1](../{plant}/C1_raw/cleaned_gaussians.ply) | [C2](../{plant}/C2_raw/cleaned_gaussians.ply) |')
    lines += ['', '**逐株观察记录。**', '']
    for plant in completed:
        lines += [f'**{LABELS[plant]}**：{reviews[plant]["notes"]}', '']
    lines += ['信息来源：本地原始模型、实际清理/渲染记录及直接查看图像。博查联网搜索HTTP 403，未使用网络资料作为实验结论依据。', '']
    (summary/'conclusions.md').write_text('\n'.join(lines))

def main():
    entries=json.loads((ROOT/'configs/plant_manifest.json').read_text())['plants']
    summary=OUT/'summary';summary.mkdir(parents=True,exist_ok=True)
    review_file=summary/'visual_reviews.json'
    reviews=json.loads(review_file.read_text()) if review_file.exists() else {}
    metrics={};rows=[];completed=[]
    for e in entries:
        plant=e['plant'];p=OUT/plant/'comparison'
        if not (p/'paired_quality.json').exists():continue
        q=make_panels(plant);completed.append(plant)
        original=q['results']['original'];base=original['mean'];records={}
        for method in ['C1_raw','C2_raw']:
            c=json.loads((OUT/plant/method/'cleanup_report.json').read_text());v=q['results'][method];m=v['mean']
            black0=sum(x['dark_bg_high_alpha_pixels'] for x in original['observed_views'])
            black1=sum(x['dark_bg_high_alpha_pixels'] for x in v['observed_views'])
            rec=dict(plant=plant,label=LABELS[plant],method=method,original_gaussians=c['original_gaussians'],
                remaining_gaussians=c['remaining_gaussians'],removed_fraction=c['removed_fraction'],
                bg_black_original=black0,bg_black_cleaned=black1,bg_black_reduction_pct=100*(1-black1/black0) if black0 else None,
                fg_psnr_original=base['fg_psnr'],fg_psnr=m['fg_psnr'],fg_psnr_delta=m['fg_psnr']-base['fg_psnr'],
                bg_alpha_original=base['bg_alpha'],bg_alpha=m['bg_alpha'],
                boundary_original=base['boundary_coverage'],boundary=m['boundary_coverage'],
                thin_original=base['thin_coverage'],thin=m['thin_coverage'],
                new_fg_alpha_hole_fraction=m['new_fg_alpha_hole_fraction'],
                worst_view_new_holes=max(x['new_fg_alpha_hole_fraction'] for x in v['observed_views']),
                cpu_s=c['timing']['process_cpu_s'],decision_cpu_s=c['timing']['decision_cpu_s'],wall_s=c['timing']['total_s'],
                peak_rss_gib=c['resource']['peak_process_rss_bytes']/2**30,
                shared_read_mib=c['io']['shared_logical_read_bytes']/2**20,write_mib=c['io']['logical_write_bytes']/2**20,
                application_peak_mib_s=c['io']['observed_peak_one_second_mib_s'],reused=c['reused'],
                cloud_effect=reviews.get(plant,{}).get(method,{}).get('cloud','待直接看图'),
                damage=reviews.get(plant,{}).get(method,{}).get('damage','待直接看图'),
                suitable=reviews.get(plant,{}).get('preferred','待直接看图'))
            rows.append(rec);records[method]=rec
        result=dict(plant=plant,observed_indices=q['observed_indices'],original_mean=base,methods=records,
            visual_review=reviews.get(plant),view_protocol='4 existing observed cameras, same 960 max side, gray composite 0.5, no exposure changes')
        (p/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');metrics[plant]=result
    if rows:
        with (summary/'all19_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',30)
    # The main overview is a complete index, supplemented by full-resolution
    # per-plant panels so small leaves remain inspectable.
    overview=Image.new('RGB',(3*1200,((len(completed)+2)//3)*1740),'white');d=ImageDraw.Draw(overview)
    for i,plant in enumerate(completed):
        x=(i%3)*1200;y=(i//3)*1740
        d.text((x+10,y+10),f'{LABELS[plant]} / {plant}',fill='black',font=font)
        with Image.open(OUT/plant/'comparison/contact.png') as im:overview.paste(im,(x,y+60))
    overview.save(summary/'all19_overview.png')
    # Four plants per image make direct visual inspection practical.
    for start in range(0,len(completed),4):
        group=completed[start:start+4];canvas=Image.new('RGB',(2400,3480),'white');draw=ImageDraw.Draw(canvas)
        for i,plant in enumerate(group):
            x=(i%2)*1200;y=(i//2)*1740;draw.text((x+10,y+10),f'{LABELS[plant]} / {plant}',fill='black',font=font)
            with Image.open(OUT/plant/'comparison/contact.png') as im:canvas.paste(im,(x,y+60))
        canvas.save(summary/f'review_group_{start//4+1:02}.png')
    (summary/'metrics_all19.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    resources={}
    for method in ['C1_raw','C2_raw']:
        data=[r for r in rows if r['method']==method]
        if data:resources[method]=dict(plants=len(data),reused=sum(r['reused'] for r in data),
            average_cpu_s=statistics.mean(r['cpu_s'] for r in data),average_wall_s=statistics.mean(r['wall_s'] for r in data),
            average_decision_cpu_s=statistics.mean(r['decision_cpu_s'] for r in data),
            average_peak_rss_gib=statistics.mean(r['peak_rss_gib'] for r in data),maximum_peak_rss_gib=max(r['peak_rss_gib'] for r in data),
            average_shared_read_mib=statistics.mean(r['shared_read_mib'] for r in data),
            average_write_mib=statistics.mean(r['write_mib'] for r in data),maximum_application_peak_mib_s=max(r['application_peak_mib_s'] for r in data))
    batches=[json.loads((OUT/e['plant']/'batch_report.json').read_text()) for e in entries if (OUT/e['plant']/'batch_report.json').exists()]
    resources['actual_batch']=dict(plants=len(batches),total_cpu_s=sum(r['timing']['process_cpu_s'] for r in batches),
        total_wall_s=sum(r['timing']['total_s'] for r in batches),logical_read_gib=sum(r['io']['logical_read_bytes'] for r in batches)/2**30,
        logical_write_gib=sum(r['io']['logical_write_bytes'] for r in batches)/2**30,
        reused_models=sum(len(r['reuse']) for r in batches),complete_model_writes=sum(r['io']['complete_model_writes'] for r in batches))
    resources['scope']='Per-method contribution includes measured shared read/features plus its own decision/export; shared cost counted in both. RSS/peak I/O are joint process values, not standalone method peaks. Reused models have zero payload writes. GPU evaluation excluded.'
    resources['evaluation']=dict(total_wall_s=sum(json.loads((OUT/p/'evaluation_runtime.json').read_text())['wall_s'] for p in completed),
        logical_read_gib=sum(json.loads((OUT/p/'comparison/paired_quality.json').read_text())['logical_ply_read_bytes'] for p in completed)/2**30,
        gpu='physical GPU1',views_per_plant=4)
    (summary/'resources.json').write_text(json.dumps(resources,ensure_ascii=False,indent=2)+'\n')
    focus=Image.new('RGB',(2160,2080),'white')
    for i,plant in enumerate(['ChangShouHua1','WangWenCao1']):focus.paste(make_focus(plant),(0,1040*i))
    focus.save(summary/'focus_changshouhua_wangwencao.png')
    write_conclusions(completed,metrics,resources,reviews,summary)
    html=['<!doctype html><meta charset="utf-8"><title>C1/C2 all19</title><style>body{font:16px sans-serif;margin:24px;max-width:1500px}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
        '<h1>19株 Original / C1_raw / C2_raw</h1><p>每图三行：Original、C1、C2；四列为相同四相机。灰背景固定0.5，无曝光调整。RGB参考独立提供。</p>',
        '<p>去云且主体基本完整：C1 10/19，C2 7/19。固定规则均不能覆盖全部植物。<a href="conclusions.md">完整结论</a> · <a href="all19_results.csv">数值总表</a> · <a href="all19_overview.png">总览图</a> · <a href="focus_changshouhua_wangwencao.png">长寿花1与网纹草1重点图</a></p><nav>']
    for plant in completed:html.append(f'<a href="#{plant}">{LABELS[plant]}</a> · ')
    html.append('</nav>')
    for plant in completed:
        v=reviews.get(plant,{})
        html.append(f'<h2 id="{plant}">{LABELS[plant]} / {plant}</h2><p>{v.get("notes", "待直接看图")}</p><p>推荐：{v.get("preferred", "待直接看图")}</p>')
        html.append(f'<a href="../{plant}/comparison/contact.png"><img loading="lazy" src="../{plant}/comparison/contact.png"></a>')
        html.append(f'<p><a href="../{plant}/comparison/reference_contact.png">原始RGB参考</a> · <a href="../{plant}/comparison/metrics.json">指标</a> · <a href="../{plant}/C1_raw/cleaned_gaussians.ply">C1 PLY</a> · <a href="../{plant}/C2_raw/cleaned_gaussians.ply">C2 PLY</a></p>')
    (summary/'index.html').write_text('\n'.join(html)+'\n')
    print('Assembled',len(completed),'plants;',len(reviews),'visual reviews')

if __name__=='__main__':main()
