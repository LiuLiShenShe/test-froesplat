"""Build reviewable tables from actual artifacts; missing metrics stay empty."""
import csv
import html
import io
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime import OUT, Meter, write

METHODS=['3DGS','2DGS','GGGS','gsplat','MCMC_3DGS']
VIEWS=['0000','0048','0096','0144','0192','0208']


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def common_depth_display():
    from matplotlib import colormaps
    meter=Meter();protocol=[]
    for name in VIEWS:
        mask=np.asarray(Image.open(OUT/'common_input/dataset/rap_masks'/f'{name}.png'))>127
        data={}
        for method in METHODS:
            path=OUT/'gaussian'/method/'evaluation'/f'{name}_geometry.npz'
            if path.exists():
                with np.load(io.BytesIO(meter.read(path))) as d:data[method]=(d['depth_camera_z'],d['valid'])
        if not data:continue
        values=[z[valid&mask] for z,valid in data.values() if (valid&mask).any()]
        if not values:continue
        lo,hi=np.quantile(np.concatenate(values),[.02,.98])
        for method,(z,valid) in data.items():
            colour=colormaps['viridis'](np.clip((z-lo)/max(hi-lo,1e-8),0,1))[...,:3];colour[~valid]=0
            im=Image.fromarray(np.rint(colour*255).astype('uint8'));b=io.BytesIO();im.save(b,format='PNG')
            meter.save(OUT/'gaussian'/method/'evaluation'/f'{name}_Depth.png',b.getbuffer())
        protocol.append(dict(view=name,methods=list(data),camera_z_min=float(lo),camera_z_max=float(hi)))
    write(OUT/'summary/depth_display_protocol.json',dict(views=protocol,colour='viridis; same range per camera across all available methods; pooled RAP-valid depth 2/98 percentiles; invalid black; numeric depths unchanged'))


def overview(channel):
    tilew,tileh=212,379
    canvas=Image.new('RGB',(170+tilew*len(VIEWS),(tileh+28)*6+36),'#ecf0f3')
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    for col,name in enumerate(VIEWS):draw.text((175+col*tilew,5),name,fill='black',font=font)
    for row,label in enumerate([('RAP mask' if channel=='Alpha' else 'Original RGB')]+METHODS):
        y=36+row*(tileh+28);draw.text((8,y+12),label,fill='black',font=font)
        for col,name in enumerate(VIEWS):
            if row==0:
                path=OUT/'common_input/dataset/rap_masks'/f'{name}.png' if channel=='Alpha' else OUT/'common_input/dataset/images'/f'{name}.jpg'
            else:path=OUT/'gaussian'/label/'evaluation'/f'{name}_{channel}.png'
            x=170+col*tilew
            if path.exists():
                image=Image.open(path).convert('RGB');image.thumbnail((tilew,tileh));canvas.paste(image,(x,y))
            else:draw.text((x+12,y+tileh//2),'Pending',fill='#69717c',font=font)
    buffer=io.BytesIO();canvas.save(buffer,format='PNG');Meter().save(OUT/'summary'/f'gaussian_{"overview" if channel=="RGB" else channel}.png',buffer.getbuffer())


def mesh_artifacts():
    rows=[];folders=[]
    for method in ['NKSR','TSDF','Poisson','native']:
        root=OUT/'mesh'/method
        if not root.exists():continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not (folder/'complete.json').exists():continue
            info=read(folder/'complete.json');evaluation=read(folder/'evaluation/metrics.json')
            resources=info.get('report',{}).get('resources',info)
            observed=read(OUT/'summary'/f'mesh_{method}_{folder.name}_resources.json')
            for row in evaluation:
                rows.append(dict(**row,extraction_wall_s=info.get('wall_s',''),
                    extraction_process_wall_s=observed.get('wall_s',''),
                    extraction_cpu_s=resources.get('process_cpu_s',resources.get('cpu_s','')),
                    peak_rss_gib=resources.get('peak_rss_gib',''),
                    peak_cuda_allocated_gib=resources.get('gpu_peak_allocated_gib',resources.get('peak_allocated_gib','')),
                    observed_gpu1_peak_mib=observed.get('peak_gpu_device_mib',''),
                    io_read_bytes=resources.get('logical_read_bytes',resources.get('io',{}).get('application_read_bytes','')),
                    io_write_bytes=resources.get('logical_write_bytes',resources.get('io',{}).get('application_write_bytes','')),
                    io_peak_mib_s=resources.get('observed_peak_one_second_mib_s',resources.get('io',{}).get('peak_one_second_mib_s','')),
                    oriented_input_sha256=info.get('oriented_input_sha256',''),
                    extraction_mode='Official overlapping chunks' if info.get('chunking') else ('GGGS native tetrahedra' if method=='native' else method)))
            folders.append(folder)
    if rows:csvwrite(OUT/'summary/mesh_comparison.csv',rows)
    if folders:
        canvas=Image.new('RGB',(180+1060,(1+len(folders)*2)*500),'#ecf0f3');draw=ImageDraw.Draw(canvas)
        draw.text((6,8),'Original RGB',fill='black')
        for index,name in enumerate(VIEWS[:4]):
            im=Image.open(OUT/'common_input/dataset/images'/f'{name}.jpg').convert('RGB');im.thumbnail((265,474))
            canvas.paste(im,(180+index*265,26));draw.text((184+index*265,5),name,fill='black')
        for index,folder in enumerate(folders):
            for offset,stage in enumerate(['raw','post']):
                y=(1+index*2+offset)*500;draw.text((6,y+8),f'{folder.parent.name}\n{folder.name}\n{stage}',fill='black')
                path=folder/'evaluation'/f'{stage}_four_views.png'
                if path.exists():
                    im=Image.open(path).convert('RGB');im.thumbnail((1060,500));canvas.paste(im,(180,y))
        b=io.BytesIO();canvas.save(b,format='PNG');Meter().save(OUT/'summary/mesh_overview.png',b.getbuffer())
    body=''
    for name in ['cleaning_decision.md','gaussian_selection.md','final_selection.md']:
        path=OUT/'summary'/name
        if path.exists():body+='<h2>'+name+'</h2><pre>'+html.escape(path.read_text())+'</pre>'
    if folders:
        body+='<h2>同相机 Raw / Post Mesh</h2><img src="mesh_overview.png"><p><a href="mesh_comparison.csv">Mesh 实测表</a></p><ul>'
        for folder in folders:
            rel=os.path.relpath(folder,OUT/'summary')
            body+='<li>'+folder.parent.name+' / '+folder.name+': '
            for stage in ['raw','post']:
                body+='<a href="'+rel+'/'+stage+'.ply">'+stage+' PLY</a> '
                if (folder/f'{stage}_3d.html').exists():body+='<a href="'+rel+'/'+stage+'_3d.html">'+stage+' 实际三角面三维预览</a> '
                body+='<a href="'+rel+'/evaluation/'+stage+'_four_views.png">'+stage+' 四视角RGB</a> '
                for view in ['0000','0048','0096','0144']:
                    for channel in ['Normal','Silhouette']:
                        body+='<a href="'+rel+'/evaluation/'+stage+'_'+view+'_'+channel+'.png">'+stage+'/'+view+'/'+channel+'</a> '
            body+='</li>'
        body+='</ul><p>三维预览使用实际网格，较大网格仅为显示而简化；完整 Raw/Post PLY 保留用于测量。</p>'
    return body


def main():
    rows=[]
    for method in METHODS:
        folder=OUT/'gaussian'/method;completed=read(folder/'complete.json');resources=read(folder/'worker_resources.json')
        observed=read(folder/'observed_resources.json')
        evaluation=read(folder/'evaluation/metrics.json');metrics=evaluation.get('metrics',{})
        multiview=read(folder/'evaluation/multiview_consistency.json').get('mean',{})
        rows.append(dict(method=method,status='evaluated' if evaluation else ('trained' if completed else 'pending'),
            **metrics,**{'multiview_'+k:v for k,v in multiview.items()},gaussian_count=completed.get('gaussian_count',''),model_bytes=completed.get('model_bytes',''),
            native_rgb_fps=1/metrics['rgb_render_s'] if metrics.get('rgb_render_s') else '',
            rgb_alpha_depth_fps=1/metrics['render_s'] if metrics.get('render_s') else '',
            training_wall_s=resources.get('wall_s',''),peak_cuda_allocated_gib=resources.get('peak_allocated_gib',''),
            peak_cuda_reserved_gib=resources.get('peak_reserved_gib',''),peak_rss_gib=resources.get('peak_rss_gib',''),
            observed_gpu1_peak_mib=observed.get('peak_gpu_device_mib',''),training_cpu_s=resources.get('cpu_s',''),
            io_read_bytes=resources.get('io',{}).get('application_read_bytes',''),
            io_write_bytes=resources.get('io',{}).get('application_write_bytes',''),
            io_peak_mib_s=resources.get('io',{}).get('peak_one_second_mib_s','')))
    csvwrite(OUT/'summary/gaussian_comparison.csv',rows)
    common_depth_display()
    for channel in ['RGB','Alpha','Depth','Normal_derived']:overview(channel)
    progress=read(OUT/'summary/progress.json')
    columns=['method','status','foreground_psnr','psnr','ssim','lpips','alpha_iou','background_alpha','gaussian_count','training_wall_s','peak_cuda_allocated_gib','peak_cuda_reserved_gib','observed_gpu1_peak_mib']
    def fmt(value):return f'{value:.4f}' if isinstance(value,float) else str(value)
    table='<table><thead><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr></thead><tbody>'
    for r in rows:table+='<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k in columns)+'</tr>'
    table+='</tbody></table>'
    complete=all(r['status']=='evaluated' for r in rows)
    body='<!doctype html><html lang="zh"><meta charset="utf-8"><title>COLMAP Gaussian Mesh Selection</title>'
    body+='<style>body{font:16px system-ui;margin:36px;max-width:1600px;background:#f7f8fa;color:#1f2937}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:10px;border:1px solid #cbd5e1;text-align:left}img{max-width:100%}a{color:#0369a1}pre{white-space:pre-wrap}</style>'
    body+='<h1>孔雀竹芋：真实 COLMAP 重建实验</h1>'
    body+='<p>'+('五种 Gaussian 的正式训练和共同测试集评价均已完成。' if complete else '实验正在执行；空白指标尚未测得，不能据此做最终选型。')+'</p>'
    body+='<p>输入：210/210 张注册，177,918 个真实 SfM 点；183/27 训练/测试；530×948；Seed 0；SH3；每种 30K。</p>'
    body+='<p>前景掩膜来自 RAP-FSAM3 预测，不是人工几何真值。深度均导出为 camera-Z，但不同算法的期望/中位估计器仍不同。</p>'+table
    body+='<p>掩膜在部分视角包含花盆。全背景 RGB 训练也会重建真实背景，背景 Alpha 高不能单独证明黑云或植物几何差。</p>'
    body+='<p><a href="final_selection.md">最终选型与限制</a> · <a href="gaussian_comparison.csv">五算法完整实测表</a> · <a href="resource_measurement_notes.json">资源测量口径与缺测说明</a> · <a href="wrapper_sha256.json">执行脚本SHA256</a> · <a href="actual_commands.jsonl">实际执行命令</a></p>'
    body+='<p><a href="nksr_source_audit.json">NKSR上游版本与已复用CUDA兼容模块</a> · <a href="preferred_mesh.json">推荐Mesh机器可读路径</a> · <a href="../README.md">目录与重现入口</a></p>'
    if (OUT/'summary/DELIVERY_VALIDATION.json').exists():body+='<p><a href="DELIVERY_VALIDATION.json">交付产物与历史输入不变性验证</a></p>'
    body+='<p><a href="metric_definitions.md">评价定义和边界</a> · <a href="../common_input/colmap_source.json">真实COLMAP溯源</a> · <a href="../common_input/rap_masks_complete.json">RAP掩膜溯源</a> · <a href="source_environment_audit.json">源码与环境审计</a> · <a href="failed_attempt_costs.json">失败训练成本</a></p>'
    body+='<h2>同视角 RGB</h2><img src="gaussian_overview.png"><h2>Alpha</h2><img src="gaussian_Alpha.png"><h2>Depth</h2><img src="gaussian_Depth.png">'
    body+='<h2>统一深度推导法向</h2><img src="gaussian_Normal_derived.png">'
    if (OUT/'cleaning/GGGS/comparison_RGB.png').exists():body+='<h2>C1_raw清理损伤审阅</h2><img src="../cleaning/GGGS/comparison_RGB.png"><p><a href="../cleaning/GGGS/comparison.json">Original/Cleaned实测差值</a></p>'
    body+='<h2>实际产物</h2><ul>'
    for method in METHODS:
        record=read(OUT/'gaussian'/method/'complete.json')
        if record:
            relative=os.path.relpath(record['native_ply'],OUT/'summary')
            checkpoint=os.path.relpath(record['checkpoint'],OUT/'summary')
            viewer=OUT/'gaussian'/method/'centres_3d.html'
            body+='<li>'+method+': <a href="'+html.escape(relative)+'">原生 Gaussian PLY</a> · <a href="../gaussian/'+method+'/worker_resources.json">资源记录</a>'
            body+=' · <a href="'+html.escape(checkpoint)+'">原生最终checkpoint</a>'
            if viewer.exists():body+=' · <a href="../gaussian/'+method+'/centres_3d.html">实际 Gaussian 中心三维预览（点显示）</a>'
            if method in ['2DGS','GGGS']:
                for view in VIEWS:body+=' · <a href="../gaussian/'+method+'/evaluation/'+view+'_Normal_native.png">'+view+'原生法向</a>'
            body+='</li>'
    body+='</ul><h2>执行状态</h2><pre>'+html.escape(json.dumps(progress,ensure_ascii=False,indent=2))+'</pre>'
    body+=mesh_artifacts()
    body+='<p>来源：本地真实训练日志、原生模型及评价记录；<a href="https://github.com/graphdeco-inria/gaussian-splatting">3DGS</a>，<a href="https://github.com/hbb1/2d-gaussian-splatting">2DGS</a>，<a href="https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids">GGGS</a>，<a href="https://github.com/nerfstudio-project/gsplat">gsplat</a>，<a href="https://github.com/ubc-vision/3dgs-mcmc">MCMC-3DGS</a>，<a href="https://github.com/nv-tlabs/NKSR">NKSR</a>。博查搜索因额度不足返回 403；官方源码直接读取，API 文档经 Context7 核对。</p></html>'
    (OUT/'summary/index.html').write_text(body)
    write(OUT/'summary/report_status.json',dict(gaussian_evaluation_complete=complete,methods=rows))


if __name__=='__main__':main()
