#!/usr/bin/env python3
"""Standalone interactive viewer over recorded completed native-render artifacts."""
import csv,json,os,argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    args=parser.parse_args();root=args.comparison.resolve();folder=root/'comparisons'
    rows=list(csv.DictReader((folder/'ALL_GGGS_CLEANING_RESULTS.csv').open()))
    for row in rows:
        out=Path(row['experiment'])
        row['relative']=os.path.relpath(out,folder)
        for key in ['native_ply','supersplat_ply','removed_ply','report']:
            row[key]=os.path.relpath(row[key],folder)
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    html=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GGGS 清理真实结果查看</title><style>
body{font:15px/1.5 system-ui,sans-serif;margin:24px;background:#f5f5f4;color:#202020}h1{font-size:23px;margin-bottom:6px}p{max-width:1000px}select,button{font:inherit;padding:6px;margin:4px 12px 8px 0}label{display:inline-block}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.card{background:white;padding:12px;border:1px solid #ccc;border-radius:6px}.card img{width:100%;height:540px;object-fit:contain;background:#eee}.card h2{font-size:16px;margin:0}.card p{font-size:13px;margin:5px 0}.links{display:flex;gap:12px;flex-wrap:wrap}table{border-collapse:collapse;background:white;margin-top:20px;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left;font-size:13px}caption{text-align:left;margin:10px 0;font-weight:600}.scroll{overflow:auto}.note{background:#fff4d6;padding:10px}a{color:#165fa1}
</style><h1>GGGS 清理真实结果查看</h1>
<p>所有主样本方法使用同一份 E10 原始 GGGS。此页面只列入具有真实完成记录的模型；删除数量、指标与图像来自本地原生渲染。开发版本按实际目录显示，尚未完成的运行不会作为成功结果列出。</p>
<p class="note">C1/C2/C5-A 的外观规则结果有前景损失；C4 与部分 C5-B 为零修改。自由视角没有对应真实照片，背景支持来自现有深度代理。白底/黑底切换不能证明不同拍摄背景的泛化。查看版 PLY 已烘焙 Filter，原生 PLY 保留 Filter；浏览器导入验证与原生渲染验证的范围见 HOW_TO_VIEW.md。</p>
<label>植物 <select id="plant"></select></label><label>视角 <select id="view"><option>top</option><option>oblique</option><option>far</option><option>near</option><option>orbit_00</option><option>orbit_01</option><option>orbit_02</option><option>orbit_03</option></select></label>
<label>背景 <select id="background"><option value="white">白底</option><option value="black">黑底</option><option value="alpha">Alpha</option></select></label>
<label>版本 <select id="scope"><option value="latest">各方法最新完成版本</option><option value="all">全部完成版本</option></select></label>
<p><a href="../artifact_index/ALL_GGGS_PLY_PATHS.md">所有实际 PLY 路径</a> · <a href="ALL_GGGS_CLEANING_RESULTS.csv">完整指标 CSV</a> · <a href="DEVELOPMENT_MECHANISM_AUDIT.md">清理机制与拒绝审计</a></p><div id="cards"></div><div class="scroll"><table id="metrics"></table></div>
<script>const data=PAYLOAD;const $=id=>document.getElementById(id);for(const p of [...new Set(data.map(r=>r.plant))].sort()){let o=document.createElement('option');o.textContent=p;$('plant').append(o)}$('plant').value=data.some(r=>r.plant==='KongQueZhuYu')?'KongQueZhuYu':data[0]?.plant;
function text(tag,value,parent){const e=document.createElement(tag);e.textContent=value;parent.append(e);return e}function number(x,d=3){return x===''?'未测':Number(x).toFixed(d)}function rank(r){return [...r.experiment_version.matchAll(/\d+(?:\.\d+)?/g)].map(x=>Number(x[0])).reduce((a,x)=>a*100+x,0)+(r.experiment.includes('/final/')?1e12:0)}
function draw(){let rows=data.filter(r=>r.plant===$('plant').value);if($('scope').value==='latest'){let m=new Map;for(const r of rows){if(!m.has(r.method)||rank(r)>rank(m.get(r.method)))m.set(r.method,r)}rows=[...m.values()]}rows.sort((a,b)=>a.method.localeCompare(b.method));$('cards').replaceChildren();$('metrics').replaceChildren();let caption=text('caption','同一原生评价流程的实际记录',$('metrics'));let head=text('tr','',$('metrics'));for(const label of ['方法 / 版本','删除','FG-PSNR','前景 Alpha','薄结构','背景 Alpha / FG','耗时（秒）','结果状态'])text('th',label,head);
for(const r of rows){let card=text('div','',$('cards'));card.className='card';text('h2',r.method+' / '+r.experiment_version,card);let image=text('img','',card);image.src=encodeURI(r.relative+'/visualizations/free/'+$('view').value+'_'+$('background').value+'.png');image.alt='该完成结果的原生 GGGS '+$('view').value+' 视角';text('p','Gaussian '+r.original_gaussians+' → '+r.cleaned_gaussians+'；删除 '+r.removed+'；'+r.outcome,card);let links=text('div','',card);links.className='links';for(const [key,name]of [['supersplat_ply','SuperSplat PLY'],['native_ply','原生 PLY'],['removed_ply','被删点'],['report','报告']]){let a=text('a',name,links);a.href=encodeURI(r[key])}let tr=text('tr','',$('metrics'));for(const v of [r.method+' / '+r.experiment_version,r.removed,number(r.fg_psnr),number(r.fg_alpha,6),number(r.thin_coverage,6),number(r.bg_alpha_mass_per_fg,6),number(r.seconds,1),r.outcome])text('td',v,tr)}}for(const id of ['plant','view','background','scope'])$(id).onchange=draw;draw();</script></html>'''
    (folder/'REVIEW.html').write_text(html.replace('PAYLOAD',payload))
    print((folder/'REVIEW.html').resolve())

if __name__=='__main__':main()
