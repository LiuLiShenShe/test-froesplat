#!/usr/bin/env python3
"""Build an offline viewer of actual matched SuperSplat captures and model links."""
import argparse
import json
import os
from pathlib import Path

from viewer_review_validity import validated_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())['plants']
    rows = []
    for item in manifest:
        out = Path(item['output'])
        review = validated_review(out)
        if not review:continue
        selection = out/'VISUAL_SELECTION.json'
        selected = json.loads(selection.read_text()) if selection.exists() else {}
        row = dict(plant=item['plant'], selected=review.get('recommended_model'),
            observations=review.get('observations', []), links={}, captures={}, marker_captures={}, c5c_captures={},
            visual_acceptance=review.get('visual_acceptance'), severity=review.get('remaining_artifact_severity'),
            classification=selected.get('selection_classification','see direct review'))
        for label,record in selected.get('models', {}).items():
            row['links'][label] = os.path.relpath(record['path'], root)
        row['links']['report'] = os.path.relpath(out/'CLEANING_REPORT.md', root)
        for image in sorted((out/'comparison/official_supersplat').glob('compare_*.png')):
            row['captures'][image.stem.removeprefix('compare_')] = os.path.relpath(image, root)
        for image in sorted((out/'comparison/official_supersplat').glob('historical_marker_compare_*.png')):
            row['marker_captures'][image.stem.removeprefix('historical_marker_compare_')] = os.path.relpath(image, root)
        for image in sorted((out/'comparison/official_supersplat').glob('C5C_compare_*.png')):
            row['c5c_captures'][image.stem.removeprefix('C5C_compare_')] = os.path.relpath(image, root)
        rows.append(row)
    payload = json.dumps(rows, ensure_ascii=False).replace('</', '<\\/')
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ForeSplat 两阶段清理实际对照</title><style>
body{font:16px/1.55 system-ui,sans-serif;margin:24px;background:#f4f4f3;color:#222}
h1{font-size:24px}select{font:inherit;padding:5px;margin:5px 18px 8px 0}
img{max-width:100%;height:auto;border:1px solid #bbb;background:white}
#links{display:flex;gap:20px;flex-wrap:wrap;margin:12px 0}a{color:#1565a5}
#notes{max-width:1000px}li{margin:5px 0}p{max-width:1100px}
</style><h1>ForeSplat 两阶段清理实际对照</h1>
<p>这里展示已完成直接审阅的真实 SuperSplat 截图。B0为原始，B2为Stage 1，B3为Stage 2；孔雀竹芋另有历史最佳B1，其B3以B1为输入。选定模型和仍存伪影见下方记录。</p>
<label>植物 <select id="plant"></select></label>
<label>对照 <select id="comparison"><option value="captures">两阶段清理</option><option value="marker_captures">历史Round5与标记物删除</option><option value="c5c_captures">旧C5-C、历史最佳与Stage2</option></select></label>
<label>视角 <select id="view"><option>top</option><option>orbit_00</option><option>orbit_01</option><option>orbit_02</option><option>orbit_03</option><option>oblique</option><option>far</option><option>near</option></select></label>
<label>背景 <select id="bg"><option value="white">白底</option><option value="black">黑底</option></select></label>
<div id="links"></div><img id="capture" alt="相同视角的实际编辑器对照"><p id="selection"></p><ul id="notes"></ul>
<p>截图使用有已知Alpha测试记录的独立兼容构建。具体版本、相机、模型与图像哈希见各株 comparison/official_supersplat/REVIEW.json；原始构建差异见 <a href="../viewer_compatibility/VISUAL_REVIEW_ERRATUM.md">编辑器说明</a>。全体完成状态及绝对路径见 <a href="FINAL_TWO_STAGE_CLEANER_REPORT.md">实测报告</a>。</p>
<script>const rows=PAYLOAD;const $=id=>document.getElementById(id);
function text(tag,value,parent){const e=document.createElement(tag);e.textContent=value;parent.append(e);return e}
for(const r of rows){let o=text('option',r.plant,$('plant'));o.value=r.plant}
function draw(){const r=rows.find(x=>x.plant===$('plant').value);if(!r){$('selection').textContent='尚无通过直接审阅的结果';$('capture').hidden=true;return}
for(const o of $('comparison').options)o.disabled=!Object.keys(r[o.value]).length;if($('comparison').selectedOptions[0].disabled)$('comparison').value='captures';
$('capture').hidden=false;$('capture').src=encodeURI(r[$('comparison').value][$('view').value+'_'+$('bg').value]);
$('links').replaceChildren();for(const [key,label]of [['native','所选原生PLY'],['supersplat','所选SuperSplat PLY'],['report','单株报告']]){if(r.links[key]){const a=text('a',label,$('links'));a.href=encodeURI(r.links[key])}}
$('selection').textContent=(r.visual_acceptance===false?'视觉验收失败；仅列出最佳可用结果：':'有清理收益，仍须查看残留记录；选择：')+r.selected+'；残留：'+r.severity;$('selection').style.color=r.visual_acceptance===false?'#a11919':'#222';$('notes').replaceChildren();for(const note of r.observations)text('li',note,$('notes'))}
for(const id of ['plant','comparison','view','bg'])$(id).onchange=draw;draw();</script></html>'''
    target = root/'REVIEW.html'
    target.write_text(html.replace('PAYLOAD', payload))
    print(f'{target}: {len(rows)} directly reviewed plants')


if __name__ == '__main__':
    main()
