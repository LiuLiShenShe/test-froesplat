"""Build same-camera image pairs and a compact report from completed real runs."""
from pathlib import Path
import json
import statistics
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PLANTS = {'ChangShouHua1':'长寿花1', 'DouBanLv1':'豆瓣绿1', 'WangWenCao1':'网纹草1'}


def main():
    folder = ROOT/'reports/C2_fast_validation'
    review = json.loads((folder/'visual_review.json').read_text())
    summary = {}
    for plant, label in PLANTS.items():
        cleanup = json.loads((ROOT/'experiments/C2_fast_validation'/plant/'cleanup_report.json').read_text())
        path = folder/plant
        quality = json.loads((path/'paired_quality.json').read_text())
        assert quality['historical_protocol'] is None
        original = quality['results']['original']; cleaned = quality['results']['C2_raw']
        assert cleaned['gaussians'] == cleanup['remaining_gaussians']
        assert len(original['observed_views']) == len(cleaned['observed_views']) == 16
        old_black = sum(v['dark_bg_high_alpha_pixels'] for v in original['observed_views'])
        new_black = sum(v['dark_bg_high_alpha_pixels'] for v in cleaned['observed_views'])
        tiles = []
        for order, (old, new) in enumerate(zip(original['observed_views'], cleaned['observed_views'])):
            assert old['name'] == new['name'] and old['index'] == new['index']
            with Image.open(path/f'original_observed_{order:02}.png') as a, Image.open(path/f'C2_raw_observed_{order:02}.png') as b:
                assert a.size == b.size
                pair = Image.new('RGB', (a.width+b.width, max(a.height,b.height)+32), 'white')
                pair.paste(a,(0,32)); pair.paste(b,(a.width,32)); draw = ImageDraw.Draw(pair)
                draw.text((10,8), 'Original GGGS | '+old['name'], fill='black')
                draw.text((a.width+10,8), 'C2_raw | '+new['name'], fill='black')
                pair.save(path/f'paired_view_{order:02}.png')
                pair.thumbnail((480,470)); tiles.append(pair.copy())
        canvas = Image.new('RGB', (480*4,470*4), 'white')
        for i,tile in enumerate(tiles): canvas.paste(tile,((i%4)*480,(i//4)*470))
        canvas.save(path/'all_16_pairs.png')
        summary[plant] = dict(label=label, cleanup=cleanup,
            psnr_original=original['mean']['fg_psnr'], psnr_cleaned=cleaned['mean']['fg_psnr'],
            psnr_delta_db=cleaned['mean']['fg_psnr']-original['mean']['fg_psnr'],
            observed_bg_black_pixels_original=old_black, observed_bg_black_pixels_cleaned=new_black,
            observed_bg_black_reduction_pct=100*(1-new_black/old_black) if old_black else None,
            original_mean=original['mean'], cleaned_mean=cleaned['mean'],
            visual_review=review[plant], observed_indices=quality['observed_indices'])
    report = dict(schema='C2_fast_validation_summary_v1', plants=summary,
        average_cleanup_wall_s=statistics.mean(s['cleanup']['timing']['total_s'] for s in summary.values()),
        average_cleanup_cpu_s=statistics.mean(s['cleanup']['timing']['process_cpu_s'] for s in summary.values()),
        recommendation=review['recommendation'],
        quality_protocol='16 matched observed views per plant, max side 960, original native GGGS renderer, GPU1; no historical free-view protocol',
        visual_review_method='Direct model vision of actual local render PNGs; no external vision API')
    (folder/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines = ['# C2_raw 三株实测（2026-11-14）','',
        '孔雀竹芋新入口保留 428,967 / 576,396 个 Gaussian，逐 ID 与历史 C2_raw 完全一致。六项回归检查通过；未改动历史 rules() 或清理阈值。','',
        '| 植物 | 原始点数 → 保留点数 | 删除比例 | 观测背景黑像素减少 | 前景 PSNR（dB） | 实际图像检查 | CPU时间 | 总时间 | 应用层峰值 I/O |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for s in summary.values():
        c=s['cleanup'];t=c['timing'];io=c['io']
        lines.append(f"| {s['label']} | {c['original_gaussians']:,} → {c['remaining_gaussians']:,} | {c['removed_fraction']*100:.2f}% | {s['observed_bg_black_reduction_pct']:.2f}% | {s['psnr_original']:.2f} → {s['psnr_cleaned']:.2f}（{s['psnr_delta_db']:+.2f}） | {s['visual_review']['assessment']} | {t['process_cpu_s']:.2f}s | {t['total_s']:.2f}s | {io['observed_peak_one_second_mib_s']:.2f} MiB/s |")
    lines += ['', '## 覆盖和资源', '',
        '| 植物 | 边缘覆盖：原始 → C2 | 薄结构覆盖：原始 → C2 | 新增前景透明孔洞 | 峰值 RSS | 逻辑读取 | 逻辑写入 | 完整读/写次数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary.values():
        c=s['cleanup'];o=s['original_mean'];n=s['cleaned_mean'];io=c['io']
        lines.append(f"| {s['label']} | {o['boundary_coverage']*100:.2f}% → {n['boundary_coverage']*100:.2f}% | {o['thin_coverage']*100:.2f}% → {n['thin_coverage']*100:.2f}% | {n['new_fg_alpha_hole_fraction']*100:.2f}% | {c['resource']['peak_process_rss_bytes']/2**30:.3f} GiB | {io['logical_read_bytes']/2**20:.2f} MiB | {io['logical_write_bytes']/2**20:.2f} MiB | {io['complete_model_reads']} / {io['complete_model_writes']} |")
    lines += ['', f"三株平均清理墙钟时间 {report['average_cleanup_wall_s']:.2f} 秒，平均 CPU 时间 {report['average_cleanup_cpu_s']:.2f} 秒。GPU 渲染评价单独执行，其时间与 I/O 不计入清理性能。", '',
        '每株采用相同的16个既有观测相机（长边960），复用图像与mask。背景黑像素为原评价器的 dark_bg_high_alpha_pixels，汇总16视角；不是孔雀竹芋固定8自由视角协议，百分比不与其直接对标。边缘、薄结构和骨架指标依赖既有前景mask，是结构代理而非逐叶语义真值。', '',
        '20 MiB/s 为应用层 PLY 读写合计限速。表中峰值为实测尾随一秒窗口逻辑吞吐；未测硬盘设备物理峰值，内核预读/回写及其他进程不受此限速保证。每株保留全部原生字段和保留记录字节，无中间模型、removed/viewer PLY、逐点NPZ/CSV或完整PLY哈希扫描。', '', '## 直接图像检查', '']
    for plant,s in summary.items():
        lines += [f"### {s['label']}", '', s['visual_review']['details'], '',
            f"![同相机原始与C2]({plant}/paired_view_{s['visual_review']['representative_view']:02}.png)", '',
            f"[全部16视角并排图]({plant}/all_16_pairs.png) · [原始渲染]({plant}/original_first_observed_full.png) · [清理后渲染]({plant}/C2_raw_first_observed_full.png) · [逐视角指标]({plant}/paired_quality.json)", '',
            f"[清理后原生PLY](../../experiments/C2_fast_validation/{plant}/cleaned_gaussians.ply) · [清理资源报告](../../experiments/C2_fast_validation/{plant}/cleanup_report.json)", '']
        if plant == 'WangWenCao1':
            lines += ['[crop_0013：目标RGB、原始GGGS、C2三联图](WangWenCao1/target_original_c2_view02.png)。', '']
    lines += ['## 结论', '',review['recommendation'], '',
        '清理入口：[fast_clean_c2.py](../../fast_clean_c2.py)；复现说明：[FAST_C2_README.md](../../FAST_C2_README.md)。原始/最终点数、时间、RSS和I/O来自每株 cleanup_report.json；渲染质量来自每株 paired_quality.json。实际输入和源码哈希记录在 [protocol.json](protocol.json)。', '',
        '网纹草首次GPU评价收到SIGTERM（退出143），中断日志和原始视角结果保留在WangWenCao1_interrupted。随后单独重跑完整16视角原始/C2配对评价，已完成；未重新清理或覆盖模型。中断和重跑均不计入清理资源统计。', '',
        '联网检索已尝试博查，返回 HTTP 403；实验结论以本地源码、实测 JSON 和直接查看的渲染图为来源。SciPy 文档通过 Context7 拉取，来源：https://github.com/scipy/scipy/blob/v1.18.1/doc/source/tutorial/spatial.rst 。']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({p:{k:s[k] for k in ['psnr_delta_db','observed_bg_black_reduction_pct']} for p,s in summary.items()},indent=2))


if __name__ == '__main__': main()
