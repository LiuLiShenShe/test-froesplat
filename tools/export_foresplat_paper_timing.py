"""Export timing evidence without mixing model preparation with the original SfM."""

import csv
import json
from pathlib import Path
import re
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'ForeSplat_Paper_Evidence_Package'
OUTPUT = EVIDENCE / '10_REPRODUCIBILITY' / 'Data_Provenance'
POTTED = ROOT / 'experiments/ForeSplat_MultiSpecies_Phenotyping'
TOMATO = ROOT / 'experiments/ForeSplat_TomatoPGT_Independent'
RERUN = ROOT / '04-COLMAP-rerun-original/rerun_report.json'
WINDOWS_RERUN = ROOT / '04-COLMAP/rerun_log.json'
WINDOWS_FIRST = ROOT / '04-COLMAP/colmap_batch_log.json'
COHORT = EVIDENCE / '07_PHENOTYPING_EVALUATION/Tomato_Metric_19/Tomato_Metric_19_Reconstruction.csv'

# Raw clips were identified by matching early video frames against the archived JPEGs.
CLIPS = {
    'ChangShouHua1': ('2026-02-11 141235.mov', '0005.jpg'),
    'ChangShouHua2': ('2026-02-11 141841.mov', '0000.jpg'),
    'ChangShouHua3': ('2026-02-11 143046.mov', '0005.jpg'),
    'WanNianQing1': ('2026-02-11 144255.mov', '0005.jpg'),
    'WanNianQing2': ('2026-02-11 145348.mov', '0000.jpg'),
    'HongZhang': ('2026-02-11 150143.mov', '0004.jpg'),
    'DouBanLv1': ('2026-02-11 152202.mov', '0005.jpg'),
    'DouBanLv2': ('2026-02-11 153211.mov', '0000.jpg'),
    'DouBanLv3': ('2026-02-11 154455.mov', '0001.jpg'),
    'XianKeLai1': ('2026-02-11 155357.mov', '0000.jpg'),
    'XianKeLai2': ('2026-02-11 160510.mov', '0000.jpg'),
    'XianKeLai3': ('2026-02-11 161217.mov', '0000.jpg'),
    'CaoMei1': ('2026-02-11 161938.mov', '0000.jpg'),
    'CaoMei2': ('2026-02-11 162634.mov', '0007.jpg'),
    'XiangPiShu1': ('2026-02-11 163521.mov', '0000.jpg'),
    'XiangPiShu2': ('2026-02-11 164323.mov', '0000.jpg'),
    'WangWenCao1': ('2026-02-12 153431.mov', '0000.jpg'),
    'WangWenCao2': ('2026-02-12 154024.mov', '0009.jpg'),
    'KongQueZhuYu': ('2026-02-12 154506.mov', '0000.jpg'),
}


def read_json(path):
    return json.loads(path.read_text())


def wall_time(path):
    record = read_json(path)
    assert record['status'] == 'complete' and record['returncode'] == 0, path
    return round(record['ended_unix'] - record['started_unix'], 3)


def source(path):
    return str(path.relative_to(ROOT))


def potted_rows():
    rerun = read_json(RERUN)
    win_rerun = read_json(WINDOWS_RERUN)
    first = read_json(WINDOWS_FIRST)
    plants = {p.parents[1].name for p in POTTED.glob('*/common_input/complete.json')}
    assert plants == set(CLIPS)
    for name in sorted(plants):
        folder = POTTED / name
        marker = folder / 'common_input/complete.json'
        model = read_json(marker)
        first_run = first[name]
        assert first_run['status'] == 'success'
        original = rerun.get(name, {}).get('best')
        raw_model = model.get('raw_model', model['source'])
        original_time = original_method = original_report = None
        if original is not None:
            assert f"/{name}/{original['matcher']}" in raw_model
            assert original['points3d'] == model['sparse_points']
            original_time = original['time_seconds']
            original_method = f"linux_rerun_{original['matcher']}"
            original_report = source(RERUN)
        else:
            log = win_rerun.get(name, {})
            if (log.get('phase1') == 'success' and log.get('points3d_after_filter') == model['sparse_points']
                    and log.get('input_images') == model['raw_images']):
                original_time = log['colmap_time']
                original_method = 'windows_original_rgb_rerun'
                original_report = source(WINDOWS_RERUN)
        prepare = folder / 'logs/colmap_process.json'
        file, frame = CLIPS[name]
        video = ROOT / '原始视频' / file
        tags = 'com.apple.quicktime.model,com.apple.quicktime.creationdate'
        entries = f'format=duration:format_tags={tags}:stream=width,height,r_frame_rate:stream_side_data=rotation'
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', entries,
            '-of', 'json', video,
        ]))
        meta, stream = probe['format'], probe['streams'][0]
        assert raw_model.startswith(str(ROOT / '04-COLMAP'))
        yield dict(plant_id=name, video_start_local=meta['tags']['com.apple.quicktime.creationdate'],
                   video_duration_s=round(float(meta['duration']), 3), video_path=source(video),
                   frame_match_path=f'01-FFmepg/{name}/{frame}', camera=meta['tags']['com.apple.quicktime.model'],
                   encoded_width=stream['width'], encoded_height=stream['height'], frame_rate=stream['r_frame_rate'],
                   display_rotation_deg=next(s['rotation'] for s in stream['side_data_list'] if 'rotation' in s),
                   batch_prepare_wall_s=wall_time(prepare), prep_mode='reuse_own_sparse_model',
                   final_source_sfm_wall_s=original_time,
                   final_source_timing_status='matched_archived_timing' if original_report else 'timing_not_tied_to_selected_model',
                   final_source_sfm_method=original_method, final_source_timer_log=original_report,
                   historical_first_pass_colmap_wall_s=first_run['time_seconds'],
                   historical_first_pass_input_images=first_run['images'], first_pass_timer_log=source(WINDOWS_FIRST),
                   sparse_input=raw_model, batch_timer_source=source(prepare))


def colmap_step_time(folder, step):
    text = (folder / f'{step}.log').read_text(errors='replace')
    matches = re.findall(r'Elapsed time:\s*([\d.]+)\s*\[minutes\]', text)
    assert matches, (folder, step)
    # COLMAP's final mapper timer is cumulative; earlier entries time intermediate work.
    return round(60 * float(matches[-1]), 3)


def tomato_rows():
    with COHORT.open(encoding='utf-8-sig', newline='') as stream:
        cohort = list(csv.DictReader(stream))
    assert len({r['sample_id'] for r in cohort}) == len(cohort) == 19
    for member in sorted(cohort, key=lambda r: r['sample_id']):
        name = member['sample_id']
        folder = TOMATO / name
        inputs = read_json(folder / 'colmap/independent_sfm.json')
        assert 'independently run masked COLMAP' in inputs['source']
        log = folder / 'logs/colmap_process.json'
        steps = folder / 'colmap/logs'
        t = [colmap_step_time(steps, s) for s in ('feature_extractor', 'exhaustive_matcher', 'mapper', 'image_undistorter')]
        duration = wall_time(log)
        assert 0 < sum(t) <= duration, name
        yield dict(scan_id=name, cultivar=member['cultivar'], input_images=inputs['input_images'],
                   registered_images=inputs['registered_images'], colmap_stage_wall_s=duration,
                   feature_extraction_s=t[0], feature_matching_s=t[1], sparse_mapper_s=t[2],
                   undistortion_s=t[3], colmap_command_timer_sum_s=round(sum(t), 3),
                   stage_timer_source=source(log), command_log_directory=source(steps))


def save(filename, rows, summary):
    rows = list(rows)
    path = OUTPUT / filename
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    for key in summary:
        valid = [row[key] for row in rows if row[key] is not None]
        print(key, 'n=', len(valid), 'min/median/max=', min(valid), statistics.median(valid), max(valid))


if __name__ == '__main__':
    save('Potted19_capture_colmap_timing.csv', potted_rows(),
         ['video_duration_s', 'batch_prepare_wall_s', 'final_source_sfm_wall_s', 'historical_first_pass_colmap_wall_s'])
    save('Tomato19_colmap_timing.csv', tomato_rows(), ['colmap_stage_wall_s', 'colmap_command_timer_sum_s'])
