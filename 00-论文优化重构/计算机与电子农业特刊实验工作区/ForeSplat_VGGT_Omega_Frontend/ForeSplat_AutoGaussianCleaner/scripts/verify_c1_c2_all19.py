"""Audit completed artifacts using headers and metadata, without rescanning PLYs."""
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fast_ply.io import Meter, read_header


def main():
    out = ROOT / 'experiments/C1_C2_all19'
    summary = out / 'summary'
    protocol = json.loads((summary / 'protocol.json').read_text())
    entries = json.loads((ROOT / 'configs/plant_manifest.json').read_text())['plants']
    meter = Meter(20)
    audited = []
    reused = 0
    for relative, expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    for entry in entries:
        plant = entry['plant']
        scene = json.loads(Path(entry['scene']).read_text())
        batch = json.loads((out / plant / 'batch_report.json').read_text())
        stamp = protocol['input_stamps'][plant]
        source = Path(scene['model']).resolve()
        stat = source.stat()
        assert str(source) == stamp['path'] == batch['input']['path']
        assert stat.st_size == stamp['size_bytes']
        assert stat.st_mtime_ns == stamp['mtime_ns']
        assert batch['io']['complete_source_model_reads'] == 1
        assert batch['execution_path'] == 'one_read_in_memory'
        assert batch['io']['observed_peak_one_second_mib_s'] <= 20
        assert batch['resource']['peak_process_rss_bytes'] < 8 * 2**30
        assert batch['resource']['gpu_used'] is False
        with source.open('rb', buffering=0) as stream:
            original = read_header(stream, meter)
        assert stat.st_size == original.offset + original.count * original.dtype.itemsize
        counts = {}
        for method in ['C1_raw', 'C2_raw']:
            folder = out / plant / method
            report = json.loads((folder / 'cleanup_report.json').read_text())
            model = folder / 'cleaned_gaussians.ply'
            with model.open('rb', buffering=0) as stream:
                header = read_header(stream, meter)
            assert header.dtype == original.dtype
            assert header.count == report['remaining_gaussians']
            assert model.stat().st_size == header.offset + header.count * header.dtype.itemsize
            assert report['original_gaussians'] == original.count
            assert not model.samefile(source)
            assert report['legacy_sha256'] == protocol['rule_sha256']
            counts[method] = header.count
            if report['reused']:
                assert model.samefile(report['reuse']['ply']['path'])
                assert report['io']['logical_write_bytes'] == 0
                reused += 1
        assert counts['C1_raw'] >= counts['C2_raw']
        comparison = out / plant / 'comparison'
        quality = json.loads((comparison / 'paired_quality.json').read_text())
        assert len(quality['observed_indices']) == 4
        assert quality['observed_max_side'] == 960
        assert quality['cleaning_stage'] == 'external_final_models'
        assert set(quality['results']) == {'original', 'C1_raw', 'C2_raw'}
        assert quality['measured_gpu'].startswith('physical GPU1')
        views = quality['results']['original']['observed_views']
        for method in ['C1_raw', 'C2_raw']:
            actual = quality['results'][method]['observed_views']
            assert [(v['index'], v['name']) for v in actual] == [(v['index'], v['name']) for v in views]
        for i in range(4):
            sizes = []
            for prefix in ['reference', 'original', 'C1_raw', 'C2_raw']:
                with Image.open(comparison / f'{prefix}_observed_{i:02}.png') as image:
                    sizes.append(image.size)
            assert len(set(sizes)) == 1
            assert max(sizes[0]) <= 960
        for filename in ['contact.png', 'reference_contact.png', 'metrics.json']:
            assert (comparison / filename).is_file()
        audited.append(dict(plant=plant, original_gaussians=original.count,
            remaining=counts, native_fields=len(original.dtype.names), cameras=[v['name'] for v in views]))
    assert len(audited) == 19
    assert len(list(out.glob('*/C?_raw/cleaned_gaussians.ply'))) == 38
    assert len(list(out.glob('*/comparison/reference_observed_*.png'))) == 76
    assert sum(len(list(out.glob(f'*/comparison/{m}_observed_*.png'))) for m in ['original', 'C1_raw', 'C2_raw']) == 228
    assert not list(out.rglob('*.partial'))
    assert not list(out.rglob('*.npz'))
    with (summary / 'all19_results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 38
    assert all('待' not in row['cloud_effect'] + row['damage'] + row['suitable'] for row in rows)
    assert reused == 5
    anchor = json.loads((out / 'KongQueZhuYu/batch_report.json').read_text())
    for method, count in [('C1_raw', 443681), ('C2_raw', 428967)]:
        v = anchor['historical_verification'][method]
        assert v['identical_ids'] and v['count'] == count
    completion = json.loads((summary / 'run_complete.json').read_text())
    assert completion['completed'] and not completion['errors']
    result = dict(passed=True, plants=19, models=38, render_images=228, reference_images=76,
        reused_models=5, newly_exported_models=33,
        checks=['frozen source hashes', 'unchanged original file stamps', 'native field schemas and file lengths',
            'one original payload read per plant', 'joint throughput <=20 MiB/s', 'RSS <8 GiB',
            'CPU-only cleanup', 'same four cameras and image dimensions', 'historical exact Gaussian IDs',
            'reuse inode provenance', 'complete visual reviews and CSV', 'no partial models or NPZ'],
        scope='Header/metadata audit; does not rescan full PLY payloads or redo rendering/KNN. Native record bytes covered by the 7 regression checks.',
        audit_ply_header_read_bytes=meter.read_bytes, details=audited)
    (summary / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['checks', 'details', 'scope']}))


if __name__ == '__main__':
    main()
