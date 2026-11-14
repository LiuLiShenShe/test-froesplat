"""Independent throttled readback: recompute upstream IDs and check every native byte.

Run after cleanup. Verification CPU/I/O is excluded from cleanup resource reports.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .run import filter_ids
from fast_ply.io import Meter, read_header, chunks


def read(path, meter):
    with path.open('rb', buffering=0) as stream:
        header = read_header(stream, meter)
        rows = np.empty(header.count, header.dtype)
        for start, block in chunks(stream, header, meter, 16384):
            rows[start:start+len(block)] = block
    return header, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plant-root', type=Path, required=True)
    args = p.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    meter, records, cached, original = Meter(20), {}, None, None
    started = time.perf_counter()
    for report_path in sorted(args.plant_root.glob('*/cleanup_report.json')):
        report = json.loads(report_path.read_text())
        source = Path(report['input_ply'])
        if source != cached:
            header, original = read(source, meter)
            cached = source
        ids, _ = filter_ids(original, report['method'], report['strength'] or 'medium')
        assert hashlib.sha256(ids.astype('<i8',copy=False).tobytes()).hexdigest() == report['retained_ids_sha256']
        with Path(report['output_ply']).open('rb', buffering=0) as stream:
            out = read_header(stream, meter)
            assert out.lines == header.output(len(ids)).splitlines(keepends=True)
            assert out.dtype == header.dtype and out.count == len(ids)
            for start, block in chunks(stream, out, meter, 16384):
                assert block.tobytes() == original[ids[start:start+len(block)]].tobytes()
            assert stream.read(1) == b''
        records[report_path.parent.name] = dict(all_records_byte_identical=True, original_id_digest_matches=True,
            gaussians=len(ids), fields=list(header.dtype.names), record_bytes=header.dtype.itemsize)
    if not records:
        raise ValueError('No cleanup reports found')
    result = dict(results=records, verification_wall_s=time.perf_counter()-started,
        logical_read_bytes=meter.read_bytes, measured_separately_from_cleaning=True)
    (args.plant_root/'byte_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
