"""Validate provenance of saved direct browser-image inspection records."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validated_review(output):
    folder = Path(output)/'comparison/official_supersplat'
    try:
        direct = json.loads((folder/'DIRECT_IMAGE_REVIEW.json').read_text())
        capture = json.loads((folder/'REVIEW.json').read_text())
        build = capture['editor_build_sha256']
        conformance = capture.get('renderer_conformance') or {}
        if (not direct.get('direct_image_inspection')
                or direct.get('renderer_build_sha256') != build
                or not conformance.get('passed')
                or conformance.get('editor_build_sha256') != build):
            return None
        records = {m['label']: m for m in capture['models']}
        required = {'B0_original','B2_stage1','B3_stage2'}
        if (Path(output)/'comparison/models/B1_historical_best_supersplat.ply').exists():
            required.add('B1_historical_best')
        if not required.issubset(set(direct['reviewed_models'])):
            return None
        for label in required:
            record = records[label]
            if digest(record['ply']) != record['sha256']:
                return None
            images = record['captures']
            if len(images) < 16 or {i['background'] for i in images} != {'white','black'}:
                return None
            if any(i.get('position_error',1.) > 1e-3 or digest(i['path']) != i['sha256'] for i in images):
                return None
        return direct
    except (OSError, ValueError, KeyError, TypeError):
        return None
