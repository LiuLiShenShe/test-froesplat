"""Summarize existing real Mesh ray queries under frozen original-RGB groups."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from queue_io import write_status

OUT = Path(__file__).resolve().parents[1]
DEST = OUT / 'comparison/reflection_mesh_correspondence'
RGB = OUT / 'comparison/reflection_analysis/ORIGINAL_RGB_QUERY_VALIDATION.csv'

def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def yes(value):
    return str(value).lower() == 'true'

def main():
    queries = read(RGB)
    query_ids = {int(r['query_id']) for r in queries}
    assert len(query_ids) == len(queries) == 24375
    groups = {
        'raw_context_candidates': {int(r['query_id']) for r in queries if yes(r['raw_context_candidate'])},
        'raw_context_controls': {int(r['query_id']) for r in queries if not yes(r['raw_context_candidate'])},
        'legacy_masked_candidates': {int(r['query_id']) for r in queries if yes(r['legacy_masked_candidate'])},
        'masked_only': {int(r['query_id']) for r in queries if r['candidate_relation'] == 'masked_only'},
    }
    identity = {int(r['query_id']): (r['source_id'], r['target_id']) for r in queries}
    results, sources = [], []
    for path in sorted(DEST.glob('*_spatial_correspondence.csv')):
        before = path.stat()
        rows = read(path)
        index = {int(r['query_id']): r for r in rows}
        assert len(rows) == len(index) and set(index) == query_ids, path
        assert all((r['source_id'], r['target_id']) == identity[q] for q, r in index.items()), path
        actual_mesh = Path(rows[0]['mesh_path']).resolve(strict=True)
        assert all(r['mesh_path'] == rows[0]['mesh_path'] for r in rows), path
        for group, ids in groups.items():
            selected = [index[q] for q in sorted(ids)]
            hits = [r for r in selected if yes(r['mesh_hit'])]
            native_valid = [r for r in selected if yes(r['native_depth_valid'])]
            paired = [r for r in hits if yes(r['native_depth_valid'])]
            result = dict(mesh_method=rows[0]['mesh_method'], RGB_candidate_group=group,
                queries=len(selected), mesh_hit_queries=len(hits), mesh_missing_queries=len(selected)-len(hits),
                mesh_hit_fraction=len(hits)/len(selected) if selected else '',
                native_depth_valid_queries=len(native_valid), paired_mesh_native_depth_queries=len(paired),
                actual_mesh_path=str(actual_mesh), source_query_geometry=str(path.resolve()),
                RGB_group_source=str(RGB.resolve()))
            for key, subset in [('mesh_vs_native_relative_depth', paired),
                                ('mesh_local_normal_jump3px_deg', hits),
                                ('mesh_neighbor_hit_fraction3px', hits)]:
                values = [float(r[key]) for r in subset if r.get(key) not in ['', None] and np.isfinite(float(r[key]))]
                result[key+'_finite_queries'] = len(values)
                result[key+'_median'] = float(np.median(values)) if values else ''
                result[key+'_p90'] = float(np.quantile(values, .9)) if values else ''
            results.append(result)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Query CSV changed during read'
        sources.append(dict(path=str(path.resolve()), sha256=digest, mesh_path=str(actual_mesh), queries=len(rows)))
    assert results, 'No actual Mesh correspondence results available'
    target = OUT / 'comparison/reflection_original_RGB_mesh_metrics.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    temporary.replace(target)
    write_status(DEST / 'ORIGINAL_RGB_SUBSET_AUDIT.json', dict(status='complete_actual_available',
        actual_mesh_routes=len(sources), fixed_queries=len(queries), candidate_group_counts={k:len(v) for k,v in groups.items()},
        RGB_validation_sha256=hashlib.sha256(RGB.read_bytes()).hexdigest(), source_geometry_files=sources,
        output_metrics=str(target.resolve()), Mesh_and_legacy_query_files_unchanged=True, CPU_only=True,
        normal_definition='Exact source-ray triangle world normal and unit normals of neighboring rays in3px. No inherited native bilinear normal metric is used.',
        native_valid_definition='Inherited native_depth_valid flag requires the original source-and-target native depth/Alpha validity; it is not merely a positive source-depth check.',
        depth_definition='Mesh cameraZ versus its source native Gaussian rendered cameraZ. This is method disagreement, not scan truth or an independent cross-view depth error.',
        missing_definition='No triangle intersection at the frozen matched source pixel; sample missing fraction is distinct from208-view enclosed-hole projection area.',
        selection='Same original02-FFT5px brightness threshold at fixed pre-existing matches. Brightness candidates are not certified specular highlights; source matches and masks are unchanged.'))
    print('OriginalRGB subsets of actual Mesh world-triangle correspondence', len(sources), 'routes', len(results), 'rows', flush=True)

if __name__ == '__main__':
    main()
