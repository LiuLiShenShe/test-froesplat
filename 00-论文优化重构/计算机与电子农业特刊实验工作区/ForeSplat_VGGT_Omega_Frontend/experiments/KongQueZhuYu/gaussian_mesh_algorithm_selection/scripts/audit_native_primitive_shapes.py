"""Separate needle aspect from thin sheets at existing fixed RGB queries.

Only actual native model fields and the previously measured nearest-center
indices are read. No parameter editing, rematching or rendering is performed.
"""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from plyfile import PlyData

OUT=Path(__file__).resolve().parents[1]
SOURCE=OUT/'comparison/reflection_analysis'
DEST=OUT/'comparison/native_primitive_shapes'

def read(path):return list(csv.DictReader(path.open()))
def truth(value):return str(value).lower()=='true'
def writecsv(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()
def stat(values,prefix):
    values=np.asarray(values);values=values[np.isfinite(values)]
    result={prefix+'_finite_queries':len(values)}
    for name,quantile in [('median',.5),('p90',.9),('p99',.99)]:
        result[prefix+'_'+name]=float(np.quantile(values,quantile)) if len(values) else ''
    return result

def main():
    DEST.mkdir(exist_ok=True)
    queries=read(SOURCE/'RGB_QUERY_MANIFEST.csv');original=read(SOURCE/'ORIGINAL_RGB_QUERY_VALIDATION.csv')
    assert [int(r['query_id']) for r in original]==list(range(len(queries)))
    bright=np.array([truth(r['raw_context_candidate']) for r in original]);assert bright.sum()==134
    scale=json.loads((OUT.parent/'geometry_mesh_final_selection/configs/standard_spatial_scale.json').read_text())
    manifests=[];summaries=[];shape_rows={}
    for spatial in sorted(SOURCE.glob('*_spatial_correspondence.csv')):
        method=spatial.name.removesuffix('_spatial_correspondence.csv')
        meta=json.loads((SOURCE/(method+'_summary.json')).read_text())
        if not meta.get('primitive_source'):continue
        model=Path(meta['primitive_source']).resolve(strict=True)
        source=OUT/'comparison/reflection_normal_v2'/spatial.name
        if not source.is_file():source=spatial
        rows=read(source);assert [int(r['query_id']) for r in rows]==list(range(len(queries)))
        native=PlyData.read(model,mmap='r')['vertex'].data
        assert all(native.dtype.fields[name][0]==np.dtype('<f4') for name in native.dtype.names),model
        finite=np.isfinite(native.view('<f4').reshape(len(native),-1)).all(1)
        fields=sorted([name for name in native.dtype.names if name.startswith('scale_')],key=lambda name:int(name.split('_')[-1]))
        raw=np.column_stack([native[name] for name in fields])
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            decoded=np.exp(raw[:,:3])*np.tanh(raw[:,3:]) if len(fields)==6 else np.exp(raw)
            if len(fields)==3:
                ordered=np.sort(decoded,axis=1)
                span=ordered[:,-1];aspect=ordered[:,-1]/ordered[:,-2]
                flattening=ordered[:,-2]/ordered[:,0]
                legacy_aspect=ordered[:,-1]/ordered[:,0]
                definition='3D largest/second-largest sigma (needle-like axis); second/smallest separately measures flattening. Neither is certified artifact.'
            else:
                tangent=np.abs(decoded[:,:2]);span=tangent.max(1);aspect=span/tangent.min(1)
                flattening=np.full(len(native),np.nan);legacy_aspect=aspect.copy()
                definition='QGS absolute signed tangent span aspect; third signed scale is native curvature, not thickness.' if len(fields)==6 else '2DGS two tangent sigma aspect; there is no third physical thickness scale.'
        usable=finite&np.isfinite(decoded).all(1)&np.isfinite(aspect)&(span>0)
        threshold=float(np.quantile(aspect[usable],.99));span_threshold=float(np.quantile(span[usable],.99))
        per=[]
        for i,row in enumerate(rows):
            associated=truth(row.get('nearest_primitive_association_within5vox')) and truth(row['valid'])
            native_id=int(row['nearest_primitive_id'])
            assert 0<=native_id<len(native),(method,native_id)
            valid=associated and bool(usable[native_id])
            per.append(dict(query_id=i,source_id=row['source_id'],target_id=row['target_id'],method=method,
                original_RGB_brightness=bool(bright[i]),native_geometry_valid=truth(row['valid']),
                nearest_native_primitive_id=native_id,nearest_center_distance=row.get('nearest_primitive_distance',''),
                nearest_center_within5vox=associated,associated_native_shape_finite=valid,
                source_world_x=row['source_world_x'],source_world_y=row['source_world_y'],source_world_z=row['source_world_z'],
                tangent_or_needle_aspect=float(aspect[native_id]) if valid else '',
                threeD_flattening_second_over_smallest=float(flattening[native_id]) if valid and len(fields)==3 else '',
                legacy_max_over_min_aspect=float(legacy_aspect[native_id]) if valid else '',
                max_characteristic_span_over_D=float(span[native_id]/scale['diagonal']) if valid else '',
                aspect_at_or_above_own_model_P99=bool(aspect[native_id]>=threshold) if valid else '',
                span_at_or_above_own_model_P99=bool(span[native_id]>=span_threshold) if valid else '',
                local_normal_jump3px_deg=row.get('local_normal_jump3px_deg',''),
                matched_relative_depth_error=row.get('matched_relative_depth_error',''),
                shape_definition=definition))
        result=DEST/(method+'_same_query_shapes.csv');writecsv(result,per)
        shape_rows[method]=per
        for group,choose in [('original_RGB_brightness',bright),('original_RGB_controls',~bright),('all_fixed_queries',np.ones(len(queries),bool))]:
            selected=[r for i,r in enumerate(per) if choose[i]];valid=[r for r in selected if r['associated_native_shape_finite']]
            summary=dict(method=method,group=group,queries=len(selected),associated_finite_shape_queries=len(valid),
                associated_finite_fraction_all_queries=len(valid)/len(selected),
                unique_associated_native_primitives=len({r['nearest_native_primitive_id'] for r in valid}),
                native_scale_fields=len(fields),native_primitives=len(native),nonfinite_raw_native_primitives=int((~finite).sum()),
                own_model_aspect_P99=threshold,own_model_max_span_over_D_P99=span_threshold/scale['diagonal'],
                aspect_above_own_P99_fraction_associated=float(np.mean([r['aspect_at_or_above_own_model_P99'] for r in valid])) if valid else '',
                span_above_own_P99_fraction_associated=float(np.mean([r['span_at_or_above_own_model_P99'] for r in valid])) if valid else '',
                native_model_path=str(model),same_query_shape_path=str(result.resolve()),shape_definition=definition)
            for key in ['tangent_or_needle_aspect','threeD_flattening_second_over_smallest','legacy_max_over_min_aspect','max_characteristic_span_over_D']:
                summary.update(stat([float(r[key]) for r in valid if r[key]!=''],key))
            summaries.append(summary)
        manifests.append(dict(method=method,native_model_path=str(model),native_model_file_size=model.stat().st_size,
            native_model_SHA256=digest(model),spatial_source=str(source.resolve()),spatial_source_SHA256=digest(source),
            raw_scale_fields=len(fields),native_primitives=len(native),raw_nonfinite_primitives=int((~finite).sum()),
            decoded_shape_usable_primitives=int(usable.sum()),raw_finite_but_decoded_shape_invalid_primitives=int((finite&~usable).sum()),
            own_model_aspect_P99=threshold,own_model_max_span_P99=span_threshold,shape_definition=definition))
        print('Actual native shape audit',method,len(native),'associated queries',sum(r['associated_native_shape_finite'] for r in per),flush=True)
    writecsv(DEST/'ALL_NATIVE_SHAPE_SUMMARY.csv',summaries)
    joined=[];mesh_sources=[]
    for path in sorted((OUT/'comparison/reflection_mesh_correspondence').glob('*_spatial_correspondence.csv')):
        mesh_rows=read(path);method=mesh_rows[0]['method']
        if method not in shape_rows:continue
        assert [int(r['query_id']) for r in mesh_rows]==list(range(len(queries))),path
        shapes=shape_rows[method];mesh_path=Path(mesh_rows[0]['mesh_path']).resolve(strict=True)
        selected_pairs=[]
        for i,(shape,mesh) in enumerate(zip(shapes,mesh_rows)):
            assert (shape['source_id'],shape['target_id'])==(mesh['source_id'],mesh['target_id'])
            if bright[i]:
                selected_pairs.append(dict(shape,mesh_method=mesh['mesh_method'],mesh_path=str(mesh_path),
                    mesh_hit=truth(mesh['mesh_hit']),mesh_face_id=mesh['mesh_face_id'],mesh_depth=mesh['mesh_depth'],
                    mesh_world_x=mesh['mesh_world_x'],mesh_world_y=mesh['mesh_world_y'],mesh_world_z=mesh['mesh_world_z'],
                    mesh_local_normal_jump3px_deg=mesh['mesh_local_normal_jump3px_deg'],
                    mesh_vs_native_relative_depth=mesh['mesh_vs_native_relative_depth']))
        target=DEST/(mesh_rows[0]['mesh_method']+'_original_brightness_native_shapes_and_Mesh.csv')
        writecsv(target,selected_pairs)
        for group,choose in [('original_RGB_brightness',bright),('original_RGB_controls',~bright),('all_fixed_queries',np.ones(len(queries),bool))]:
            for category in ['all_queries','associated_shape_finite','aspect_at_or_above_own_model_P99','span_at_or_above_own_model_P99']:
                selected=[(shape,mesh) for i,(shape,mesh) in enumerate(zip(shapes,mesh_rows)) if choose[i] and
                    (category=='all_queries' or shape['associated_native_shape_finite'] and
                        (category=='associated_shape_finite' or shape[category]))]
                hits=[mesh for shape,mesh in selected if truth(mesh['mesh_hit'])]
                entry=dict(native_method=method,mesh_method=mesh_rows[0]['mesh_method'],group=group,shape_subset=category,
                    RGB_group_queries=int(choose.sum()),subset_queries=len(selected),mesh_hit_queries=len(hits),
                    mesh_missing_queries=len(selected)-len(hits),mesh_missing_fraction_subset=(len(selected)-len(hits))/len(selected) if selected else '',
                    native_model_path=next(r['native_model_path'] for r in manifests if r['method']==method),
                    mesh_path=str(mesh_path),raw_brightness_spatial_join=str(target.resolve()),
                    scope='Exact existing query IDs and source/target cameras; nearest center association is not Gaussian rendering contribution. Shape quantiles and brightness are descriptive, not causal artifact labels. Missing-query fractions differ from208-view enclosed-hole area.')
                entry.update(stat([float(r['mesh_local_normal_jump3px_deg']) for r in hits if r['mesh_local_normal_jump3px_deg']!=''],'mesh_local_normal_jump3px_deg'))
                joined.append(entry)
        mesh_sources.append(dict(native_method=method,mesh_path=str(mesh_path),spatial_mesh_source=str(path.resolve()),spatial_mesh_source_SHA256=digest(path),original_brightness_join=str(target.resolve())))
    if joined:writecsv(DEST/'ALL_NATIVE_SHAPE_MESH_JOIN.csv',joined)
    (DEST/'PROTOCOL.json').write_text(json.dumps(dict(status='complete_actual_available_models',model_records=manifests,
        same_query_Mesh_sources=mesh_sources,
        query_manifest_SHA256=digest(SOURCE/'RGB_QUERY_MANIFEST.csv'),queries=len(queries),original_RGB_brightness_queries=int(bright.sum()),
        source_views=len({r['source_id'] for r in queries}),original_models_modified=False,
        association='Previously recorded nearest finite native center within5 shared voxels of native rendered source point; not rasterizer contribution or visibility attribution. Invalid shape associations remain counted and blank, without replacing them by a farther primitive.',
        thresholds='Own-model finite-primitive P99, descriptive quantile only, not a learned artifact classifier. Different native representations and geometric curvature remain distinct.',
        limitations='Query-weighted repeated centers, image-only matching and brightness are not independent leaf samples or certified specularity; thin valid leaves can have large flattening. No physical thickness or scan truth claim.'),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
