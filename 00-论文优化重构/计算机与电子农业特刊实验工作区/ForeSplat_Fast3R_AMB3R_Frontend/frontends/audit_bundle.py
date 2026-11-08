"""CPU audit of actual artifacts, provenance, coordinates and confidence filters.

Unmapped SfM input rows remain present with explicit statuses and NaN geometry.
Passing this audit establishes artifact consistency, never geometric accuracy.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import csv
import json
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frontends.common import json_write
FILTERS = ('raw', 'confidence', 'mask', 'mask_confidence', 'mask_confidence_q85')

def npz_headers(path, shape_fields=()):
    """Read small NPY headers and stored CRCs, without re-reading array payloads.

    Stored CRC equality is a content fingerprint comparison, not a freshly
    computed cryptographic checksum or full-payload integrity validation.
    """
    records = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith('.npy'):
                raise ValueError('unexpected NPZ member: ' + info.filename)
            key = info.filename[:-4]
            record = dict(stored_crc=info.CRC, uncompressed_bytes=info.file_size)
            if key in shape_fields:
                with archive.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1,0): shape,order,dtype=np.lib.format.read_array_header_1_0(member)
                    elif version == (2,0): shape,order,dtype=np.lib.format.read_array_header_2_0(member)
                    else: raise ValueError('unsupported NPY header version: ' + str(version))
                    offset = member.tell()
                if dtype.hasobject or info.file_size != offset + int(np.prod(shape))*dtype.itemsize:
                    raise ValueError('NPY array header/payload length mismatch: ' + info.filename)
                record.update(shape=shape, dtype=dtype.str)
            records[key] = record
    return records

def ply_count(path):
    """Check binary XYZ/RGB header and the entire payload length."""
    with Path(path).open('rb') as f:
        lines = []
        while True:
            line = f.readline()
            if not line or f.tell() > 65536:
                raise ValueError('missing/oversized PLY header')
            lines.append(line.decode('ascii').strip())
            if lines[-1] == 'end_header':
                offset = f.tell()
                break
    if lines[:2] != ['ply', 'format binary_little_endian 1.0']:
        raise ValueError('expected binary little-endian PLY')
    properties = ['property float x', 'property float y', 'property float z',
                  'property uchar red', 'property uchar green', 'property uchar blue']
    if [x for x in lines if x.startswith('property ')] != properties:
        raise ValueError('expected XYZ float32 and RGB uint8')
    elements = [x for x in lines if x.startswith('element ')]
    if len(elements) != 1 or not elements[0].startswith('element vertex '):
        raise ValueError('expected only vertex element')
    n = int(elements[0].split()[-1])
    if n < 0 or Path(path).stat().st_size != offset + 15 * n:
        raise ValueError('PLY payload length differs from header')
    return n

def audit(path):
    path = Path(path)
    issues, warnings, exemptions, counts = [], [], [], {}
    totals = {key: 0 for key in FILTERS}
    def required(relative):
        p = path / relative
        if not p.is_file():
            issues.append('missing required artifact: ' + relative)
            return None
        return p
    def read_json(relative):
        p = required(relative)
        if p is None: return None
        try:
            def reject(token):
                raise ValueError('nonportable JSON token ' + token + '; use null for unmapped values')
            return json.loads(p.read_text(), parse_constant=reject)
        except Exception as exc:
            issues.append(relative + ': ' + str(exc))
    def check_ply(relative, expected=None):
        p = required(relative)
        if p is None: return None
        try:
            n = ply_count(p)
            counts['ply_files_checked'] = counts.get('ply_files_checked', 0) + 1
            if expected is not None and n != int(expected):
                issues.append(f'{relative}: {n} vertices, expected {expected}')
            return n
        except Exception as exc:
            issues.append(relative + ': ' + str(exc))
    bundle = read_json('bundle.json')
    if not isinstance(bundle, dict):
        report = dict(schema='foresplat.bundle.audit.v2', path=str(path), issues=issues, passed=False)
        json_write(path / 'bundle_audit.json', report)
        return report
    meta = bundle.get('metadata', {})
    names = bundle.get('input_names', [])
    n = len(names); method = bundle.get('method', '')
    legacy = method.startswith('VGGT-Ω'); sfm = 'SfM' in method
    fast = method.startswith('Fast3R'); base = method.startswith('AMB3R_Base_')
    if bundle.get('schema') != 'foresplat.frontend.v1' or not n or len(set(names)) != n:
        issues.append('bundle schema/unique input names invalid')
    if bundle.get('frame_count') != n or meta.get('input_views', n) != n:
        issues.append('bundle/metadata frame count differs')
    if bundle.get('pose_convention') != 'camera_to_world' or not bundle.get('world_frame') or not bundle.get('scale'):
        issues.append('coordinate/scale convention missing')
    if meta.get('method') != method: issues.append('metadata method differs')
    if not re.fullmatch('[0-9a-f]{64}', str(meta.get('weight_sha256', ''))):
        issues.append('weight SHA256 missing/invalid')
    commit_keys = ('amb3r_git_commit', 'fast3r_git_commit') if sfm and fast else ('amb3r_git_commit',) if sfm else ('git_commit',)
    for key in commit_keys:
        if not re.fullmatch('[0-9a-f]{40}', str(meta.get(key, ''))):
            if (legacy and meta.get('git_commit') is None
                    and Path(meta.get('source', '')).is_dir()
                    and all(re.fullmatch('[0-9a-f]{64}', str(meta.get(k, '')))
                            for k in ('source_cameras_sha256', 'source_metadata_sha256'))):
                exemptions.append('Legacy VGGT commit cannot be recovered from the original archive; not invented.')
            else: issues.append('official commit missing/invalid: ' + key)
    params = read_json('run_parameters.json')
    if isinstance(params, dict) and params.get('method') != method:
        issues.append('run_parameters method differs')
    state = read_json('status.json')
    if not isinstance(state, dict) or state.get('status') != 'complete':
        issues.append('status is not complete')
    report_file = required('RUN_REPORT.md')
    if report_file and report_file.stat().st_size < 80: issues.append('run report incomplete')
    for key in ('inference_seconds', 'peak_vram_gib'):
        x = meta.get(key)
        if not isinstance(x, (float, int)) or not np.isfinite(x) or x < 0:
            issues.append('missing measured ' + key)
    if legacy:
        exemptions += ['Existing VGGT timing/resolution are inherited from its original run.',
                       'Legacy VGGT has no archived native point heads; geometry is documented depth unprojection.']
    elif str(meta.get('gpu_physical', meta.get('gpu_physical_index',
                         meta.get('cuda_visible_devices', meta.get('CUDA_VISIBLE_DEVICES', ''))))) != '1':
        issues.append('physical GPU1 provenance missing')
    transforms = read_json('pixel_transforms.json'); statuses = read_json('pose_status.json')
    if not isinstance(transforms, list) or len(transforms) != n:
        issues.append('pixel transform count differs'); transforms = []
    if statuses is None and legacy:
        exemptions.append('Legacy archive did not record per-frame PnP/registration statuses.')
    elif not isinstance(statuses, list) or len(statuses) != n:
        issues.append('pose status count differs'); statuses = []
    if statuses:
        if any(s.get('name', names[i]) != names[i] or not s.get('status') for i,s in enumerate(statuses)):
            issues.append('pose status order/state missing')
    manifest_path = path.parent / 'manifest.csv'
    if manifest_path.is_file():
        with manifest_path.open() as f: manifest = [r for r in csv.DictReader(f) if r['status'] == 'ok']
        source_names = [r['name'] for r in manifest]
        if any(name not in source_names for name in names):
            issues.append('input frame absent from usable manifest')
        else:
            indices = [source_names.index(name) for name in names]
            if indices != sorted(indices): issues.append('original usable frame order changed')
        if meta.get('full_sequence') and names != source_names:
            issues.append('full_sequence claim omits usable source frames')
    else:
        manifest = []
        issues.append('shared manifest.csv missing')
    source_rows = {r['name']: r for r in manifest}
    metrics_file = required('per_frame_metrics.csv'); metrics = []
    if metrics_file:
        with metrics_file.open() as f: metrics = list(csv.DictReader(f))
        if [r.get('name') for r in metrics] != names: issues.append('per-frame metrics order/count differs')
    stems = [Path(name).stem for name in names]
    for directory, ext in [('raw_predictions','.npz'),('pointmaps','.npz'),('confidence','.npz'),
                           ('rgb','.png'),('mask','.png'),('cameras','.json')]:
        files = {p.name for p in (path / directory).glob('*' + ext)}
        if directory == 'cameras':
            auxiliary = files - {stem + ext for stem in stems}
            counts['camera_auxiliary_json'] = len(auxiliary)
            files -= auxiliary
        counts[directory] = len(files)
        if files != {stem + ext for stem in stems}: issues.append(directory + ': per-frame files/count differ')
    cameras = None
    camera_path = required('cameras/cameras.npz')
    if camera_path:
        try:
            with np.load(camera_path, allow_pickle=False) as a: cameras = dict(a)
            if list(cameras['names']) != names: issues.append('camera names/order differ')
            for key, shape in [('c2w',(n,4,4)),('w2c',(n,4,4)),('intrinsics_model',(n,3,3)),('intrinsics_original',(n,3,3))]:
                if cameras[key].shape != shape: issues.append('camera shape differs: ' + key)
        except Exception as exc:
            issues.append('cameras.npz: ' + str(exc)); cameras = None
    raw_keys = ['world_points','world_points_conf']
    if fast and not sfm: raw_keys += ['local_points','local_confidence','pts3d_in_other_view','pts3d_local','conf','conf_local']
    if base: raw_keys += ['depth','depth_conf','depth_metric','pose','intrinsic','extrinsic','pts3d_by_unprojection']
    mapped = local_count = 0; max_inverse = 0.
    for i, (name, stem) in enumerate(zip(names, stems)):
        try:
            with Image.open(path/'rgb'/f'{stem}.png') as im: h,w = im.height,im.width
            with Image.open(path/'mask'/f'{stem}.png') as im: m = np.asarray(im)
            if m.shape != (h,w) or not set(np.unique(m)).issubset({0,255}):
                issues.append(name + ': mask geometry/binarity invalid')
            m = m > 0; t = transforms[i]
            a = np.asarray(t['pixel_transform'], float); ai = np.asarray(t['pixel_transform_inverse'], float)
            max_inverse = max(max_inverse, float(np.abs(ai@a-np.eye(3)).max()))
            if t['name'] != name or t['input_size_wh'] != [w,h]: issues.append(name + ': transform order/dimensions')
            if t.get('mask_interpolation') != 'nearest' or t.get('mask_erosion') != 0 or t.get('crop_ltrb') != [0,0,0,0]:
                issues.append(name + ': mask interpolation/erosion/crop violates policy')
            ow,oh=t['original_size_wh']; rw,rh=t['resize_size_wh']; left,top,right,bottom=t['padding_ltrb']
            source_row = source_rows.get(name)
            if source_row and (ow,oh)!=(int(source_row['width']),int(source_row['height'])):
                issues.append(name + ': original resolution differs from source manifest')
            if len(metrics)==n and source_row and int(metrics[i]['source_index'])!=int(source_row['source_index']):
                issues.append(name + ': original source index changed')
            sx,sy=rw/ow,rh/oh
            expected_a=np.array([[sx,0,left+(sx-1)/2],[0,sy,top+(sy-1)/2],[0,0,1]])
            if not np.allclose(a,expected_a,atol=1e-12) or not np.allclose(ai@a,np.eye(3),atol=1e-12) or (rw+left+right,rh+top+bottom)!=(w,h):
                issues.append(name + ': resize/padding pixel matrix mismatch')
            valid=np.zeros((h,w),bool); valid[top:top+rh,left:left+rw]=True
            if (m&~valid).any(): issues.append(name + ': mask includes padding')
            raw=npz_headers(path/'raw_predictions'/f'{stem}.npz',
                            ('world_points','world_points_conf','local_points','local_confidence'))
            pointmaps=npz_headers(path/'pointmaps'/f'{stem}.npz')
            confidence=npz_headers(path/'confidence'/f'{stem}.npz')
            for key in raw_keys:
                if key not in raw: issues.append(name + ': original prediction missing: ' + key)
            if raw['world_points']['shape']!=(h,w,3):
                issues.append(name + ': world map shape differs')
            if raw['world_points_conf']['shape'] not in ((h,w),(h,w,1)):
                issues.append(name + ': confidence map shape differs')
            for directory,separate in [('pointmaps',pointmaps),('confidence',confidence)]:
                expected_fields={key for key in raw if ('point' in key or 'pts3d' in key or 'depth' in key)} if directory=='pointmaps' else {key for key in raw if 'conf' in key}
                if set(separate)!=expected_fields:
                    issues.append(name + ': separate archive fields differ: ' + directory)
                for key in expected_fields & separate.keys():
                    if any(separate[key][field]!=raw[key][field] for field in ('stored_crc','uncompressed_bytes')):
                        issues.append(name + ': stored member header/CRC fingerprint differs: ' + key)
            if any(record['dtype'] not in ('<f4','<f8','<f2') for key,record in raw.items() if key in ('world_points','world_points_conf')):
                issues.append(name + ': unsupported prediction numeric dtype')
            if fast and not sfm:
                for canonical,native in [('world_points','pts3d_in_other_view'),('world_points_conf','conf'),
                                         ('local_points','pts3d_local'),('local_confidence','conf_local')]:
                    if native in raw and any(raw[canonical][field]!=raw[native][field] for field in ('stored_crc','uncompressed_bytes')):
                        issues.append(name + ': canonical Fast3R field differs from native prediction: ' + native)
            global_mask_n=check_ply(f'pointcloud/global_per_frame/{stem}_mask.ply')
            global_good_n=check_ply(f'pointcloud/global_per_frame/{stem}_mask_confidence.ply')
            if global_mask_n is not None: totals['mask']+=global_mask_n
            if global_good_n is not None: totals['mask_confidence']+=global_good_n
            if len(metrics)==n:
                row=metrics[i]
                if int(row['foreground_pixels'])!=int(m.sum()) or int(row['valid_foreground_points'])!=global_good_n:
                    issues.append(name + ': metrics differ from actual mask/PLY counts')
                if global_mask_n is not None and global_mask_n>int(m.sum()):
                    issues.append(name + ': mask-filtered point count exceeds foreground pixels')
                totals['confidence']+=round(float(row['valid_image_geometry'])*int(valid.sum()))
                if 'mask_conf_q85_points' in row:
                    totals['mask_confidence_q85']+=int(row['mask_conf_q85_points'])
                else: issues.append(name + ': q85 filter point count is missing from export metrics')
            if 'local_points' in raw:
                local_count+=1
                if raw['local_points']['shape']!=(h,w,3) or 'local_confidence' not in raw:
                    issues.append(name + ': local pointmap/confidence header invalid')
                for key in ('raw','confidence','mask','mask_confidence'):
                    check_ply(f'pointcloud/local_per_frame/{stem}_{key}.ply')
            finite_camera=cameras is not None and np.isfinite(cameras['c2w'][i]).all()
            if finite_camera:
                mapped+=1; c2w=cameras['c2w'][i]; w2c=cameras['w2c'][i]; r=c2w[:3,:3]
                if not np.isfinite(cameras['intrinsics_model'][i]).all(): issues.append(name + ': invalid intrinsics')
                if not np.allclose(w2c@c2w,np.eye(4),atol=1e-4) or not np.allclose(c2w[3],[0,0,0,1],atol=1e-5):
                    issues.append(name + ': camera inverse/direction')
                if not np.allclose(r.T@r,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(r),1,atol=1e-4):
                    issues.append(name + ': improper camera rotation')
                if not np.allclose(ai@cameras['intrinsics_model'][i],cameras['intrinsics_original'][i],atol=1e-5):
                    issues.append(name + ': original-pixel intrinsics mapping')
            elif not statuses or statuses[i].get('status') not in ('unmapped','failed','failed_identity_fallback'):
                issues.append(name + ': invalid camera without explicit failure status')
            if sfm and not finite_camera and (global_mask_n!=0 or global_good_n!=0):
                issues.append(name + ': unmapped frame contributed canonical global PLY points')
            if fast and sfm and finite_camera and 'local_points' not in raw:
                issues.append(name + ': registered Fast3R frame lacks supported local map; inspect accepted local call provenance')
            camera_json=read_json(f'cameras/{stem}.json')
            if cameras is not None and isinstance(camera_json,dict):
                for key in ('c2w','w2c','intrinsics_model','intrinsics_original'):
                    if not np.allclose(np.asarray(camera_json[key],float),cameras[key][i],equal_nan=True,atol=1e-7):
                        issues.append(name + ': camera JSON differs: ' + key)
                if statuses and camera_json.get('pose_status') != statuses[i]:
                    issues.append(name + ': per-camera pose status differs')
        except Exception as exc:
            issues.append(f'{name}: artifact read/audit failed: {type(exc).__name__}: {exc}')
        if (i+1)%32==0: print(f'{path.name}: checked {i+1}/{n} frames',flush=True)
    for key in FILTERS:
        if key == 'raw': totals[key]=bundle.get('cloud_counts',{}).get(key,0)
        check_ply(f'pointcloud/global_{key}.ply',totals[key])
        if bundle.get('cloud_counts',{}).get(key)!=totals[key]: issues.append('cloud_counts differs: ' + key)
    nv=check_ply('pointcloud/visualization_only.ply')
    if nv is not None and nv>250000: issues.append('visualization-only cloud exceeds documented limit')
    if fast and not sfm:
        focal_file=read_json('cameras/estimated_focal_lengths.json')
        if focal_file is not None:
            counts['explicit_focal_length_file']=1
            if not isinstance(focal_file,list) or [r.get('name') for r in focal_file]!=names:
                issues.append('estimated focal length file count/order differs')
            elif cameras is not None:
                for i,row in enumerate(focal_file):
                    k=cameras['intrinsics_model'][i]; ko=cameras['intrinsics_original'][i]
                    if not np.allclose([row.get('focal_length_model_px')]*2,[k[0,0],k[1,1]],equal_nan=True):
                        issues.append(names[i] + ': explicit focal length differs from official model intrinsics')
                    if not np.allclose(row.get('focal_length_original_xy_px'),[ko[0,0],ko[1,1]],equal_nan=True):
                        issues.append(names[i] + ': original-coordinate focal lengths differ')
    if base:
        for key in ('world_points_metric_derived_gauge_validated','global_mask_confidence_metric_derived_gauge_validated'):
            if meta.get(key) is not False: issues.append('conditional direct-point gauge flag missing: ' + key)
        if not isinstance(meta.get('scene_metric_scale_predicted'),(int,float)) or len(meta.get('frame_metric_scales_predicted',[]))!=n:
            issues.append('official scene/per-frame predicted scales missing')
        for relative in ('cameras/cameras_metric_derived.npz','head_consistency_diagnostic.json','camera_geometry_checks.json'): required(relative)
        check_ply('pointcloud/global_mask_confidence_metric_derived.ply',totals['mask_confidence'])
        predicted_count = meta.get('metric_geometry_exports',{}).get('depth_unprojection',{}).get('point_count')
        check_ply('pointcloud/global_depth_unprojection_mask_confidence_metric_predicted.ply', predicted_count)
        warnings.append('Point-head metric gauge is unvalidated; depth-scale geometry has predicted, not measured, metric accuracy.')
    if sfm:
        for relative in ('registration_status.json','unmapped_frames.json','keyframes.json','initial_clusters.json','registration_stages.json','local_call_index.json'): read_json(relative)
        required('official_memory_raw.npz')
        counts['official_local_prediction_calls']=len(list((path/'local_predictions').glob('call_*.npz')))
        if not counts['official_local_prediction_calls']: issues.append('SfM original local prediction calls missing')
        if meta.get('mapped_frames',meta.get('mapped_frame_count'))!=mapped:
            issues.append('SfM recorded mapped count differs/missing')
        if meta.get('unmapped_frames')!=n-mapped:
            issues.append('SfM recorded unmapped count differs/missing')
        registration=read_json('registration_status.json')
        if not isinstance(registration,list) or [r.get('name') for r in registration]!=names:
            issues.append('SfM registration status must retain every input frame in source order')
        unmapped=read_json('unmapped_frames.json')
        if not isinstance(unmapped,list) or len(unmapped)!=n-mapped:
            issues.append('SfM unmapped list count differs from nonfinite camera rows')
        elif statuses and [r.get('name') for r in unmapped]!=[r.get('name') for r in statuses if r.get('status')=='unmapped']:
            issues.append('SfM unmapped list differs from full per-frame statuses')
        if fast and isinstance(unmapped, list) and unmapped:
            supplements=read_json('unmapped_local_geometry_index.json')
            if not isinstance(supplements, list) or len(supplements)!=len(unmapped):
                issues.append('Fast3R unmapped local geometry requires one explicit outcome per unmapped input')
            else:
                expected_unmapped_indices={r['frame_index'] for r in unmapped}
                if {r.get('frame_index') for r in supplements}!=expected_unmapped_indices:
                    issues.append('Fast3R unmapped local geometry index differs from failed global registrations')
                local_supplement_count=0
                for row in supplements:
                    index=row.get('frame_index')
                    if not isinstance(index,int) or index not in expected_unmapped_indices:
                        continue
                    if 'source_local_call_id' not in row:
                        if not row.get('status'):
                            issues.append(names[index] + ': missing explanation for unavailable local prediction')
                        continue
                    local_supplement_count+=1; stem=stems[index]
                    supplement=required(f'local_unmapped_predictions/{stem}.npz')
                    provenance=read_json(f'local_unmapped_predictions/{stem}.json')
                    if supplement:
                        try:
                            heads=npz_headers(supplement,('local_points','local_confidence'))
                            w,h=transforms[index]['input_size_wh']
                            if heads['local_points']['shape']!=(h,w,3) or heads['local_confidence']['shape'] not in ((h,w),(h,w,1)):
                                issues.append(names[index] + ': unregistered local head shape differs')
                            if not {'cluster_world_points_unregistered','cluster_c2w_unregistered','source_local_call_id'}.issubset(heads):
                                issues.append(names[index] + ': unregistered local cluster provenance fields missing')
                        except Exception as exc:
                            issues.append(names[index] + ': unregistered local prediction: ' + str(exc))
                    if isinstance(provenance,dict) and (provenance.get('global_registration_status')!='unmapped'
                            or provenance.get('global_bundle_points_and_camera_changed') is not False
                            or provenance.get('name')!=names[index]):
                        issues.append(names[index] + ': local supplement incorrectly claims global registration')
                    for key in ('raw','confidence','mask','mask_confidence'):
                        check_ply(f'pointcloud/local_unmapped_per_frame/{stem}_{key}.ply')
                counts['unmapped_local_prediction_supplements']=local_supplement_count
                if meta.get('unmapped_local_geometry_supplement_count')!=local_supplement_count:
                    issues.append('Fast3R unmapped local prediction count differs from metadata')
    report=dict(schema='foresplat.bundle.audit.v2',path=str(path),method=method,frames=n,finite_camera_count=mapped,
                local_map_count=local_count,pixel_transform_inverse_max_error=max_inverse,required_artifact_counts=counts,
                expected_export_filter_point_counts=totals,issues=issues,warnings=warnings,exemptions=exemptions,passed=not issues,
                array_comparison='NPY headers and stored ZIP member CRC/size fingerprints; full array payload checksums are not recomputed',
                raw_value_evidence='Separate evaluate.py and confidence_control_audit.json read actual raw values; this audit does not repeat that geometric/confidence value pass',
                scope='Actual artifact existence/count, strict JSON, NPY header/stored-CRC agreement, masks/order, pixel matrices, camera inverses, PLY payload length and export-metric counts; not geometric/metric accuracy.')
    json_write(path/'bundle_audit.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundles',nargs='+')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=1,
                        help='Bounded independent-bundle CPU audits; each bundle writes only its own report (default: 1).')
    args=parser.parse_args()
    paths=[str(Path(path).resolve()) for path in args.bundles]
    if len(set(paths))!=len(paths):
        parser.error('Duplicate bundle paths would write the same report concurrently.')
    reports=[]
    if args.workers==1:
        for path in args.bundles:
            report=audit(path); reports.append(report)
            print(json.dumps(report,ensure_ascii=False),flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(audit,path):path for path in args.bundles}
            for future in as_completed(pending):
                report=future.result(); reports.append(report)
                print(json.dumps(report,ensure_ascii=False),flush=True)
    raise SystemExit(0 if all(r['passed'] for r in reports) else 1)

