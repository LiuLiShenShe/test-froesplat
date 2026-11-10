"""Compact direct-review panels from actual native-camera Mesh image caches."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'comparison/visualizations/scientific_review'
DEST.mkdir(exist_ok=True)
OLD = ROOT.parent / 'geometry_mesh_final_selection'
routes = [
    ('M4', OLD / 'meshes/M4_E10_GGGS_TSDF/post.ply'),
    ('P3_shared', ROOT / 'P3_GS2Mesh/meshes_shared_depth_range/post.ply'),
    ('E12_TSDF', ROOT / 'E12_GGGS_EO/TSDF/post.ply'),
    ('E13_QGS', ROOT / 'E13_QGS/meshes/native_TSDF/post.ply'),
    ('E13_official_top10', ROOT / 'E13_QGS/meshes/native_TSDF/official_top10_post.ply'),
    ('E14_initial', ROOT / 'E14_2D_SuGaR/initial_mesh/post.ply'),
    ('E14_refined', ROOT / 'E14_2D_SuGaR/refinement/mesh_post/post.ply'),
]
bundles = {}
for marker in (ROOT / 'E14_2D_SuGaR/visual_views/mesh').glob('*/complete.json'):
    info = json.loads(marker.read_text())
    if info.get('status') == 'complete' and all((marker.parent/f'fixed_{sid:04d}/normal_camera.png').is_file() for sid in [0,26,53,79,105,131,157,183]):
        bundles[str(Path(info['source_mesh']).resolve())] = marker.parent
for marker in (ROOT / 'comparison/visualizations/mesh').glob('*/complete.json'):
    info = json.loads(marker.read_text())
    if info.get('status') == 'complete' and info.get('normal_space') == 'camera':
        bundles[str(Path(info['source_mesh']).resolve())] = marker.parent
rois = json.loads((ROOT / 'comparison/visualizations/mesh_comparison/PROTOCOL.json').read_text())['records']
records = []
for roi in rois:
    sid = roi['source_id']
    bounds = roi['ROIs']['leaf_edge']
    selected = [(name, path.resolve(), bundles[str(path.resolve())]) for name, path in routes if str(path.resolve()) in bundles]
    if not selected:
        continue
    panel = Image.new('RGB', (320 * len(selected), 696), '#202020')
    draw = ImageDraw.Draw(panel)
    sources = []
    for col, (name, mesh, folder) in enumerate(selected):
        for row, channel in enumerate(['gray', 'normal_camera']):
            source = folder / f'fixed_{sid:04d}' / f'{channel}.png'
            crop = Image.open(source).convert('RGB').crop(bounds)
            panel.paste(crop.resize((320, 320), Image.Resampling.LANCZOS), (320 * col, 348 * row + 28))
            draw.text((320 * col + 3, 348 * row + 5), name + ' / ' + channel, fill='white')
            sources.append(dict(method=name, mesh=str(mesh), channel=channel, source_image=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    target = DEST / f'leaf_edge_{sid:04d}_gray_normal.png'
    panel.save(target)
    overviews={}
    for channel in ['gray','normal_camera','wireframe']:
        overview=Image.new('RGB',(270*len(selected),508),'#202020')
        draw=ImageDraw.Draw(overview)
        for col,(name,mesh,folder) in enumerate(selected):
            source=folder/f'fixed_{sid:04d}'/(channel+'.png')
            overview.paste(Image.open(source).convert('RGB').resize((270,480),Image.Resampling.LANCZOS),(270*col,28))
            draw.text((270*col+3,5),name+' / '+channel,fill='white')
            sources.append(dict(method=name,mesh=str(mesh),channel=channel,source_image=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        destination=DEST/f'whole_{sid:04d}_{channel}.png'
        overview.save(destination);overviews[channel]=str(destination)
    records.append(dict(source_id=sid, bounds_native=bounds, output=str(target), overview_outputs=overviews, methods=[x[0] for x in selected], sources=sources))
(DEST / 'PROTOCOL.json').write_text(json.dumps(dict(status='actual_available_review_panels', records=records, camera='Existing exact corrected native source cameras; no new view estimation', display='Same GT-selected native leaf-edge window; gray and camera normal, no synthetic plant imagery', scope='Availability does not imply direct visual inspection; actual reviewed files are separately recorded in VISUAL_REVIEW.json'), ensure_ascii=False, indent=2) + '\n')
print('Scientific review panels', len(records), sorted({m for r in records for m in r['methods']}), flush=True)
