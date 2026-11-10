"""Compact actual existing Mesh crops for direct review of remaining fixed views."""
import json,csv,hashlib
from pathlib import Path
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT/'comparison/visualizations/mesh_comparison'
DEST=OUT/'comparison/visualizations/fixed_view_continuation'

def main():
    DEST.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    sources={r['mesh_id']:r['evaluated_mesh_path'] for r in csv.DictReader((OUT/'comparison/ALL_MESH_EXPERIMENTS.csv').open()) if r['mesh_stage']=='post'}
    selected=['M4_E10_GGGS_TSDF','P2_TSDF','P3_shared','E12_TSDF','P1','P4_median_mask_alpha']
    selected=[name for name in selected if name in protocol['methods']]
    records=[]
    for sid in [26,79,105,183]:
        camera=next(r for r in protocol['records'] if r['source_id']==sid)
        paths=[];canvas=Image.new('RGB',(320*len(selected),680),'#202020');draw=ImageDraw.Draw(canvas)
        draw.text((5,5),f'Actual source {sid}: shared frozen leaf-edge gray / camera normals. Available post routes only.',fill='white')
        for column,name in enumerate(selected):
            draw.text((320*column+4,25),name,fill='white')
            for row,label in enumerate(['leaf_edge_gray','leaf_edge_normal_camera']):
                path=ROOT/name/f'{sid:04d}'/(label+'.png');assert path.is_file(),path
                with Image.open(path) as image:canvas.paste(image.convert('RGB').resize((320,320),Image.Resampling.LANCZOS),(320*column,45+320*row))
                paths.append(dict(method=name,source_mesh=str(Path(sources[name]).resolve(strict=True)),
                    source_image=str(path.resolve()),source_image_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),channel=label))
        output=DEST/f'source_{sid:04d}_leaf_edge_gray_normal.png';canvas.save(output)
        records.append(dict(source_id=sid,output=str(output.resolve()),ROIs=camera['ROIs'],sources=paths))
    (DEST/'PROTOCOL.json').write_text(json.dumps(dict(status='complete_actual_available_fixed_view_checks',records=records,
        methods=selected,views=[26,79,105,183],source_protocol=str((ROOT/'PROTOCOL.json').resolve()),
        scope='Actual common leaf-edge windows, gray and signed camera-normal display. Limited available-method supplement; not a terminal E13/E14 substitution or physical truth. Original images/Mesh unchanged.'),indent=2)+'\n')
    print('Actual remaining fixed view checks',len(records),flush=True)

if __name__=='__main__':main()
