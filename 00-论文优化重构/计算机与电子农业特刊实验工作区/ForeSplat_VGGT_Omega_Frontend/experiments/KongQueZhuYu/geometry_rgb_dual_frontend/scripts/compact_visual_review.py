"""Small review sheets from saved evidence; native texture pixels stay native."""
from pathlib import Path
import json
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'render_comparison/visualizations'
OUT = ROOT / 'render_comparison/compact_review'
LABELS = ['GT', 'E1', 'E5', 'E7', 'E8', 'E9', 'E10']


def native_sheet(ids, kind, name):
    # Crop central 192 pixels of each existing 320-pixel window, no resize.
    canvas = Image.new('RGB', (192 * 7, 216 * len(ids)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, sid in enumerate(ids):
        path = SOURCE / f'source_{sid:04d}_{kind}.png'
        with Image.open(path) as im:
            assert im.size == (2240, 344)
            for col, label in enumerate(LABELS):
                draw.text((col * 192 + 4, row * 216 + 5), f'{sid:04d} {label}', fill='black')
                crop = im.crop((col * 320 + 64, 24 + 64, col * 320 + 256, 24 + 256))
                canvas.paste(crop, (col * 192, row * 216 + 24))
    canvas.save(OUT / name)


def overview(ids, kind, name, labels, panel_width):
    canvas = Image.new('RGB', (180 * len(labels), 344 * len(ids)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, sid in enumerate(ids):
        with Image.open(SOURCE / f'source_{sid:04d}_{kind}.png') as im:
            for col, label in enumerate(labels):
                # Existing overview uses nearest; this further reduction is display only.
                crop = im.crop((col * panel_width, 24, (col + 1) * panel_width, 504))
                crop = crop.resize((180, 320), Image.Resampling.NEAREST)
                canvas.paste(crop, (col * 180, row * 344 + 24))
                draw.text((col * 180 + 4, row * 344 + 5), f'{sid:04d} {label}', fill='black')
    canvas.save(OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    native_sheet([0, 26, 53, 79], 'leaf_texture_GT_selected_native320', 'texture_early_native.png')
    native_sheet([105, 131, 157, 183], 'leaf_texture_GT_selected_native320', 'texture_late_native.png')
    native_sheet([0, 105], 'leaf_edge_native320', 'edge_native.png')
    overview([0, 105], 'whole_plant', 'whole_plant_overview.png', LABELS, 270)
    # Same representative source, all methods, three native renderer channels.
    diagnostic = Image.new('RGB', (1080, 1000), 'white')
    draw = ImageDraw.Draw(diagnostic)
    for row, kind in enumerate(['alpha', 'normal', 'depth_per_extent']):
        with Image.open(SOURCE / f'source_0105_{kind}.png') as im:
            crop = im.crop((0, 24, 1620, 504)).resize((1080, 320), Image.Resampling.NEAREST)
            diagnostic.paste(crop, (0, row * 333 + 13))
            for col, label in enumerate(LABELS[1:]):
                draw.text((col * 180 + 3, row * 333), f'{kind} {label}', fill='black')
    diagnostic.save(OUT / 'renderer_channels_0105.png')
    geometry = Image.new('RGB', (1248, 980), 'white')
    draw = ImageDraw.Draw(geometry)
    for name, y, size in [
        ('foreground_pointclouds.png', 18, (1248, 520)),
        ('gaussian_means_all_bounds.png', 560, (1248, 416)),
    ]:
        with Image.open(ROOT / 'geometry_comparison' / name) as im:
            if name == 'gaussian_means_all_bounds.png':
                im = im.crop((0, 900, 2700, 1800))  # E8/E9/E10 row, same aspect ratio.
            geometry.paste(im.resize(size, Image.Resampling.NEAREST), (0, y))
        draw.text((4, y - 15), name + ' (display overview only)', fill='black')
    geometry.save(OUT / 'geometry_overview.png')
    protocol = {
        'source_ids': [0, 26, 53, 79, 105, 131, 157, 183],
        'native_texture_and_edge': 'Central192x192 crop of each saved320x320 GT-selected window; no interpolation',
        'whole_channel_geometry': 'Nearest display reduction only; original full artifacts remain available',
        'no_metric_recalculation_from_previews': True,
        'purpose': 'Bound image payload for direct model vision, avoiding per-file full-size reads',
        'unchanged_predictions_training_masks_checkpoints': True,
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for p in sorted(OUT.glob('*.png')):
        with Image.open(p) as im:
            print(p.name, im.size, p.stat().st_size)


if __name__ == '__main__':
    main()
