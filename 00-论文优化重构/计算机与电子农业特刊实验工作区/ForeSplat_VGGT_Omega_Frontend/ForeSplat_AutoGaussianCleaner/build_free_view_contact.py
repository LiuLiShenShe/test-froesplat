#!/usr/bin/env python3
"""Present already rendered identical free cameras, with image source hashes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw, ImageOps
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json


def build(output):
    output = output.resolve()
    marker = json.loads((output/'COMPLETE.json').read_text())
    if (output/'INVALIDATED.json').exists() or sha(output/'cleaned_gaussians.ply') != marker['native_sha256']:
        raise ValueError('Free-view contact requires the unaltered completed native model')
    quality = output/'quality_inspection'
    inspection = json.loads((quality/'QUALITY_INSPECTION.json').read_text())
    if inspection['native_sha256'] != marker['native_sha256']:
        raise ValueError('Existing native inspection images belong to another model')
    definitions = json.loads((output/'visualizations/automatic_free_views.json').read_text())['definitions']
    width, height = 400, 360
    canvas = Image.new('RGB', (width*4, height*len(definitions)), '#dddddd')
    draw = ImageDraw.Draw(canvas)
    records = []
    columns = [('original', 'white'), ('cleaned', 'white'), ('original', 'black'), ('cleaned', 'black')]
    for row, definition in enumerate(definitions):
        sizes = []
        for column, (label, background) in enumerate(columns):
            path = quality/label/'free'/f'{definition["name"]}_{background}.png'
            with Image.open(path) as original:
                sizes.append(original.size)
                fitted = ImageOps.contain(original.convert('RGB'), (width-8, height-28))
                canvas.paste(fitted, (column*width+(width-fitted.width)//2, row*height+24+(height-28-fitted.height)//2))
            draw.text((column*width+4, row*height+4), f'{definition["name"]} | {label} {background}', fill='black')
            records.append(dict(view=definition['name'], label=label, background=background,
                                image=str(path.resolve()), image_sha256=sha(path)))
        if len(set(sizes)) != 1:
            raise ValueError('Before/after native free-view render sizes differ')
    path = quality/'FREE_VIEW_CONTACT_SHEET.png'
    canvas.save(path)
    write_json(quality/'FREE_VIEW_CONTACT_SHEET.json', dict(native_sha256=marker['native_sha256'],
        input_sha256=marker['input_sha256'], contact_sheet=str(path), contact_sheet_sha256=sha(path),
        definitions=definitions, images=records, source_sha256=sha(Path(__file__)),
        scope='All existing automatic native free cameras, both backgrounds; resized for presentation only; no camera, ROI, threshold or model is modified'))
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    for output in args.outputs:
        build(output)


if __name__ == '__main__':
    main()
