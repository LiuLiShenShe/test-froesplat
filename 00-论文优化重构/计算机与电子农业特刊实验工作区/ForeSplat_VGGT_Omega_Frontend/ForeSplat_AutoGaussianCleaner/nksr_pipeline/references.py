"""Attach original capture RGB to completed, matched mesh comparison panels."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import (OUT, PLANTS, ROOT, PacedFile, Stage, load_image,
                     save_image, stamp, write)


def run(plant):
    out = OUT/plant/'visualizations'
    audit = out/'reference_rgb'
    if (audit/'complete.json').exists():
        return
    quality = json.loads((out/'metrics.json').read_text())
    spec = json.loads((ROOT/f'configs/scenes/{plant}.json').read_text())
    dataset = Path(spec['dataset'])
    transforms = {Path(f['image_name']).stem:f for f in
        json.loads((dataset/'metadata/preprocess_transforms.json').read_text())['frames']}
    manifest = {Path(f['image_name']).stem:f for f in
        json.loads((dataset/'metadata/image_manifest.json').read_text())['frames']}
    records = []
    with Stage(audit, gpu=False) as stage:
        meter = stage.meter
        for order, camera in enumerate(quality['cameras']):
            name = camera['name']
            transform, frame = transforms[name], manifest[name]
            # These saved cameras cover the entire original capture canvas.
            # Refuse an unimplemented crop/pad mapping instead of guessing it.
            assert not transform['crop_applied']
            assert not any(transform[k] for k in ['pad_left','pad_top','pad_right','pad_bottom'])
            assert name.startswith('crop_')
            source = Path(spec['raw_rgb_dir'])/(name[len('crop_'):]+'.jpg')
            with PacedFile(source, 'rb', meter) as stream:
                with Image.open(stream) as image:
                    assert image.size == (transform['original_width'], transform['original_height'])
            size = camera['width'], camera['height']
            original = load_image(source, size, meter)
            training = load_image(dataset/frame['exported_image'], size, meter)
            mask = load_image(dataset/frame['exported_mask'], size, meter, mask=True)
            difference = float(np.abs(original-training)[mask].mean())
            # A name/shape match alone is insufficient evidence of frame identity.
            assert difference < .05, (plant, name, difference)
            save_image(out/f'training_reference_{order:02}.png', training, meter)
            save_image(out/f'reference_{order:02}.png', original, meter)
            records.append(dict(camera=name, source=stamp(source),
                training_source=str((dataset/frame['exported_image']).resolve()),
                preprocess_source=str((dataset/'metadata/preprocess_transforms.json').resolve()),
                mapping='Identical full original image canvas; resize to the recorded evaluation camera dimensions; no crop, pad, mask, or exposure change',
                foreground_rgb_mean_absolute_difference_0_1=difference))
        canvas = Image.new('RGB', (1200,2320), 'white')
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
        for row,(prefix,label) in enumerate([('reference','Original RGB'),('C1_GGGS','C1 GGGS'),
                ('NKSR_raw','NKSR Raw'),('NKSR_post','NKSR Post')]):
            for order,camera in enumerate(quality['cameras']):
                with PacedFile(out/f'{prefix}_{order:02}.png','rb',meter) as stream:
                    with Image.open(stream) as source:
                        image=source.convert('RGB')
                image.thumbnail((300,535))
                canvas.paste(image,(order*300+(300-image.width)//2,row*580+45))
                draw.text((order*300+4,row*580+5),label,fill='black',font=font)
                draw.text((order*300+4,row*580+25),camera['name'],fill='black',font=font)
        with PacedFile(out/'mesh_comparison.png','wb',meter) as stream:
            canvas.save(stream,format='PNG')
        write(audit/'sources.json',dict(plant=plant,views=records,
            original_rgb='Unmasked original capture RGB, not the black-background RAP training image',
            render_background='C1 and mesh rows retain fixed gray 0.5; original photographs retain their actual scene background'))
    quality['reference_rgb_provenance']='reference_rgb/sources.json'
    write(out/'metrics.json',quality)
    write(audit/'complete.json',dict(status='complete',plant=plant,views=4))
    print('Original RGB references attached:',plant,flush=True)


def main():
    for plant in PLANTS:
        run(plant)


if __name__=='__main__':main()
