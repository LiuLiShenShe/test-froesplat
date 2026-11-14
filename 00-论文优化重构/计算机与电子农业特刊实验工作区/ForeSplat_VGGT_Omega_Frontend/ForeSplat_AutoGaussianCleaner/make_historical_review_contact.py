#!/usr/bin/env python3
"""Arrange actual browser captures for direct comparison without model synthesis."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();folder=a.experiment/'comparison/official_supersplat'
    r=json.loads((folder/'REVIEW.json').read_text())
    order=['B0_original','B1_historical_best','B2_stage1','B3_stage2']
    records={m['label']:m for m in r['models']}
    labels=[k for k in order if k in records]
    keys=sorted(set.intersection(*[{(c['view'],c['background']) for c in records[k]['captures']} for k in labels]))
    for view,bg in keys:
        panel=Image.new('RGB',(1440,1128),'#e7e7e7');draw=ImageDraw.Draw(panel)
        for n,label in enumerate(labels):
            path=next(c['path'] for c in records[label]['captures'] if c['view']==view and c['background']==bg)
            im=Image.open(path).convert('RGB');im.thumbnail((720,540))
            x=(n%2)*720;y=(n//2)*564
            panel.paste(im,(x,y+24));draw.text((x+8,y+5),label+' / '+view+' / '+bg,fill='black')
        panel.save(folder/f'compare_{view}_{bg}.png')
    print(f'Arranged {len(keys)} actual browser comparisons')


if __name__=='__main__':main()
