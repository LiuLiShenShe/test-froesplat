"""CPU actual-data reproduction of the already saved first/last background PNG."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from geometry_rgb_adapter import restore_geometry_rgb

def main():
    data=json.loads((HERE/'prepared_sources.json').read_text())
    rows=[]
    for source in [data['source_ids'][0],data['source_ids'][-1]]:
        original=next(r for r in data['pixel_assertions'] if r['source_index']==source)
        result=restore_geometry_rgb(original['rap_path'],original['mask_path'],original['original_jpeg_path'])
        png=HERE/'prepared_images/background_restored'/original['image_name']
        with Image.open(png) as im:saved=np.array(im.convert('RGB'))
        equal=bool(np.array_equal(result.geometry_rgb,saved))
        if not equal:raise AssertionError('Reusable adapter does not reproduce actual saved experimental input')
        rows.append(dict(source_index=source,prepared_image=str(png),adapter_matches_saved_experimental_RGB_pixelwise=equal,metadata=result.metadata))
    adapter=HERE.parent/'geometry_rgb_adapter.py'
    output=dict(status='passed',adapter_source=str(adapter),adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
        scope='Actual first/last fixed24 frames; all decoded adapter RGB pixels compared to prior experimental PNG. No neural inference or new208 data processing.',
        frames=rows,new_inference_executed=False,historical_files_changed=False)
    (HERE/'geometry_rgb_adapter_validation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Passed exact saved input reproduction',[(r['source_index'],r['adapter_matches_saved_experimental_RGB_pixelwise']) for r in rows])

if __name__=='__main__':main()
