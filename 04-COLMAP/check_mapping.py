import json, os

sam = json.load(open(r'D:\CAAS\03-SAM\CaoMei1\segmentation_log.json'))
fft = json.load(open(r'D:\CAAS\02-FFT\CaoMei1\filter_log.json'))

kept = [p['file'] for p in fft['per_frame'] if p['kept']]
sam_imgs = [s['image'] for s in sam]

print(f"SAM entries: {len(sam)}")
print(f"FFT kept: {len(kept)}")
print(f"First 5 SAM images: {sam_imgs[:5]}")
print(f"First 5 FFT kept: {kept[:5]}")
print(f"SAM imgs == FFT kept: {sam_imgs == kept}")

# So mask_0000.png -> sam[0]['image'] -> original frame name
# mapping: sequential index -> original frame name
for i in range(min(5, len(sam))):
    print(f"  mask_{i:04d}.png / crop_{i:04d}.png -> {sam_imgs[i]}")
