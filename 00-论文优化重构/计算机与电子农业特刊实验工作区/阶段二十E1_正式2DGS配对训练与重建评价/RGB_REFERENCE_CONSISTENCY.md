# Shared raw RGB reference consistency

Primary metrics compare all groups to the same unmasked canonical undistorted RGB.
The original and resized-tensor SHA256 are asserted equal in every paired view.
PIL channelwise BICUBIC at actual resolution=2 is identical for every group;
metrics use float renders, not PNG quantization. All frozen views remain present.

204 table rows, 177 actual model-camera renders;
HongZhang B is a labelled C-model alias, not another rendering/training experiment.

SSIM: production 11px Gaussian window, sigma=1.5, C1=0.01²/C2=0.03².
PSNR: production max-intensity=1. LPIPS: package 0.1.4, VGG v0.1, normalize=True.
Weights, code SHA256 and package versions are in each VALIDATION.json.

Native iteration-7000/30000 metrics compare treatment-specific composites and are
secondary only; they never enter this common-reference table.
