#!/usr/bin/env python3
"""Fast3R smoke test on 3 KongQueZhuYu frames.

Verifies output shapes, PnP pose estimation, and VRAM before full inference.
Run: .venv-fast3r/bin/python frontends/fast3r/smoke_test.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import time
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "fast3r"))

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

WEIGHTS = os.path.join(ROOT, "weights", "fast3r")   # local dir with model.safetensors
IMAGES = [
    "/data/fj/F2DMAS/03-SAM/KongQueZhuYu/crop_0000.png",
    "/data/fj/F2DMAS/03-SAM/KongQueZhuYu/crop_0066.png",
    "/data/fj/F2DMAS/03-SAM/KongQueZhuYu/crop_0132.png",
]

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    model = Fast3R.from_pretrained(WEIGHTS)
    device = torch.device("cuda")
    model = model.to(device)
    lit = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval(); lit.eval()

    images = load_images(IMAGES, size=512, verbose=True)
    print("[smoke] loaded", len(images), "views; keys:", sorted(images[0].keys()))
    print("[smoke] img shape:", tuple(images[0]["img"].shape),
          "| true_shape:", images[0]["true_shape"].tolist(),
          "| intrinsics:\n", images[0]["intrinsics"])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, prof = inference(images, model, device, dtype=torch.bfloat16, verbose=True, profiling=True)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    preds = out["preds"]
    print("[smoke] inference %.1fs | peak %.2f GiB | %d preds" % (dt, peak, len(preds)))
    for i, p in enumerate(preds):
        for k in ("pts3d_in_other_view", "pts3d_local", "conf", "conf_local"):
            if k in p:
                print(f"  pred[{i}] {k}: {tuple(p[k].shape)} {p[k].dtype}")

    t1 = time.time()
    with torch.inference_mode():
        poses, focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            preds, niter_PnP=100,
            focal_length_estimation_method="first_view_from_global_head")
    dt2 = time.time() - t1
    c2w = poses[0]
    print("[smoke] PnP %.1fs | poses list len %d | first %s" % (dt2, len(poses), tuple(np.asarray(c2w[0]).shape)))
    for i, p in enumerate(c2w):
        print(f"  view {i} c2w:\n{np.asarray(p)}")
    print("[smoke] focals:", focals)

    out_dir = os.path.join(ROOT, "experiments", "KongQueZhuYu", "fast3r_smoke")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"inference_s": dt, "pnp_s": dt2, "peak_gib": round(peak, 2),
               "n_views": len(preds), "focals": [float(f) for f in focals[0]]},
              open(os.path.join(out_dir, "smoke_meta.json"), "w"), indent=2)
    np.savez(os.path.join(out_dir, "smoke_poses.npz"), c2w=np.asarray(c2w))
    print("[smoke] saved to", out_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
