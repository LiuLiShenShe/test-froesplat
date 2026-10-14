from __future__ import annotations

import argparse
import importlib.metadata
import math

from common import STAGE, import_paths, read_csv, sha256, write_csv
from evaluate import resize_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample")
    parser.add_argument("--group")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    import_paths()
    import torch
    import lpips
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    rows = [row for row in read_csv(STAGE / "HELDOUT_PER_VIEW_METRICS.csv")
            if row["status"] == "PASS" and not row["shared_model_alias"]
            and (not args.sample or row["sample"] == args.sample)
            and (not args.group or row["group"] == args.group)]
    if args.limit:
        rows = rows[:args.limit]
    model = lpips.LPIPS(net="vgg", version="0.1", verbose=False).eval().cuda()
    results = []
    with torch.no_grad():
        for row in rows:
            assert sha256(row["float_render_path"]) == row["float_render_sha256"]
            assert sha256(row["reference_path"]) == row["reference_original_sha256"]
            rgb = torch.load(row["float_render_path"], map_location="cpu", weights_only=True)["rgb"].cuda()
            reference = resize_reference(row["reference_path"], int(row["render_width"]), int(row["render_height"])).cuda()
            values = {"PSNR": psnr(rgb[None], reference[None]).item(),
                      "SSIM": ssim(rgb, reference).item(),
                      "LPIPS": model(rgb[None], reference[None], normalize=True).item()}
            for field, value in values.items():
                difference = abs(value - float(row[field]))
                assert difference <= 1e-6, (row["sample"], row["group"], row["registered_name"], field, difference)
                results.append({"sample": row["sample"], "group": row["group"], "registered_name": row["registered_name"],
                                "metric": field, "recomputed": value, "original": row[field],
                                "abs_difference": difference, "status": "PASS"})
    write_csv(STAGE / "reports/FLOAT_RENDER_METRIC_RECOMPUTATION.csv", results)
    print("Metric recomputation PASS", len(rows), "real float renders", len(results), "metric values")


if __name__ == "__main__":
    main()
