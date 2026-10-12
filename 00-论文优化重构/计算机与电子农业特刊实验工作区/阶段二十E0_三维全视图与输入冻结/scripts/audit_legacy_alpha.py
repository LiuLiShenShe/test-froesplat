from __future__ import annotations

import numpy as np

from build_datasets import remap_nearest
from phase20e_common import DATA, ROOT, WORK, binary_mask, image_array, read_csv, sha256, write_json


def run():
    rows = read_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv")
    results = []
    for frame in ("0160", "0210", "0232", "0233"):
        row = next(row for row in rows if row["sample"] == "CaoMei2" and row["raw_frame"] == frame)
        raw = image_array(ROOT / row["raw_rgb_path"])
        rgba = image_array(ROOT / row["historical_rgba_path"], "RGBA")
        historical = image_array(ROOT / row["colmap_input_path"])
        mask = rgba[..., 3] > 0
        composite = raw.copy()
        composite[~mask] = 0
        results.append({"sample": "CaoMei2", "frame": frame, "rgba_path": row["historical_rgba_path"],
                        "rgba_sha256": sha256(ROOT / row["historical_rgba_path"]),
                        "rgba_rgb_equal_full_raw": bool(np.array_equal(rgba[..., :3], raw)),
                        "rgba_alpha_composite_equal_registered_crop": bool(np.array_equal(composite, historical)),
                        "different_pixels": int(np.any(composite != historical, axis=2).sum()),
                        "historical_missing_alpha_recovery_certified": False})
    for sample in ("CaoMei2", "WanNianQing1"):
        row = next(row for row in rows if row["sample"] == sample and row["historical_mask_verified"] == "True" and not row["historical_rgba_error"])
        rawmask = binary_mask(ROOT / row["historical_mask_path"])
        rgba = image_array(ROOT / row["historical_rgba_path"], "RGBA")
        alpha = rgba[..., 3] > 0
        if alpha.shape == rawmask.shape:
            expected = rawmask
            description = "RAW_GRID_IDENTITY"
        else:
            with np.load(DATA / sample / "common/camera_remap.npz") as maps:
                expected = remap_nearest(rawmask, maps["map_x"], maps["map_y"], maps["valid"])
            description = "OFFICIAL_NEAREST_CAMERA_MAP"
        results.append({"sample": sample, "frame": row["raw_frame"], "comparison": description,
                        "alpha_pixels_changed": int((alpha != expected).sum()),
                        "alpha_pixelwise_identity": bool(np.array_equal(alpha, expected)),
                        "rgba_path": row["historical_rgba_path"], "rgba_sha256": sha256(ROOT / row["historical_rgba_path"])})
    write_json(WORK / "reports/LEGACY_RGBA_RECOVERY_PROOF.json", results)
    write_json(WORK / "logs/LEGACY_A_RECOVERY_RECHECK.json", {"status": "AUDIT_COMPLETE", "historical_A_restored": False,
               "reason": "Readable RGBA is not an identity-certified replacement for damaged original masks; alpha/version conflicts measured",
               "formal_training": False})
    print("LEGACY_ALPHA_AUDIT_COMPLETE", len(results), "historical_A_restored=False", flush=True)


if __name__ == "__main__":
    run()
