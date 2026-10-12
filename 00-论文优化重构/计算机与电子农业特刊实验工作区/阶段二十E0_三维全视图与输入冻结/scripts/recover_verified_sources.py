from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phase20e_common import DATA, ROOT, WORK, image_array, pixel_hash, read_csv, relative, safe_copy, sha256, write_csv, write_json


def run():
    rows = read_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv")
    status_path = WORK / "reports/SOURCE_RECOVERY_STATUS.json"
    manifest_path = WORK / "reports/SOURCE_RECOVERY_MANIFEST.csv"
    if status_path.exists() and manifest_path.exists() and all("original_raw_rgb_path" in row for row in rows):
        status = json.loads(status_path.read_text())
        if status.get("status") != "PASS" or status.get("registered_views") != len(rows):
            raise ValueError("Existing recovery status is incomplete; preserve the evidence and inspect it")
        if any(row["common_raw_domain_proven"] != "True" or row["raw_rgb_error"] for row in rows):
            raise ValueError("Recovered identity manifest is no longer complete")
        for recovery in read_csv(manifest_path):
            if recovery.get("recovered_path") and sha256(recovery["recovered_path"]) != recovery["sha256"]:
                raise ValueError("Previously recovered RGB changed")
        print("RECOVERY_CACHE_VERIFIED", len(rows), flush=True)
        return
    recoveries = []
    for row in rows:
        row["original_raw_rgb_path"] = row["raw_rgb_path"]
        row["original_raw_rgb_sha256"] = row["raw_rgb_sha256"]
        row["original_raw_rgb_error"] = row["raw_rgb_error"]
        row["source_recovery"] = "NOT_REQUIRED"
        if row["sample"] == "CaoMei2" and row["common_raw_domain_proven"] != "True":
            raw = image_array(ROOT / row["raw_rgb_path"])
            historical = image_array(ROOT / row["colmap_input_path"])
            foreground = np.any(historical != 0, axis=2)
            if historical.shape != raw.shape or not np.array_equal(historical[foreground], raw[foreground]):
                raise ValueError("Secondary registered crop does not prove original RGB coordinate identity")
            row["common_raw_domain_proven"] = "True"
            row["crop_pixel_identity_verified"] = "True"
            row["source_recovery"] = "COORDINATE_IDENTITY_FROM_REGISTERED_COLMAP_CROP_NONBLACK_PIXELS"
            recoveries.append({"sample": row["sample"], "registered_name": row["registered_name"], "source": row["colmap_input_path"],
                               "sha256": row["colmap_input_sha256"], "raw_frame": row["raw_frame"],
                               "recovery": row["source_recovery"], "nonblack_pixels": int(foreground.sum()),
                               "raw_foreground_pixels_sha256": pixel_hash(raw[foreground]),
                               "crop_foreground_pixels_sha256": pixel_hash(historical[foreground]),
                               "legacy_alpha_restored": False})
        if row["sample"] == "HongZhang" and row["raw_rgb_error"]:
            source = ROOT / row["colmap_input_path"]
            rgb = image_array(source)
            if rgb.shape[:2] != (3840, 2160) or np.all(rgb == 0, axis=2).mean() > 0.01:
                raise ValueError("Registered input is not a full raw-RGB recovery")
            recovered = DATA / row["sample"] / "recovered_raw_rgb" / f"{row['raw_frame']}.jpg"
            safe_copy(source, recovered)
            input_path = DATA / row["sample"] / "inputs" / f"{row['sample']}_{row['raw_frame']}.jpg"
            if input_path.exists() and sha256(input_path) == row["original_raw_rgb_sha256"]:
                preserved = DATA / row["sample"] / "corrupt_copies_preserved" / input_path.name
                preserved.parent.mkdir(parents=True, exist_ok=True)
                input_path.rename(preserved)
            safe_copy(source, input_path)
            row["raw_rgb_path"] = str(recovered)
            row["raw_rgb_sha256"] = sha256(recovered)
            row["raw_rgb_width"], row["raw_rgb_height"] = 2160, 3840
            row["raw_rgb_mode"] = "RGB"
            row["raw_rgb_error"] = ""
            row["common_raw_domain_proven"] = "True"
            row["colmap_input_identity_verified"] = "True"
            row["source_recovery"] = "CANONICAL_RAW_RGB_FROM_READABLE_REGISTERED_COLMAP_INPUT"
            recoveries.append({"sample": row["sample"], "registered_name": row["registered_name"], "source": row["colmap_input_path"],
                               "sha256": row["raw_rgb_sha256"], "raw_frame": row["raw_frame"], "recovery": row["source_recovery"],
                               "recovered_path": str(recovered), "original_corrupt_file_preserved": row["original_raw_rgb_path"],
                               "original_corrupt_sha256": row["original_raw_rgb_sha256"], "legacy_alpha_restored": False})
    if any(row["common_raw_domain_proven"] != "True" for row in rows):
        raise ValueError("Not all sources recoverable; do not infer unproven frames")
    write_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv", rows)
    write_csv(WORK / "reports/SOURCE_RECOVERY_MANIFEST.csv", recoveries)
    write_json(WORK / "reports/SOURCE_RECOVERY_STATUS.json", {"status": "PASS", "registered_views": len(rows),
                "coordinate_secondary_proof": sum(row["sample"] == "CaoMei2" for row in recoveries),
                "raw_rgb_recovered": sum(row["sample"] == "HongZhang" for row in recoveries), "original_files_overwritten": 0})
    print(json.dumps(recoveries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
