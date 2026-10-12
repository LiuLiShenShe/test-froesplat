from __future__ import annotations

from direct_common import FIX, PHASEC, WORK, local_path, read_csv, read_mask, sha256, write_csv
from evaluate_test32 import overlap


def main():
    manifest = read_csv(PHASEC / "configs/PHASE20C_INPUT_MANIFEST.csv")
    rows = []
    expected = {"DouBanLv2_0039": "DouBanLv2_0039_01", "DouBanLv2_0040": "DouBanLv2_0040_00",
                "DouBanLv2_0041": "DouBanLv2_0041_03"}
    for frame in manifest:
        stem = frame["stem"]
        if stem not in expected:
            continue
        ground_path = local_path(frame["gt_path"])
        ground = read_mask(ground_path)
        for method, folder in (("R0", "V0"), ("R1", "V1")):
            prediction_path = FIX / "runs" / folder / "GT10/选择后掩膜" / f"mask_{stem}.png"
            prediction = read_mask(prediction_path)
            score, iou = overlap(ground, prediction)
            matching = [path for path in local_path(frame["candidate_alias_dir"]).glob(f"mask_{stem}_*.png")
                        if (read_mask(path) == prediction).all()]
            if len(matching) != 1:
                raise ValueError("GT10 regression selected identity is not unique")
            candidate_id = matching[0].stem.removeprefix("mask_")
            if method == "R1" and (candidate_id != expected[stem] or score < 0.98):
                raise ValueError("Historical three-fix development regression failed")
            rows.append({"frame_id": stem, "dataset": "GT10_DEVELOPMENT_ONLY_NOT_TEST32", "method": method,
                         "candidate_id": candidate_id, "f1": score, "iou_fg": iou,
                         "gt_sha256": sha256(ground_path), "prediction_sha256": sha256(prediction_path),
                         "candidate_sha256": sha256(matching[0]), "source": str(prediction_path),
                         "kind": "RECOMPUTATION_OF_EXISTING_DEV_RUNS_NOT_NEW_INDEPENDENT_VALIDATION"})
    if len(rows) != 6:
        raise ValueError("Missing GT10 three-frame regression evidence")
    write_csv(WORK / "GT10_DEVELOPMENT_REGRESSION.csv", rows)
    print("GT10 three historical repairs numerically reproduced; kept outsideTEST32", flush=True)


if __name__ == "__main__":
    main()
