from __future__ import annotations

from common import E0, ROOT, STAGE, now, read_csv, sha256, write_csv, write_json


def main():
    previous = STAGE.parent / "阶段二十D_固定轮廓直接评价/RASTERIZED_MASK_MANIFEST.csv"
    masks = [row for row in read_csv(previous) if row["rasterizer"] == "opencv"]
    development_path = STAGE.parent / "01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计/GT10_manifest/Phase17A_GT10_manifest.csv"
    development_ids = {f"{row['sample']}_{row['frame']}" for row in read_csv(development_path)}
    identities = {(row["sample"], row["registered_name"]): row
                  for row in read_csv(E0 / "FULL_VIEW_IDENTITY_MANIFEST.csv")}
    rows = []
    for view in read_csv(E0 / "reports/EVALUATION_REFERENCE_MANIFEST.csv"):
        identity = identities[(view["sample"], view["registered_name"])]
        matching = [mask for mask in masks if mask["frame_id"] == identity["actual_raw_frame_id"]]
        assert len(matching) <= 1
        development_match = identity["actual_raw_frame_id"] in development_ids
        rows.append({"sample": view["sample"], "registered_name": view["registered_name"],
                     "actual_raw_frame_id": identity["actual_raw_frame_id"],
                     "registered_source_rgb_sha256": identity["original_raw_rgb_sha256"],
                     "manual_fixed_contour_available": bool(matching),
                     "GT10_development_only_match": development_match,
                     "matching_mask_sha256": matching[0]["mask_sha256"] if matching else "",
                     "source": str(previous.relative_to(ROOT))})
    write_csv(STAGE / "reports/HELDOUT_MANUAL_REFERENCE_AUDIT.csv", rows)
    result = {"timestamp": now(), "heldout_views": len(rows),
              "fixed_contour_matches": sum(row["manual_fixed_contour_available"] for row in rows),
              "GT10_development_matches": sum(row["GT10_development_only_match"] for row in rows),
              "source_sha256": sha256(previous), "GT10_manifest_sha256": sha256(development_path),
              "matching_uses": "verified actual raw identity, never equal filename stem"}
    write_json(STAGE / "reports/MANUAL_REFERENCE_AUDIT.json", result)
    print(result)
    assert result["fixed_contour_matches"] == 0, "Add verified contour coordinate mapping before ROI/IoU GT evaluation"
    assert result["GT10_development_matches"] == 0, "Development GT overlap must be explicitly distinguished from independent validation"


if __name__ == "__main__":
    main()
