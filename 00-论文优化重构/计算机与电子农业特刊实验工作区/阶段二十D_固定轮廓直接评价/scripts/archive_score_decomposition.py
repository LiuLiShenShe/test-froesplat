from __future__ import annotations

import json
import math
import sys

import numpy as np

from direct_common import OLD, WORK, local_path, production_module, read_csv, read_mask, sha256, write_csv


def main():
    production_module()
    ranking = sys.modules["ranking_policy"]
    provenance = {row["candidate_id"]: row for row in read_csv(OLD / "TEST32_CANDIDATE_PROVENANCE.csv")}
    rows = []
    for policy in ("base", "r1"):
        output = WORK / "runs" / f"full_{policy}"
        parameters = json.loads((output / "参数.json").read_text())
        scores = read_csv(output / "提示词评分.csv")
        for stem in sorted({row["图像"].removesuffix(".jpg") for row in scores}):
            frame_scores = [row for row in scores if row["图像"] == stem + ".jpg"]
            details = read_csv(output / "候选掩膜" / f"候选评分明细_{stem}.csv")
            for score, detail in zip(frame_scores, details, strict=True):
                candidate_id = f"{stem}_{int(detail['实例编号']):02d}"
                source = provenance[candidate_id]
                sam_confidence = float(source["sam_score_archived_text"])
                mask = read_mask(local_path(source["mask_path"]))
                occupied_rows = np.flatnonzero(mask.any(axis=1))
                vertical_coverage = 1 - float(occupied_rows.min() / mask.shape[0]) if len(occupied_rows) else 0
                risk_penalty = 0.25 if vertical_coverage < parameters["vertical_coverage_min_ratio"] else 0.0
                q_sam = 1 / (1 + math.exp(-6 * (sam_confidence - 0.5)))
                weighted = {"area": float(score["面积得分"]), "comp": float(score["连通域得分"]),
                            "edge": float(score["边界得分"]), "temp": float(score["时序得分"]),
                            "contrast": float(score["前背景对比得分"]), "sam": 0.5 * q_sam}
                base_total = sum(weighted.values()) / 5.5
                primary, tie = ranking.primary_and_tiebreak(base_total, float(score["前背景对比得分"]),
                               float(score["面积得分"]), 1.0, 5.5, float(score["语义门控总修正"]), risk_penalty, policy)
                if abs(primary - float(score["主评分"])) > 1e-12 or abs(base_total - risk_penalty - float(score["总分"])) > 1e-12:
                    raise ValueError("Explicit score decomposition disagrees with real frozen scores")
                rows.append({"frame_id": stem, "policy": policy, "candidate_id": candidate_id,
                             "source_candidate_sha256": source["mask_sha256"], "source_detail_sha256": source["source_detail_sha256"],
                             "sam_score_archived": sam_confidence, "q_sam": q_sam, "weights_denominator": 5.5,
                             **{f"weighted_{key}": value for key, value in weighted.items()},
                             "base_total_before_risk": base_total, "vertical_coverage": vertical_coverage,
                             "risk_penalty": risk_penalty, "semantic_total": float(score["语义门控总修正"]),
                             "total_score_after_risk": float(score["总分"]), "primary_score": primary,
                             "tie_break_score": tie, "contrast_removed_from_primary": int(policy == "r1"),
                             "original_box": source["box"], "candidate_order": source["candidate_order"],
                             "full_config_sha256": sha256(output / "参数.json"), "source_scores_only_not_new_inference": True})
    if len(rows) != 142:
        raise ValueError("Full142score decomposition coverage required")
    write_csv(WORK / "FROZEN_SCORE_DECOMPOSITION.csv", rows)
    print("Verified explicit decomposition ofall71candidates underbothpolicies", flush=True)


if __name__ == "__main__":
    main()
