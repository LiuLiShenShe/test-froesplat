from __future__ import annotations

import math
from collections import defaultdict

from fix_common import WORK, read_csv, write_csv, write_json


def sigmoid(value):
    return 1 / (1 + math.exp(-6 * (value - 0.5)))


def main():
    grouped = defaultdict(list)
    for row in read_csv(WORK / "CANDIDATE_SCORE_REPLAY.csv"):
        grouped[(row["dataset"], row["sample"], row["frame"])].append(row)
    rows = []
    for (dataset, sample, frame), candidates in grouped.items():
        for policy in ("R0", "R1"):
            field = f"restored_{policy}_primary"
            chosen = max(candidates, key=lambda row: float(row[field]))
            confidence = float(chosen["sam_score"])
            lower = float(chosen[field]) + (sigmoid(max(0, confidence - 5e-7)) - sigmoid(confidence)) * 0.5 / 5.5
            other_bounds = []
            for candidate in candidates:
                if candidate is chosen:
                    continue
                value = float(candidate["sam_score"])
                upper = float(candidate[field]) + (sigmoid(min(1, value + 5e-7)) - sigmoid(value)) * 0.5 / 5.5
                other_bounds.append(upper)
            gap = lower - max(other_bounds) if other_bounds else None
            rows.append({"dataset": dataset, "sample": sample, "frame": frame, "policy": policy,
                         "selected_candidate_id": chosen["candidate_id"], "worst_case_winner_lower": lower,
                         "worst_case_best_other_upper": max(other_bounds) if other_bounds else "",
                         "worst_case_margin": gap if gap is not None else "SINGLE_CANDIDATE",
                         "certificate": "SELECTION_EQUIVALENT_FOR_ALL_SERIALIZATION_CONSISTENT_SCORES" if gap is None or gap > 0 else "NOT_VERIFIED",
                         "original_confidence_inferred": False})
    write_csv(WORK / "ROUNDING_SELECTION_CERTIFICATE.csv", rows)
    write_json(WORK / "ROUNDING_CERTIFICATE_AUDIT.json", {
        "pairs": len(rows), "certified": sum(row["certificate"].startswith("SELECTION_EQUIVALENT") for row in rows),
        "minimum_worst_case_margin": min(row["worst_case_margin"] for row in rows if isinstance(row["worst_case_margin"], float)),
        "confidence_rounding_interval_half_width": 5e-7, "formula_and_weights_changed": False,
        "scope": "Selection exactness certificate, NOT unrounded score reconstruction"})
    if any(row["certificate"] == "NOT_VERIFIED" for row in rows):
        raise ValueError("Some original-precision selections remain unverified")


if __name__ == "__main__":
    main()
