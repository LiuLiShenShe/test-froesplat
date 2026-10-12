from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np

from phase20e_common import DATA, WORK, binary_mask, frozen_parameters, namespace, production, read_csv, write_csv


def run(sample):
    prod = production()
    parameters = frozen_parameters()
    scores = read_csv(WORK / f"reports/{sample}_ALL_SCORES.csv")
    source = read_csv(WORK / f"reports/{sample}_FULL_CANDIDATES.csv")
    rows = []
    for frame in sorted({row["frame_id"] for row in scores}):
        candidates = [prod.Candidate(prompt_id="P6", prompt_text=parameters["prompt_texts"]["P6"],
                                    mask=binary_mask(row["mask_path"]), scores=[], raw_detection_count=1,
                                    instance_id=int(row["instance_id"]), candidate_order=int(row["candidate_order"]))
                      for row in source if row["frame_id"] == frame]
        candidates.sort(key=lambda candidate: candidate.candidate_order)
        for policy in ("R0", "R1"):
            records = [SimpleNamespace(prompt_id="P6", instance_id=int(row["instance_id"]), total_score=float(row["total_score"]),
                                       primary_score=float(row["primary_score"]), tie_break_score=float(row["tie_break_score"]),
                                       empty_flag=row["empty_flag"] == "True")
                       for row in scores if row["frame_id"] == frame and row["policy"] == policy]
            args = namespace({**parameters, "ranking_policy": "base" if policy == "R0" else "r1"})
            actual = prod.select_mask(DATA / sample / "inputs" / f"{frame}.jpg", candidates, records, parameters["prompt_texts"], args)[0]
            if not np.array_equal(actual, binary_mask(DATA / sample / policy / f"mask_{frame}.png")):
                raise ValueError(f"Actual production selection mismatch {frame}/{policy}")
            rows.append({"frame_id": frame, "policy": policy, "actual_production_select_mask_agrees": True})
    write_csv(WORK / f"reports/{sample}_REAL_PRODUCTION_SELECTION_REPLAY.csv", rows)
    print("REAL_PRODUCTION_SELECTION", sample, len(rows), "PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    run(parser.parse_args().sample)
