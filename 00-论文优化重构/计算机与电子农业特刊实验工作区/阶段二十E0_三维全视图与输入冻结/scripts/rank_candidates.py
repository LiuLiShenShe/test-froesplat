from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np
from PIL import Image

from phase20e_common import DATA, WORK, frozen_parameters, namespace, pixel_hash, production, save_mask, scene_rows, sha256, view_key, write_csv, write_json
from generate_candidates import verify_cache


def run(sample):
    prod = production()
    parameters = frozen_parameters()
    args_base = namespace({**parameters, "ranking_policy": "base"})
    args_r1 = namespace({**parameters, "ranking_policy": "r1"})
    if {key: value for key, value in vars(args_base).items() if key != "ranking_policy"} != {key: value for key, value in vars(args_r1).items() if key != "ranking_policy"}:
        raise ValueError("R0/R1 configuration differs beyond ranking policy")
    weights = prod.parse_key_values(parameters["score_weights"], {})
    selection, manifest, score_rows = [], [], []
    import ranking_policy
    for index, row in enumerate(scene_rows(sample)):
        key = view_key(row)
        candidates, meta = verify_cache(row, prod, parameters)
        rgb = DATA / sample / "inputs" / f"{key}.jpg"
        with Image.open(rgb) as image:
            image = image.convert("RGB")
            base_scores = [prod.score_candidate(rgb, image, candidate, {}, weights, args_base) for candidate in candidates]
            r1_scores = [prod.score_candidate(rgb, image, candidate, {}, weights, args_r1) for candidate in candidates]
        best_base = ranking_policy.select_best_policy(base_scores, "base")
        best_r1 = ranking_policy.select_best_policy(r1_scores, "r1")
        selected = {}
        for policy, scores, best in [("R0", base_scores, best_base), ("R1", r1_scores, best_r1)]:
            candidate = candidates[scores.index(best)]
            selected_by_production = prod.select_mask(rgb, candidates, scores, parameters["prompt_texts"], args_base if policy == "R0" else args_r1)[0]
            if not np.array_equal(candidate.mask, selected_by_production):
                raise ValueError("Pure policy selector disagrees with unchanged production selection path")
            target = DATA / sample / policy / f"mask_{key}.png"
            save_mask(target, candidate.mask)
            selected[policy] = candidate
            manifest.append({"sample": sample, "frame_id": key, "policy": policy, "registered_image_id": row["registered_image_id"],
                             "candidate_id": f"{key}_{candidate.instance_id:02d}", "mask_path": str(target),
                             "mask_sha256": sha256(target), "pixel_sha256": pixel_hash(candidate.mask.astype(np.uint8) * 255),
                             "candidate_sha256": candidate.mask_sha256, "rgb_sha256": row["raw_rgb_sha256"],
                             "width": candidate.mask.shape[1], "height": candidate.mask.shape[0], "empty_mask": not candidate.mask.any()})
            for candidate, score in zip(candidates, scores, strict=True):
                score_rows.append({"sample": sample, "frame_id": key, "policy": policy,
                                   "candidate_id": f"{key}_{candidate.instance_id:02d}", "sam_score_full_precision": candidate.sam_score,
                                   "box_json": json.dumps(candidate.box), "mask_sha256": candidate.mask_sha256,
                                   "candidate_order": candidate.candidate_order, **dataclasses.asdict(score)})
        base, r1 = selected["R0"], selected["R1"]
        changed = base.mask != r1.mask
        selection.append({"sample": sample, "frame_id": key, "registered_image_id": row["registered_image_id"],
                          "registered_name": row["registered_name"], "split": row["split"],
                          "R0_candidate_id": f"{key}_{base.instance_id:02d}", "R1_candidate_id": f"{key}_{r1.instance_id:02d}",
                          "switched": base.instance_id != r1.instance_id, "pixelwise_identity": not changed.any(),
                          "changed_pixel_ratio": float(changed.mean()), "R0_mask_sha256": sha256(DATA / sample / "R0" / f"mask_{key}.png"),
                          "R1_mask_sha256": sha256(DATA / sample / "R1" / f"mask_{key}.png"),
                          "R0_raw_score": best_base.total_score, "R1_raw_score": best_r1.total_score,
                          "R0_ranking_score": best_base.primary_score, "R1_ranking_score": best_r1.primary_score,
                          "R0_risk_penalty": best_base.leak_penalty, "R1_risk_penalty": best_r1.leak_penalty,
                          "candidate_count": len(candidates), "empty_candidate_count": sum(not candidate.mask.any() for candidate in candidates),
                          "R0_empty": not base.mask.any(), "R1_empty": not r1.mask.any(),
                          "fallback": any(entry.get("empty_fallback", False) for entry in meta["candidates"]),
                          "missing": False, "set_type": meta["cache_set_type"]})
        if index % 20 == 0:
            print("RANK", sample, index + 1, "switch", selection[-1]["switched"], flush=True)
    write_csv(WORK / f"reports/{sample}_SELECTION.csv", selection)
    write_csv(WORK / f"reports/{sample}_MASKS.csv", manifest)
    write_csv(WORK / f"reports/{sample}_ALL_SCORES.csv", score_rows)
    write_json(WORK / f"reports/{sample}_RANKING_PARAMETERS.json", {"R0": {**parameters, "ranking_policy": "base"}, "R1": parameters})
    print("RANK_DONE", sample, "views", len(selection), "switches", sum(row["switched"] for row in selection), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    run(parser.parse_args().sample)
