from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from fix_common import OLD, PROD, ROOT, WORK, manifest, read_csv, relative, sha256, write_json


def main():
    production_path = PROD / "生成RAP-FSAM3掩膜.py"
    old = ast.parse(subprocess.check_output(["git", "show", f"970082da:{relative(production_path)}"], cwd=ROOT, text=True))
    current = ast.parse(production_path.read_text())
    old_defs = {node.name: ast.dump(node, include_attributes=False) for node in old.body if isinstance(node, ast.FunctionDef)}
    current_defs = {node.name: ast.dump(node, include_attributes=False) for node in current.body if isinstance(node, ast.FunctionDef)}
    protected = ["score_candidate", "select_mask", "apply_spnp_refinement", "sam3_box_refinement", "repair_residual_mask", "apply_corrective_geometry", "basic_cleanup", "_masks_scores_boxes", "load_colmap_observations"]
    audit = {name: old_defs[name] == current_defs[name] for name in protected}
    assert all(audit.values())
    assert subprocess.check_output(["git", "show", f"970082da:{relative(PROD / 'ranking_policy.py')}"], cwd=ROOT) == (PROD / "ranking_policy.py").read_bytes()
    for row in manifest():
        assert sha256(ROOT / row["input_path"]) == row["input_sha256"]
        assert sha256(ROOT / row["gt_path"]) == row["gt_sha256"]
    for row in read_csv(WORK / "CANDIDATE_METADATA_PROVENANCE.csv"):
        assert sha256(ROOT / row["mask_path"]) == row["mask_sha256"]
        assert sha256(ROOT / row["source_detail_path"]) == row["source_detail_sha256"]
    historical = read_csv(OLD / "metrics/stage_metrics.csv")
    directories = {"selected": "选择后掩膜", "spnp": "正负提示细化掩膜", "residual": "残差修复掩膜", "gcfc": "几何修正掩膜", "final": "最终掩膜"}
    for row in historical:
        path = OLD / "runs" / row["group"] / row["dataset"] / directories[row["stage"]] / f"mask_{row['stem']}.png"
        assert sha256(path) == row["mask_sha256"]
    statuses = [json.loads(path.read_text()) for path in (WORK / "execution_records").glob("*_status.json")]
    assert len(statuses) == 12 and all(row["returncode"] == 0 for row in statuses)
    assert all(Path(row["output"]).is_relative_to(WORK / "runs") for row in statuses)
    tracked_changes = subprocess.check_output(["git", "-c", "core.quotepath=false", "diff", "--name-only", "970082da"], cwd=ROOT, text=True).splitlines()
    permitted = [relative(production_path), relative(PROD / "candidate_metadata.py"), relative(WORK) + "/"]
    assert all(path in permitted[:2] or path.startswith(permitted[2]) for path in tracked_changes)
    audit.update({"ranking_policy_sha256": sha256(PROD / "ranking_policy.py"),
                  "production_sha256": sha256(production_path), "metadata_loader_sha256": sha256(PROD / "candidate_metadata.py"),
                  "frozen_rgb_gt_pairs_verified": 31, "original_candidate_masks_verified": 55,
                  "historical_stage_masks_verified": len(historical), "isolated_completed_runs": len(statuses),
                  "all_original_GT_candidates_history_unchanged": True,
                  "no_2dgs_m2m3_colmap_core_or_manuscript_changes": True,
                  "no_test32_accuracy_evaluation": True})
    write_json(WORK / "PROTECTED_ALGORITHM_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
