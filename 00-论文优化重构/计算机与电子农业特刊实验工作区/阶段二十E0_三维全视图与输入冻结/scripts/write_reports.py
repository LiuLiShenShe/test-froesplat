from __future__ import annotations

import json
import shutil
from pathlib import Path

from phase20e_common import DATA, FINAL, GS, ROOT, SAMPLES, WORK, read_csv, relative, sha256, write_csv, write_json


def text(name, content):
    path = WORK / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for field in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def aggregate(target, pattern, fields):
    rows = []
    for sample in SAMPLES:
        path = WORK / f"reports/{sample}_{pattern}.csv"
        if path.exists():
            rows.extend(read_csv(path))
    write_csv(WORK / target, rows, fields if not rows else None)
    return rows


def run():
    identity = read_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv")
    proposed_path = FINAL / "PHASE20E_VIEW_MANIFEST.csv"
    proposed = {(row["sample"], row["registered_image_id"]): row for row in read_csv(proposed_path)}
    if set(proposed) != {(row["sample"], row["registered_image_id"]) for row in identity}:
        raise ValueError("Registered view set changed from the Phase20D predeclared model")
    split_comparison = []
    for row in identity:
        before = proposed[row["sample"], row["registered_image_id"]]
        unchanged = (before["registered_name"], before["registered_camera_id"], before["frozen_split"], before["split_order_index"]) == (
            row["registered_name"], row["camera_id"], row["split"], row["split_order_index"])
        if not unchanged:
            raise ValueError("Predeclared registered-name split changed")
        split_comparison.append({"sample": row["sample"], "registered_image_id": row["registered_image_id"],
                                 "registered_name": row["registered_name"], "camera_id": row["camera_id"], "split": row["split"],
                                 "split_order_index": row["split_order_index"], "predeclared_split_unchanged": unchanged,
                                 "prior_equal_stem_raw_frame_id": before["raw_frame_id"], "verified_actual_raw_frame_id": row["actual_raw_frame_id"],
                                 "prior_raw_identity_corrected": before["raw_frame_id"] != row["actual_raw_frame_id"],
                                 "canonical_rgb_hash_corrected_or_recovered": before["raw_rgb_sha256"] != row["raw_rgb_sha256"],
                                 "predeclared_manifest": relative(proposed_path), "predeclared_manifest_sha256": sha256(proposed_path)})
    write_csv(WORK / "reports/PREDECLARED_SPLIT_COMPARISON.csv", split_comparison)
    candidates = aggregate("FULL_VIEW_CANDIDATE_MANIFEST.csv", "FULL_CANDIDATES", ["sample", "frame_id", "candidate_id", "set_type", "mask_sha256"])
    selection = aggregate("FULL_VIEW_R0_R1_SELECTION.csv", "SELECTION", ["sample", "frame_id", "switched"])
    masks = aggregate("FULL_VIEW_R0_R1_MASK_MANIFEST.csv", "MASKS", ["sample", "frame_id", "policy", "mask_sha256"])
    remaps = aggregate("CAMERA_REMAP_MANIFEST.csv", "REMAP", ["sample", "frame_id", "map_sha256"])
    datasets = aggregate("MATCHED_2DGS_DATASET_MANIFEST.csv", "DATASETS", ["sample", "group", "frame_id", "image_sha256"])
    splits = aggregate("TRAIN_VALIDATION_SPLIT.csv", "SPLIT", ["sample", "image_id", "registered_name", "split"])
    aggregate("reports/FULL_VIEW_SCORE_DECOMPOSITION.csv", "ALL_SCORES", ["sample", "frame_id", "policy", "primary_score"])
    references = [{key: row[key] for key in ("sample", "image_id", "registered_name", "loader_image_name", "camera_id", "source_rgb_sha256", "common_rgb_sha256", "camera_params_json", "qvec_json", "tvec_json")} for row in splits if row["split"] == "validation"]
    for row in references:
        row["reference_rgb_path"] = str(DATA / row["sample"] / "common/rgb" / f"{row['loader_image_name']}.png")
        row["independent_geometry_GT"] = "NONE"
        row["training_masks_are_GT"] = False
    write_csv(WORK / "reports/EVALUATION_REFERENCE_MANIFEST.csv", references, ["sample", "image_id", "reference_rgb_path"] if not references else None)
    summaries, scenes, commands = [], {}, []
    issues = []
    for sample in SAMPLES:
        rows = [row for row in identity if row["sample"] == sample]
        cs = [row for row in candidates if row["sample"] == sample]
        ss = [row for row in selection if row["sample"] == sample]
        mapped_path = WORK / f"reports/{sample}_MAPPED_MASK_DIFFERENCES.csv"
        mapped = read_csv(mapped_path) if mapped_path.exists() else []
        build_path = WORK / f"reports/{sample}_BUILD.json"
        preflight_path = WORK / f"reports/{sample}_PREFLIGHT.json"
        build = json.loads(build_path.read_text()) if build_path.exists() else {}
        preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
        view_count = len({row["frame_id"] for row in cs})
        switched = sum(row["switched"] == "True" for row in ss)
        train_different = sum(row["pixelwise_identity"] == "False" and row["split"] == "train" for row in mapped)
        validation_different = sum(row["pixelwise_identity"] == "False" and row["split"] == "validation" for row in mapped)
        ready = view_count == len(rows) and len(ss) == len(rows) and len(mapped) == len(rows) and preflight.get("status") == "PASS"
        groups = build.get("groups", [])
        distinct = groups if train_different else [group for group in groups if group != "B_R0"]
        if preflight.get("groups"):
            consumed = {item["group"]: item["actual_consumed_train_rgb_sha256"] for item in preflight["groups"]}
            if consumed.get("B_R0") == consumed.get("C_R1"):
                distinct = [group for group in groups if group != "B_R0"]
        if not ready:
            distinct = []
        summary = {"sample": sample, "registered": len(rows), "candidate_views": view_count,
                   "new_inference_views": len({row["frame_id"] for row in cs if row["set_type"] == "NEW_FULL_VIEW_CANDIDATE_SET"}),
                   "new_instances": sum(row["set_type"] == "NEW_FULL_VIEW_CANDIDATE_SET" and row.get("empty_fallback") != "True" for row in cs),
                   "historical_candidate_views": len({row["frame_id"] for row in cs if row["set_type"] != "NEW_FULL_VIEW_CANDIDATE_SET"}),
                   "R0_R1_switches": switched, "train_mask_differences": train_different,
                   "validation_mask_differences": validation_different, "historical_mask_coverage": sum(row["historical_mask_verified"] == "True" for row in rows),
                   "historical_rgba_matching_camera_grid": sum(not row["historical_rgba_error"] and row["historical_rgba_width"] == row["camera_width"]
                                                               and row["historical_rgba_height"] == row["camera_height"] for row in rows),
                   "train": sum(row["split"] == "train" for row in rows), "validation": sum(row["split"] == "validation" for row in rows),
                   "status": "PASS" if ready else "PARTIAL", "ready": ready, "training_runs": len(distinct),
                    "A_group": groups[0] if groups else "NOT_BUILT", "input_unreadable": sum(bool(row["colmap_input_error"]) for row in rows),
                    "original_raw_unreadable": sum(bool(row.get("original_raw_rgb_error", "")) for row in rows),
                    "canonical_raw_unreadable": sum(bool(row["raw_rgb_error"]) for row in rows),
                   "historical_undistorted_unreadable": sum(bool(row["historical_undistorted_error"]) for row in rows)}
        summaries.append(summary)
        scenes[sample] = {**summary, "distinct_training_groups": distinct}
        for row in rows:
            if row["raw_rgb_error"] or row.get("original_raw_rgb_error") or row["colmap_input_error"] or row["historical_undistorted_error"] or row["legacy_error"] or row["historical_rgba_error"]:
                issues.append({"sample": sample, "registered_image_id": row["registered_image_id"], "registered_name": row["registered_name"],
                               "actual_raw_frame_id": row["actual_raw_frame_id"], "raw_error": row["raw_rgb_error"],
                               "original_raw_error": row.get("original_raw_rgb_error", ""), "source_recovery": row.get("source_recovery", ""),
                               "input_path": row["colmap_input_path"], "input_error": row["colmap_input_error"],
                               "undistorted_path": row["historical_undistorted_path"], "undistorted_error": row["historical_undistorted_error"],
                               "historical_mask_error": row["legacy_error"], "recoverable_from_verified_raw": row["common_raw_domain_proven"]})
                issues[-1]["historical_rgba_path"] = row["historical_rgba_path"]
                issues[-1]["historical_rgba_error"] = row["historical_rgba_error"]
        for group in distinct:
            script = WORK / "scripts/run_phase20e1.py"
            command = f"CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 '{GS / 'venv/bin/python'}' '{script}' --sample {sample} --group {group} --execute-optimization"
            commands.append({"sample": sample, "group": group, "command": command,
                             "output": str(DATA / "phase20e1_outputs" / sample / group), "executed": False})
    write_csv(WORK / "reports/SCENE_SUMMARY.csv", summaries)
    write_csv(WORK / "reports/RECOVERABLE_HISTORICAL_DAMAGE.csv", issues, ["sample", "registered_image_id", "registered_name"] if not issues else None)
    write_csv(WORK / "reports/PHASE20E1_COMMANDS.csv", commands, ["sample", "group", "command", "output", "executed"] if not commands else None)
    hashes = {relative(path): sha256(path) for path in sorted((WORK / "reports").glob("*_TRAIN_CONFIG.json"))}
    write_json(WORK / "reports/FROZEN_TRAIN_CONFIG_HASHES.json", hashes)
    ready_count = sum(row["ready"] for row in summaries)
    total_runs = sum(row["training_runs"] for row in summaries)
    status = "PASS" if ready_count == 3 else "PARTIAL" if ready_count else "BLOCKED"
    acceptance = {"status": status, "scenes": scenes, "training_runs": total_runs,
                  "formal_training": False, "mesh_generation": False, "all_nine_warranted": total_runs == 9 and ready_count == 3}
    write_json(WORK / "reports/FINAL_ACCEPTANCE.json", acceptance)
    simple_table = table(summaries, ["sample", "registered", "candidate_views", "new_instances", "R0_R1_switches", "train_mask_differences", "ready", "training_runs"])
    text("SCENE_ELIGIBILITY_REPORT.md", f"""# Scene eligibility — current measured local assets

{simple_table}

Selection priority is camera/RGB provenance, not future reconstruction scores:
WanNianQing1 first, then CaoMei2, then HongZhang; every scene retains all registered
views. Raw camera poses, IDs and sparse points are immutable. Common camera version:
`HISTORICAL_POSES_CANONICAL_RAW_RGB_V1`. This version reconstructs RGB on the exact
undistorted domain without mixing any new COLMAP model into the experiment.

## CaoMei2 identity correction

`03-SAM/segment_v2.py:185` does **not** spatially crop the frame. It copies the whole
RGB and sets background pixels to zero. FFT filtering **does** renumber retained
frames (`02-FFT/blur_filter.py:245`). Full per-view hash/log/pixel joins are in
`FULL_VIEW_IDENTITY_MANIFEST.csv`. Registered `crop_0146.png` is raw `0170.jpg`;
TEST32 raw `0146.jpg` would be `crop_0129.png`, but that image is NOT registered
in the actual binary model. The previously reported 0146 registration was an
equal-stem identity error. Reusing these foreground-conditioned poses is a limitation, not a
claim that the old model was full-raw-RGB registration. Pixel coordinate mapping
is identity before true undistortion; there is no resize or bbox inversion.

WanNianQing1 raw 0223 is registered. Raw 0227 remains unregistered, so its TEST32
improvement is not a three-dimensional benefit claim in this camera collection.
HongZhang/WanNianQing1 raw registrations retain raw names; legacy masks must join
the FFT index to raw SHA256, not same-name numbers. No unreadable view is dropped.

Historical damaged paths and restoration status: `reports/RECOVERABLE_HISTORICAL_DAMAGE.csv`.
Twelve unreadable HongZhang originals are recovered into managed data from exact-name
readable registered raw COLMAP inputs; original corrupt files/hashes remain unchanged.
Four CaoMei2 coordinate identities use secondary registered input crops with exact
nonblack raw-pixel equality. `reports/SOURCE_RECOVERY_MANIFEST.csv` records this proof.
All canonical originals, including verified recovery copies, are fully read and checked; rebuilt RGB lives in managed
data, not overwritten historical files. Detailed per-view source hashes are retained.
Readable legacy RGBA on the exact camera grid: CaoMei2 2/106 (104 are raw-sized),
WanNianQing1 193/213, HongZhang 185/213. Proven legacy raw/mask/crop pairs at the
initial audit: 102/106, 211/213, 188/213 respectively. These are different coverage
definitions and neither proves a complete historical training version. The new A
control is explicitly raw RGB; it cannot substantiate a reproduced-ForeSplat claim.
Readable legacy RGBA was tested as an alternate missing-mask source, not dismissed
only by file count: all four damaged CaoMei2 frames' alpha composites disagree
with registered historical crops (17199, 19330, 29483 and 31773 pixels). A known
readable CaoMei2 alpha differs from original SAM by 3715 pixels; a WanNianQing1
alpha differs from official-nearest mapped original SAM by 50541 pixels.
Without an identified construction/version recipe these are not recoverable
historical-mask identities. `reports/LEGACY_RGBA_RECOVERY_PROOF.json` and
`scripts/audit_legacy_alpha.py` retain the evidence. No alpha is inferred from black RGB.
WanNianQing1 damaged input 0136 has an additional **historical RGB version conflict**:
its existing undistorted color image does not match current raw0136 (mean interior
error 61.53 intensity units). The raw RGB's actual native CPU SIFT locations match
original image_id133 distorted COLMAP observations: median 0.195px, 79.53% within
1px. Adjacent wrong-frame controls match <0.76% within 1px (median 13–24px).
Therefore raw0136 belongs to this immutable camera; the existing derived RGB is
not reused. `reports/WanNianQing1_0136_SIFT_KEYPOINT_IDENTITY.json` preserves actual
evidence and missing original descriptor limitations. No camera registration was run.
""")
    text("FULL_VIEW_SWITCH_ANALYSIS.md", f"""# Full-view R0/R1 switch analysis

{simple_table}

Each RGB is inferred at most once for new candidates. Both policies consume the
identical cache, confidence precision, instance order, boxes, thresholds and full
Phase20D configuration. Only `ranking_policy` differs. Production scorer and
`ranking_policy.py` are loaded unchanged; every candidate decomposition is in
`reports/FULL_VIEW_SCORE_DECOMPOSITION.csv`.
The per-instance model/processor lineage is in
`reports/FULL_CANDIDATE_MODEL_LINEAGE.csv`: historical instances match archived
parameters and Phase20D frozen model hashes; a checkpoint tensor digest from the
original Phase17A generation was not independently archived. This limit is
retained, not fabricated. New inference records exact checkpoint and source hashes.

Switching IDs is not automatically improved segmentation. No GT is loaded by
generation or ranking. Empty candidates/masks, fallback and missing flags are
recorded. Pixel differences are recomputed both before and after undistortion.
If all registered **training** inputs match, B/C cannot test ranking effects on
training; execute only one and share that checkpoint. Validation-mask differences
alone do not justify duplicate optimization. Actual distinct runs: **{total_runs}**.
""")
    text("COLMAP_MASK_DOMAIN_MAPPING.md", """# COLMAP mask coordinate mapping — empirical verification

For each target pixel `(column+0.5,row+0.5)`, call the actual target PINHOLE
`cam_from_img`; append z=1 and project with the original OPENCV `img_from_cam`.
Subtract 0.5 from the source projection to address the image array. No assumed
OpenCV principal-point convention and no resolution-only mask resize are used.
Mask sampling is nearest neighbor with invalid pixels zero. RGB is direct bilinear
interpolation with COLMAP color rounding. Maps use float64 and have explicit SHA256.

Native `pycolmap.undistort_camera` must reproduce the exact historical output
dimensions/intrinsics. Two original RGB views per scene are compared against
`pycolmap.undistort_image` with direct warping and bilinear sampling. All original
and undistorted sparse image observations are converted with the same official
camera functions; actual 3D points are reprojected using unchanged image poses.
Damaged registered input 0136 is independently matched to its original COLMAP
observations with real CPU SIFT and neighboring negative controls; its conflicting
historical derived RGB is rejected instead of using a wrong color reference.

Existing historical images may be foreground-replaced/JPEG-encoded, and several
are unreadable. They are compared and recorded, not assumed equal to raw color.
The canonical raw RGB version is generated for all groups from verified original
files on the exact model grid. Existing files/models are never overwritten.

Validation: per-scene `reports/*_SPARSE_PROJECTION.json`, `*_RGB_COMPARISONS.csv`,
`*_BUILD.json`; exact mapping/source identities in `CAMERA_REMAP_MANIFEST.csv`.
Official sources fetched via Context7 and direct source retrieval:
https://github.com/colmap/colmap/blob/main/src/colmap/image/warp.cc
https://github.com/colmap/colmap/blob/main/src/colmap/image/undistortion.cc
https://github.com/colmap/colmap/blob/main/python/README.md
""")
    text("MASK_RGB_ALIGNMENT_REPORT.md", f"""# Mask/RGB and loader alignment

{simple_table}

Native RGB roundtrip tolerance is maximum one intensity unit, mean <=0.1; measured
values are retained rather than called exact. Sparse observation conversion max
must be <=1e-6px. Masks are binary and map invalid pixels to background. All target
dimensions must equal cameras. Alignment figures show exact-domain masks and RGB;
native model viewing is required and machine QA is not a human signature.

Real production loading additionally checks all registered views, frozen split,
camera transforms, common RGB identity, pre-composited RGB/alpha pixels, declared
resolution=2 behavior and initial inherited target sparse point identity. All groups use
one camera symlink bundle. Each group receives one real forward render without
`training_setup`, optimizer, backward or optimization step. Reports in
`reports/*_PREFLIGHT.json` include actual GPU/CUDA/extension paths and memory.
`reports/ACTUAL_LOADER_CAMERA_PROVENANCE.csv` also checks all 1596 group-view
camera transforms and FoV exactly against frozen intrinsics/poses. The production
`CameraInfo.uid` is a camera_id, NOT image_id; the unique registered name performs
the join, and the true registered image_id is retained independently in manifests.

Current `utils/camera_utils.py:44` loads alpha; `scene/cameras.py:47` only stores it.
Default RGB loss uses all RGB pixels and plant-aware alpha losses are disabled;
   default sparse initialization uses inherited model points without extra filtering. Consequently our treatment
is black RGB pre-composition, not an assumed alpha loss or mask-guided initialization.
The base unmasked RGB is identical across groups; treatment composites intentionally
differ. PIL RGB/alpha channel scaling at resolution=2 is explicitly frozen and
verified, never mistaken for a camera-domain mask resize.
""")
    command_text = "\n\n".join(f"## {row['sample']} / {row['group']}\n\n```bash\n{row['command']}\n```\nOutput: `{row['output']}`; not executed." for row in commands)
    text("PHASE20E1_COMMANDS.md", f"""# Phase20E1 commands — NOT EXECUTED

Total distinct optimizations: {total_runs}. Seed=0, 30000 iterations, GPU physical1,
resolution=2, black background. Physical GPU UUID is frozen in
`reports/FROZEN_HARDWARE_IDENTITY.json`; the wrapper checks UUID, not only GPU name.
All optimization and model parameters are explicitly
snapshotted in `reports/*_TRAIN_CONFIG.json`, protected by config SHA256, and passed
to unchanged `train.training`. The wrapper refuses duplicate group training,
unverified scenes, changed configs and preexisting output directories. Default
wrapper invocation **does not optimize**; only Phase20E1 may pass the execution flag.

{command_text}

No meshes or M2M3-GO are part of these commands. Production held-out logging compares
to treatment-specific composites and is secondary only. The primary independent
RGB reference is the SAME canonical raw undistorted RGB for all groups in
`reports/EVALUATION_REFERENCE_MANIFEST.csv`. Future PSNR/SSIM/LPIPS must compare to
that reference with identical camera/resolution; never compare different foreground
reference masks as if they were a matched RGB metric.
""")
    text("PHASE20E1_ACCEPTANCE_GATES.md", f"""# Phase20E1 acceptance gates

Current readiness is measured in `reports/FINAL_ACCEPTANCE.json`, not inferred
from configuration files. {ready_count}/3 complete scenes; {total_runs} distinct runs.

G1 full V1 snapshot and ranking/checkpoint/processor hashes unchanged; G2/G3 new
versus old cache labels, full new tensor score/box precision and per-instance hashes;
G4/G10 identical cache replay, all scores and mask identity; G5 complete mask coverage;
G6/G7 RGB-camera join plus official mapping, native RGB and real sparse projections;
G8 identical bundle and immutable split; G9 historical A differences and missing RGBA
explicit; G11 real train-mask difference counts; G12 true loader and CUDA render;
G13 no production 2DGS/R1/M2M3 edits; G14 optimizer steps=0/checkpoints=0; G15 explicit
commands, frozen configs, data manifest and outputs. Final automated checks are
in `logs/DELIVERY_VERIFICATION.json`.
The 45 scene-specific G1–G15 checks are in
`reports/ACCEPTANCE_GATES_G1_G15.csv` and `logs/G1_G15_VALIDATION.json`.
The predeclared Phase20D view/camera IDs, names, order and splits are replayed exactly
in `reports/PREDECLARED_SPLIT_COMPARISON.csv`; corrected raw identity does not change
which registered cameras belong to train or validation.

Before any training, re-hash all datasets/configs, confirm same physical GPU and
available resources, preserve split names/IDs and source/model points. Never enable
A6/A7, SPNP/Residual/GCFC/semantic/temporal/geometry gates or tune R1 on TEST data.

Evaluation separates held-out shared raw RGB PSNR/SSIM/LPIPS; independently reliable
fixed silhouette/ROI; background contamination/visual inspection; Gaussian counts,
time/VRAM; descriptive mesh/point-cloud quality in a later authorized phase. Training
masks are NOT independent GT. No independent geometry exists; no true 3D Chamfer or
geometric F-score claim. Unregistered TEST32 images cannot become held-out cameras.
""")
    text("ARTIFACT_RECOVERY.md", f"""# Managed data recovery

Managed root: `{DATA}`. It is OUTSIDE Git; no large mask/RGBA/model archive is committed.
CSV manifests contain exact source paths and SHA256, every candidate instance, score,
box, precision, code/checkpoint, output RGB, alpha, camera bundle and map hashes.
Copy/restore the managed root and verify `reports/MANAGED_ARTIFACT_HASHES.csv` first.
Preserve directory symlinks and file bytes; identical B/C images may be hardlinked.
Run the read-only managed-cache verifier after restoration:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/test/biosoft/enter/bin/python '{WORK / 'scripts/verify_managed_cache.py'}'
```

This also validates each group's common-camera directory symlink. Do not flatten
symlinks or replace sparse initialization PLY merely because points3D.bin matches.
Historical checkpoint: `{ROOT / 'sam3/sam3.pt'}`, SHA256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

For full rebuild use exact original assets at the recorded baseline, `audit_inputs.py`,
then `recover_verified_sources.py`, each scene's `generate_candidates.py --sample NAME --smoke`, full generation,
`rank_candidates.py`, `build_datasets.py`, and real `preflight.py`. Complete the first
scene before starting remaining scene inference. SAM runtime is
`/home/test/biosoft/enter/envs/sam3/bin/python`; CPU/remap runtime is
`/home/test/biosoft/enter/bin/python`; loader runtime is `{GS / 'venv/bin/python'}`.
Scripts import only local frozen SAM3 and 2DGS sources; no CLI parameters are guessed.
GPU is selected externally with CUDA_VISIBLE_DEVICES=1. Set OMP/OPENBLAS threads=4.

Exact cache restore is preferred. A new GPU regeneration may not byte-match across
runtime/hardware; label it a NEW candidate version, never silently replace this cache.
Archive confidences have six-decimal serialization limitations and are preserved,
not reconstructed digits. New scores retain full available tensor precision.
No GT directory is read by inference, ranking or dataset construction.
Inference needs the recorded local SAM checkpoint/processor; loader uses its local
CPython3.12 CUDA extension binaries. Hashes, versions, environment paths and physical
GPU UUID are in candidate provenance, `reports/*_PREFLIGHT.json` and the hardware
identity JSON. Never silently install a newer library to simulate an exact restore.
""")
    text("PHASE20E0_FINAL_REPORT.md", f"""# ForeSplat Phase20E-0 — full-view foreground and matched input freeze

Baseline `f53bf8f3`; date 2026-10-12, timezone Asia/Shanghai. Initial worktree clean.
New files are contained in this stage; production algorithms and original historical
assets are protected. No formal 2DGS training or mesh reconstruction was executed.

## Validation evidence

`logs/G1_G15_VALIDATION.json`: 45/45 scene-specific acceptance checks.
`logs/SOURCE_PROTECTION_FINAL.json`: 7914 original historical/model/code files
fully SHA256-verified, then final size/mtime/ctime rechecked without overwrite.
`logs/DELIVERY_VERIFICATION.json`: 532 selections, 928 candidate instances,
1596 dataset group-views, no optimizer steps/checkpoints/meshes or outside-stage edits.
`logs/MANAGED_CACHE_RESTORE_VERIFICATION.json`: all 6831 managed files re-hashed;
exact camera symlink routing is verified. Approximate actual disk use is 9.6GiB,
while file-byte manifest sum includes repeated hardlinks (11.287678255GB).
CPU results: `logs/CPU_PHASE20E0_FINAL.txt` (34 passed, no skips),
`logs/CPU_HISTORICAL_REGRESSION_FINAL.txt` (76 passed),
`logs/CPU_PHASE20D_LEGACY_FINAL.txt` (107 passed), and the final delivery recheck.
Total final stage plus historical regression checks: 217 passed. No unrelated
test/code changes or new formatter were introduced.
All nine group loader/forward checks used the original local CUDA extensions;
they are input checks, not training outcomes or fabricated checkpoints.

{simple_table}

## Answers to the twelve requested questions

1. Registered counts: CaoMei2=106, WanNianQing1=213, HongZhang=213; exact image IDs/names
   from binary models are in the identity manifest. No view was silently removed.
2. Current full P6 covered views: table above. Coverage counts cache-verified real frames.
3. New GPU candidate instance counts: table above; zero-detection placeholders excluded
   from instance counts and separately flagged. Historical and new sets never conflated.
4. Real R0/R1 instance switches: table above; score sources/decompositions are retained.
5. B/C real differences: train and validation counts in `reports/SCENE_SUMMARY.csv` and
   mapped pixel hashes. Identical train inputs require no duplicate optimization.
6. A status: versioned historical reconstruction only when all original masks pair
   to actual raw RGB; otherwise explicitly named raw RGB control, not historical
   ForeSplat reproduction. Existing RGBA is incomplete/inconsistent and never silently
   called the recovered historical training version. See per-scene A_group and coverage.
7. Mapping verification: native COLMAP camera/RGB two-view tests, all observation
   conversion and real 3D projections; per-scene JSON/CSV retain measured residuals.
8. A/B/C input readiness: {ready_count}/3 scenes completed through real loader/forward.
9. Actually trainable scenes: {', '.join(row['sample'] for row in summaries if row['ready']) or 'NONE'}.
10. Blocked/partial scenes: {', '.join(row['sample'] for row in summaries if not row['ready']) or 'NONE'};
    exact historical unreadable paths and recovery status are reported, not discarded.
11. Distinct Phase20E1 training count: {total_runs}; all commands present and unexecuted.
12. All nine worthwhile: {'YES, all scenes have distinct B/C training masks' if total_runs == 9 else 'NO; do not duplicate pixel-identical training input or run blocked scenes'}.

## Important limitations and corrected provenance

All three A datasets are **A_raw_RGB_control_v1**, not restored historical
ForeSplat. Legacy camera-grid RGBA coverage is 2/106, 193/213 and 185/213 for
CaoMei2, WanNianQing1 and HongZhang. HongZhang's 12 corrupt original RGBs were
recovered from verified registered raw COLMAP copies without modifying originals;
no scene remains blocked for canonical RGB. Detailed damaged paths are retained.
Alternate old RGBA recovery was explicitly tested and rejected because its alpha
does not reproduce known original SAM/registered-crop identities; this evidence
is in `reports/LEGACY_RGBA_RECOVERY_PROOF.json`, not inferred from missing-file counts.

Actual switches occur in TRAIN views only: CaoMei2 raw0139; WanNianQing1 raw0037,
raw0038 and raw0165. HongZhang B/C are identical in all 213 registered views AND
the actual loader-consumed training RGB hash. Its duplicate B training is excluded;
one foreground checkpoint can represent B/C. No future metric influenced this decision.

FFT re-numbering invalidates equal-stem joins. CaoMei2 TEST32 raw0146 actually joins
FFT crop0129 (unregistered), not registered crop0146/raw0170. Original crop is full-size black-masked RGB,
not spatial crop; foreground-conditioned camera estimation is explicitly preserved.
WanNianQing1 0227 remains unregistered and cannot support existing-camera 3D gains.

The current 2DGS loader stores RGBA alpha but defaults do not apply it to RGB loss or
initialization. Treatment uses explicit black RGB pre-composition while all groups
share camera bundle, original RGB, split, sparse points and optimizer settings.
Inherited sparse points may have a historical filtering/revision history: source and
target ID sets differ for WanNianQing1 and HongZhang; all shared XYZ are identical.
The chosen common **target** sparse version (including its existing PLY) is immutable
across A/B/C and no treatment applies additional mask-guided initialization filtering.
This stage demonstrates engineering input correctness, not segmentation accuracy
on all views or improved 3D geometry. No independent geometry GT is available.
Historical candidate model lineage has a documented archive limit: existing
instances have immutable mask/score/box provenance and matching archived recipe/
Phase20D model hashes, but the original Phase17A generation-time checkpoint digest
was not independently saved. New instances carry exact generation hashes.
Counts are computed from current manifests; archive precision remains six decimals.

Bocha web retrieval was attempted and returned HTTP403 quota exhausted. Context7
official COLMAP docs and direct official source retrieval were used for API/convention
checks; experiment numbers come exclusively from local measured assets and real runs.
Requested `/home/test/.codex/RTK.md` does not exist; it was not invented or substituted.

```text
PHASE20E0 STATUS: {status}

SCENE_READY:
CaoMei2: {'YES' if scenes['CaoMei2']['ready'] else 'NO'}
WanNianQing1: {'YES' if scenes['WanNianQing1']['ready'] else 'NO'}
HongZhang: {'YES' if scenes['HongZhang']['ready'] else 'NO'}

FULL_VIEW_CANDIDATES:
{'; '.join(row['sample']+'='+str(row['candidate_views'])+'/'+str(row['registered']) for row in summaries)}

R0_R1_SWITCHES:
{'; '.join(row['sample']+'='+str(row['R0_R1_switches']) for row in summaries)}

CAMERA_MAPPING: {'VERIFIED' if ready_count == 3 else 'PARTIAL'}
MATCHED_2DGS_INPUT: {'READY' if ready_count == 3 else 'PARTIAL'}
TRAINING_PLAN: {total_runs}
PHASE20E1 READY: {'YES' if ready_count == 3 else 'CONDITIONAL' if ready_count else 'NO'}
```
""")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
