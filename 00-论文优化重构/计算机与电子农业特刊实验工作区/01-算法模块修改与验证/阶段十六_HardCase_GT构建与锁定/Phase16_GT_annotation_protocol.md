# Phase 16 — Hard-case GT Annotation Protocol

**Date**: 2026-09-07
**Target frames**: 54 hard-case frames, 15 non-GT samples (7 DEV / 8 TEST)
**GT output**: `/data/fj/F2DMAS/03-GT-区分_challenge/<sample>/<frame>.json` (labelme format)
**Processing output**: `GT_potted_clean_challenge/<sample>/mask_potted_clean_<frame>.png`
**Canonical reference**: `处理GT_v2.py` (Phase 12 GT_v2 conventions, this doc inherits fully)

---

## §1. Objective

Annotate 54 hard-case frames (difficulty-sampled from 15 non-GT samples) as binary
ground truth for `potted_clean = potted_plant AND NOT blue_cube`, consistent with the
existing 21-frame easy set (Phase 12) so that all 75 frames use the same GT definition.

---

## §2. Label Schema

| Label | Meaning | Required? | Notes |
|---|---|---|---|
| `potted_plant` | **Target plant** (the specific individual in this sample) | **Required** | Defines the segmentation target |
| `plant` | Leaf + stem only (no pot) | **Recommended** | Enables formal_p2 analysis |
| `pot` | Flower pot / saucer / tray | **Recommended** | Enables pot/plant decomposition |
| `blue_cube` | Blue calibration cube (if present in scene) | **If present** | Removed from potted_clean via subtraction |

**Final GT**: `potted_clean = potted_plant` minus `blue_cube` overlap

---

## §3. Target Identity Rule (Critical for Hard Cases)

Hard-case frames contain multiple plants, occlusion, or background vegetation.
Use the following identity rules to determine which plant is the **target**:

1. **Sequence context is authoritative.** Each sample is a temporal sequence (0000.jpg → 0001.jpg → ...).
   The target is the *same individual plant* across all frames of the same sample.
2. **Annotate the target plant only**, even if neighbor plants are partially visible.
3. **Reference preceding/following frames** (±5 frames) to resolve ambiguity when occlusion
   or clipping makes identity uncertain in the target frame.
4. **The target is always the same individual** across all frames labeled under the same `<sample>`.
   When in doubt, check earlier frames of the same sample to confirm which plant is which.

---

## §4. Include vs Exclude

### Include in potted_plant:
- All visible plant tissue belonging to the target individual: leaves, stems, flowers, fruits
- Petioles connecting to the main stem
- Pot rim/saucer only if structurally connected to the plant (pot = separate `pot` label)

### Exclude from potted_plant:
- Neighboring plants (different individuals)
- Background vegetation
- Soil / potting medium (unless connected to plant tissue via stem)
- Support structures (stakes, wires, labels, clips)
- Shadows and reflections
- Greenhouse structures (bars, benches, walls)

### Pot:
- If labeled as `pot`, included in `potted_plant` (whole pot = potted_plant)
- The `potted_clean` formula handles pot separately

### Blue cube:
- If present: label `blue_cube` as a separate polygon
- `potted_clean` = `potted_plant` minus `blue_cube` (handled automatically by processing script)

---

## §5. Annotation Format (Labelme)

Every frame must produce exactly one JSON file:

```json
{
  "version": "0.4.43",
  "flags": {},
  "shapes": [
    {
      "label": "potted_plant",
      "text": "",
      "points": [[x1,y1],[x2,y2],...],
      "shape_type": "linestrip"
    },
    {
      "label": "plant",
      "text": "",
      "points": [[x1,y1],[x2,y2],...],
      "shape_type": "linestrip"
    },
    {
      "label": "pot",
      "text": "",
      "points": [[x1,y1],[x2,y2],...],
      "shape_type": "linestrip"
    }
  ],
  "imagePath": "0000.jpg",
  "imageData": null,
  "imageHeight": 3840,
  "imageWidth": 2160,
  "text": ""
}
```

- `imagePath`: just the frame filename (e.g., `"0052.jpg"`)
- `shape_type`: `"linestrip"` (closed polygon; script auto-closes)
- `points`: clockwise or counterclockwise; all vertices
- One JSON per frame; one or more `shapes` entries per JSON

**Reference file**: `/data/fj/F2DMAS/03-GT-区分/CaoMei1/0000.json`

---

## §6. Annotation Workflow

### 6.1 Annotation package location

All annotation materials are in:
```
阶段十六_HardCase_GT构建与锁定/annotation_package/<challenge_id>/
  ├── target.jpg          ← the frame to annotate (copy, do not modify)
  ├── context_prev.jpg    ← frame at margin-1 (±5 frames)
  ├── context_next.jpg    ← frame at margin+1
  ├── instruction_card.md ← difficulty tag + identity hint
  └── contact_sheet.jpg   ← per-sample overview (shared per sample, in sample root)
```

### 6.2 Labelme startup

```bash
# From the annotation package target frame directory:
labelme <path-to-target.jpg>  # will open labelme GUI
```

### 6.3 Labeling order (recommended)

1. **Open target.jpg** in labelme
2. **Scan context_prev.jpg and context_next.jpg** (view, do not annotate)
3. **Draw `potted_plant`** polygon around the target plant + pot (linestrip, closed)
4. **Draw `plant`** polygon around the leaf+stem portion only (optional but recommended)
5. **Draw `pot`** polygon around the pot only (optional but recommended)
6. **Draw `blue_cube`** if cube is visible (only if present)
7. **Save** as `<frame>.json` alongside `target.jpg`

### 6.4 Save convention

Save the JSON in the **same directory** as `target.jpg`:
```
/data/fj/F2DMAS/03-GT-区分_challenge/<sample>/<frame>.json
```

The original `target.jpg` must remain accessible. The annotation package
provides the original path in `instruction_card.md`.

---

## §7. Quality Gates (Before Accepting a Frame)

| Gate | Criterion | Action |
|---|---|---|
| G-A1 | `potted_plant` shape exists and is non-empty | Reject frame if missing |
| G-A2 | No overlap between `potted_plant` and `blue_cube` > 10px² (if cube present) | Review and correct |
| G-A3 | `potted_plant` area ≥ 500 pixels (no degenerate tiny masks) | Reject or re-annotate |
| G-A4 | `potted_plant` area ≤ 60% of image (no full-image flood-fill mistakes) | Review and correct |
| G-A5 | Shape covers ≥ 80% of visually visible target plant | Review and correct |
| G-A6 | `plant` polygon subset of `potted_plant` polygon (if both labeled) | Fix consistency |

---

## §8. Difficult-Case Guidance (Per Difficulty Category)

| Category | Label | What to expect | Guidance |
|---|---|---|---|
| D1 | neighbor_plant_adhesion | Target leaves touch neighbor leaves | Draw potted_plant boundary carefully; split at the narrowest junction; if ambiguous, use context frames to confirm identity |
| D3 | foreground_background_confusion | Target plant blends with background vegetation | Use color/texture difference; if impossible to split, draw the boundary conservatively (exclude uncertain regions) |
| D6 | small_target | Very small target plant (few leaves) | Zoom in to 400%; annotate all visible plant tissue; use context frames to confirm it's the target |
| D11 | cluttered_background | Complex greenhouse background | Ignore background entirely; focus on the target plant alone; use context frames for identity confirmation |
| D12 | low_contrast | Poor contrast between plant and background | Increase labelme brightness/contrast if needed; use ExG visualization aid (optional) |

---

## §9. Anti-Bias Rules

- **Do not look at any model prediction** (V00, V10, V01, V11). The annotation package
  contains only the RGB frame and context frames. No model output is provided.
- **Identity is determined from context frames**, not from any computed heatmap or saliency.
- **If you cannot determine the target identity**, mark the frame as "ambiguous" in the
  `instruction_card.md` and leave a comment in the JSON `text` field.

---

## §10. File Output Structure

After annotation, all labeled frames must be in:
```
/data/fj/F2DMAS/03-GT-区分_challenge/
├── BaiZhang/
│   ├── 0120.jpg          ← original frame (from annotation_package)
│   ├── 0120.json         ← labelme annotation
│   ├── 0137.jpg
│   ├── 0137.json
│   ├── 0197.jpg
│   ├── 0197.json
│   ├── 0215.jpg
│   └── 0215.json
├── ChangShouHua1/
│   ├── 0043.jpg
│   ├── 0043.json
│   └── ...
└── ... (15 sample directories, 54 frames total)
```

Processing (`process_gt_challenge.py`) will then produce:
```
/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/
阶段十六_HardCase_GT构建与锁定/
├── GT_potted_clean_challenge/     ← binary masks (potted & ~cube)
│   ├── BaiZhang/
│   │   ├── mask_potted_clean_0120.png
│   │   └── ...
│   └── ...
├── GT_QA_challenge/
│   ├── gt_audit_challenge.csv     ← per-frame QA audit
│   └── 54帧可视化/                 ← overlay visualizations
│       ├── BaiZhang/
│       │   ├── 0120_qa.png        ← 3-panel: orig | potted_clean+cube | plant+pot
│       │   └── ...
│       └── ...
└── annotation_package/             ← source materials (read-only after annotation)
```

---

## §11. Inter-Annotator Agreement (Optional, Recommended)

For 20–30% of frames (~11–16 frames), a second annotator independently labels without
seeing the first annotation. Then compute:
- Pixel-level IoU between two `potted_clean` masks
- Agreement threshold: IoU ≥ 0.85
- Frames below threshold require adjudication (third annotator or joint review)

---

## §12. Checklist Before Proceeding to Phase 17

- [ ] All 54 frames annotated and saved as labelme JSON
- [ ] All 54 `potted_clean` masks generated by `process_gt_challenge.py`
- [ ] All 54 overlay visualizations reviewed (at minimum: one pass, no red flags)
- [ ] gt_audit_challenge.csv shows 54/54 frames with `formal_p6_valid = True`
- [ ] No frames with `pot_status = missing` (all pots labeled or verified absent)
- [ ] Foreground ratio: every mask between 500px and 60% of image area
- [ ] SHA256 manifest frozen for all GT masks
- [ ] DEV manifest (26 frames) frozen; TEST manifest (28 frames) frozen
- [ ] Phase 16 report updated to reflect actual annotation results
- [ ] Then: Phase 17 (Locked Hard-case Confirmatory Factorial Evaluation)

---

## §13. Failed / Excluded Frames

If a frame cannot be reliably annotated (ambiguous identity, extreme occlusion,
unreadable frame), do not annotate it. Instead:
1. Remove it from the manifest (update `Phase16_selection_manifest_preGT.csv`)
2. Note the reason in the annotation package's `instruction_card.md`
3. Update the manifest hash (re-run freeze script)
4. Target: ≥45 frames successfully annotated (out of 54)

---

## §14. Summary

| Item | Value |
|---|---|
| Total target frames | 54 (26 DEV + 28 TEST) |
| Samples | 15 non-GT samples |
| GT type | Binary potted_clean (potted_plant AND NOT blue_cube) |
| Annotation tool | Labelme 0.4.43 (linestrip polygons) |
| Target identity | Sequence-context-resolved, not model-output |
| Model bias prevention | No V00/V10/V01/V11 shown |
| QA gates | G-A1 through G-A6 (above) |
| Processing script | `process_gt_challenge.py` (Phase 16 upgraded) |
| Acceptance | ≥45/54 frames with valid GT, all formal_p6_valid |
