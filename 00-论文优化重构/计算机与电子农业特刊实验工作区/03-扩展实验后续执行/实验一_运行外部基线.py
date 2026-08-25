#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run experiment-one external segmentation baselines on the fixed GT subset.

This script intentionally uses a small, shared evaluation subset first:
- KongQueZhuYu GT5
- XianKeLai1 GT1

Outputs are written under:
  00-论文优化重构/数据管理/05-评测结果/S23_Experiment1_VFM_Benchmark

Supported methods in this first executable version:
- clipseg_p2: CLIPSeg text prompt baseline.
- florence2_res: Florence-2 referring-expression segmentation polygon baseline.
- sam2_oracle_box: SAM2 prompted with the manual-GT bounding box. This is an
  oracle prompt upper-bound, not a zero-shot text baseline.
- grounded_sam: Grounding DINO text boxes + local SAM1 masks.
- grounded_sam2: Grounding DINO text boxes + SAM2 masks, if the HF detector and
  SAM2 dependencies are available.
- import_internal: copy existing SAM3/RAP-FSAM3-v2 masks into this shared
  six-frame benchmark directory for same-subset evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
PAPER_ROOT = WORKSPACE_DIR.parent
PROJECT_ROOT = PAPER_ROOT.parent
DATA_ROOT = PAPER_ROOT / "数据管理"
THIRD_PARTY = PROJECT_ROOT / "第三方源码"
SAM1_ROOT = PROJECT_ROOT / "segment-anything"
SAM1_CHECKPOINT = SAM1_ROOT / "checkpoints/sam_vit_h_4b8939.pth"
BENCH_ROOT = DATA_ROOT / "05-评测结果/S23_Experiment1_VFM_Benchmark"

GT_INDEX_FILES = [
    DATA_ROOT / "05-评测结果/S21_KongQueZhuYu_E2_E3/gt_index.csv",
    DATA_ROOT / "05-评测结果/S22_XianKeLai1_RAP_FSAM3_GT1/gt_index.csv",
]

PROMPT_P2 = "entire plant excluding pot"
GROUNDING_PROMPT = "plant."
FLORENCE2_TASK = "<REFERRING_EXPRESSION_SEGMENTATION>"

INTERNAL_MASK_SOURCES = {
    ("SAM3_P2", "KongQueZhuYu"): DATA_ROOT
    / "03-分割Mask/05-RAP-FSAM3掩膜/E3v2_KongQueZhuYu_GT5_A0_单提示词P2/最终掩膜",
    ("SAM3_P2", "XianKeLai1"): DATA_ROOT
    / "05-评测结果/S22_XianKeLai1_RAP_FSAM3_GT1/method_masks/P2_candidate",
    ("RAP-FSAM3-v2", "KongQueZhuYu"): DATA_ROOT
    / "03-分割Mask/05-RAP-FSAM3掩膜/E3v2_KongQueZhuYu_GT5_A5c_完整RAPFSAM3v2/最终掩膜",
    ("RAP-FSAM3-v2", "XianKeLai1"): DATA_ROOT
    / "03-分割Mask/05-RAP-FSAM3掩膜/XianKeLai1_A5c_语义门控几何修正_GT1冒烟/最终掩膜",
}


@dataclass(frozen=True)
class SampleFrame:
    sample: str
    frame: str
    source_image: Path
    selected_image: Path
    gt_mask: Path

    @property
    def stem(self) -> str:
        return f"{self.sample}_{self.frame}"


def read_index_rows() -> list[SampleFrame]:
    frames: list[SampleFrame] = []
    for index_path in GT_INDEX_FILES:
        with index_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                source = PROJECT_ROOT / row["source_image"]
                selected = PROJECT_ROOT / row["selected_image"]
                gt_mask = PROJECT_ROOT / row["gt_mask"]
                sample = source.parent.name
                frames.append(
                    SampleFrame(
                        sample=sample,
                        frame=row["frame"],
                        source_image=source,
                        selected_image=selected,
                        gt_mask=gt_mask,
                    )
                )
    return frames


def prepare_subset() -> None:
    image_dir = BENCH_ROOT / "selected_frames"
    gt_dir = BENCH_ROOT / "gt_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for item in read_index_rows():
        image_out = image_dir / f"{item.stem}.jpg"
        gt_out = gt_dir / f"mask_{item.stem}.png"
        shutil.copy2(item.selected_image, image_out)
        shutil.copy2(item.gt_mask, gt_out)
        manifest_rows.append(
            {
                "sample": item.sample,
                "frame": item.frame,
                "image": str(image_out),
                "gt_mask": str(gt_out),
                "source_image": str(item.source_image),
                "original_gt_mask": str(item.gt_mask),
            }
        )

    write_csv(
        BENCH_ROOT / "manifest.csv",
        ["sample", "frame", "image", "gt_mask", "source_image", "original_gt_mask"],
        manifest_rows,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_paths() -> list[Path]:
    prepare_subset()
    return sorted((BENCH_ROOT / "selected_frames").glob("*.jpg"))


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def mask_bbox(mask: np.ndarray, pad_ratio: float = 0.02) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0, 0, mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = int(mask.shape[1] * pad_ratio)
    pad_y = int(mask.shape[0] * pad_ratio)
    return np.array(
        [
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(mask.shape[1] - 1, x1 + pad_x),
            min(mask.shape[0] - 1, y1 + pad_y),
        ],
        dtype=np.float32,
    )


def run_clipseg(threshold: float = 0.5) -> None:
    import torch
    from PIL import Image
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    out_dir = BENCH_ROOT / "method_masks/CLIPSeg_P2"
    log_rows: list[dict[str, object]] = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
    model.eval()

    for image_path in image_paths():
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[PROMPT_P2], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)[0].detach().float().cpu().numpy()
        probs = cv2.resize(
            probs,
            image.size,
            interpolation=cv2.INTER_LINEAR,
        )
        mask = probs >= threshold
        out_path = out_dir / f"mask_{image_path.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "image": str(image_path),
                "mask": str(out_path),
                "prompt": PROMPT_P2,
                "threshold": threshold,
                "foreground_ratio": float(mask.mean()),
            }
        )

    write_csv(out_dir / "run_log.csv", ["image", "mask", "prompt", "threshold", "foreground_ratio"], log_rows)


def polygons_to_mask(parsed: dict[str, object], task: str, image_size: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    task_result = parsed.get(task, {})
    polygons = task_result.get("polygons", []) if isinstance(task_result, dict) else []
    polygon_count = 0
    coord_count = 0

    for group in polygons:
        for poly in group:
            if len(poly) < 6:
                continue
            points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
            polygon_count += 1
            coord_count += len(poly)

    return mask > 0, polygon_count, coord_count


def run_florence2_res() -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Florence-2-base-ft"
    out_dir = BENCH_ROOT / "method_masks/Florence2_RES_P2"
    log_rows: list[dict[str, object]] = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False
    if hasattr(model, "language_model"):
        model.language_model.config.use_cache = False
    model.eval()

    for image_path in image_paths():
        image = Image.open(image_path).convert("RGB")
        prompt = FLORENCE2_TASK + PROMPT_P2
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=FLORENCE2_TASK,
            image_size=image.size,
        )
        mask, polygon_count, coord_count = polygons_to_mask(parsed, FLORENCE2_TASK, image.size)
        out_path = out_dir / f"mask_{image_path.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "image": str(image_path),
                "mask": str(out_path),
                "model_id": model_id,
                "task": FLORENCE2_TASK,
                "prompt": PROMPT_P2,
                "polygon_count": polygon_count,
                "coord_count": coord_count,
                "foreground_ratio": float(mask.mean()),
            }
        )

    write_csv(
        out_dir / "run_log.csv",
        [
            "image",
            "mask",
            "model_id",
            "task",
            "prompt",
            "polygon_count",
            "coord_count",
            "foreground_ratio",
        ],
        log_rows,
    )


def import_sam2() -> None:
    sam2_root = THIRD_PARTY / "SAM2"
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))


def get_sam2_predictor():
    import torch

    import_sam2()
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    return predictor, device


def get_grounding_dino_detector():
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    detector_id = "IDEA-Research/grounding-dino-base"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(detector_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device)
    detector.eval()
    return detector_id, processor, detector, device


def detect_grounding_boxes(
    *,
    processor,
    detector,
    device: str,
    pil_image,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = detector(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_image.size[::-1]],
    )[0]
    boxes = results["boxes"].detach().float().cpu().numpy()
    scores = results["scores"].detach().float().cpu().numpy()
    return boxes, scores


def run_sam2_oracle_box() -> None:
    import torch

    predictor, device = get_sam2_predictor()
    out_dir = BENCH_ROOT / "method_masks/SAM2_oracle_box"
    log_rows: list[dict[str, object]] = []

    for image_path in image_paths():
        gt_path = BENCH_ROOT / "gt_masks" / f"mask_{image_path.stem}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Cannot read GT mask: {gt_path}")
        box = mask_bbox(gt > 127)
        image = read_rgb(image_path)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        best = int(np.argmax(scores))
        mask = masks[best] > 0
        out_path = out_dir / f"mask_{image_path.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "image": str(image_path),
                "mask": str(out_path),
                "prompt_source": "manual_gt_bbox_oracle",
                "box_xyxy": " ".join(f"{float(x):.1f}" for x in box),
                "score": float(scores[best]),
                "foreground_ratio": float(mask.mean()),
            }
        )

    write_csv(
        out_dir / "run_log.csv",
        ["image", "mask", "prompt_source", "box_xyxy", "score", "foreground_ratio"],
        log_rows,
    )


def run_grounded_sam2(
    box_threshold: float = 0.2,
    text_threshold: float = 0.2,
) -> None:
    import torch
    from PIL import Image

    predictor, device = get_sam2_predictor()
    detector_id, processor, detector, device = get_grounding_dino_detector()

    out_dir = BENCH_ROOT / "method_masks/GroundedSAM2_Plant"
    log_rows: list[dict[str, object]] = []

    for image_path in image_paths():
        pil_image = Image.open(image_path).convert("RGB")
        boxes, scores = detect_grounding_boxes(
            processor=processor,
            detector=detector,
            device=device,
            pil_image=pil_image,
            prompt=GROUNDING_PROMPT,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        image = read_rgb(image_path)
        if len(boxes) == 0:
            mask = np.zeros(image.shape[:2], dtype=bool)
            chosen = ""
            best_score = ""
        else:
            # Use all detected plant boxes and union their SAM2 masks.
            union = np.zeros(image.shape[:2], dtype=bool)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                predictor.set_image(image)
                for box in boxes:
                    masks, mask_scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
                    union |= masks[int(np.argmax(mask_scores))] > 0
            mask = union
            chosen = ";".join(" ".join(f"{float(x):.1f}" for x in box) for box in boxes)
            best_score = float(scores.max())
        out_path = out_dir / f"mask_{image_path.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "image": str(image_path),
                "mask": str(out_path),
                "prompt": GROUNDING_PROMPT,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "num_boxes": len(boxes),
                "best_score": best_score,
                "boxes_xyxy": chosen,
                "foreground_ratio": float(mask.mean()),
            }
        )

    write_csv(
        out_dir / "run_log.csv",
        [
            "image",
            "mask",
            "prompt",
            "box_threshold",
            "text_threshold",
            "num_boxes",
            "best_score",
            "boxes_xyxy",
            "foreground_ratio",
        ],
        log_rows,
    )


def import_sam1() -> None:
    if str(SAM1_ROOT) not in sys.path:
        sys.path.insert(0, str(SAM1_ROOT))


def run_grounded_sam(
    box_threshold: float = 0.2,
    text_threshold: float = 0.2,
) -> None:
    import torch
    from PIL import Image

    if not SAM1_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing SAM1 checkpoint: {SAM1_CHECKPOINT}")
    import_sam1()
    from segment_anything import SamPredictor, sam_model_registry

    detector_id, processor, detector, device = get_grounding_dino_detector()
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM1_CHECKPOINT)).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    out_dir = BENCH_ROOT / "method_masks/GroundedSAM1_Plant"
    log_rows: list[dict[str, object]] = []

    for image_path in image_paths():
        pil_image = Image.open(image_path).convert("RGB")
        boxes, scores = detect_grounding_boxes(
            processor=processor,
            detector=detector,
            device=device,
            pil_image=pil_image,
            prompt=GROUNDING_PROMPT,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        image = read_rgb(image_path)
        if len(boxes) == 0:
            mask = np.zeros(image.shape[:2], dtype=bool)
            chosen = ""
            best_score = ""
        else:
            union = np.zeros(image.shape[:2], dtype=bool)
            predictor.set_image(image)
            for box in boxes:
                masks, mask_scores, _ = predictor.predict(
                    box=box.astype(np.float32),
                    multimask_output=True,
                )
                union |= masks[int(np.argmax(mask_scores))] > 0
            mask = union
            chosen = ";".join(" ".join(f"{float(x):.1f}" for x in box) for box in boxes)
            best_score = float(scores.max())
        out_path = out_dir / f"mask_{image_path.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "image": str(image_path),
                "mask": str(out_path),
                "detector_id": detector_id,
                "sam_checkpoint": str(SAM1_CHECKPOINT),
                "prompt": GROUNDING_PROMPT,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "num_boxes": len(boxes),
                "best_score": best_score,
                "boxes_xyxy": chosen,
                "foreground_ratio": float(mask.mean()),
            }
        )

    write_csv(
        out_dir / "run_log.csv",
        [
            "image",
            "mask",
            "detector_id",
            "sam_checkpoint",
            "prompt",
            "box_threshold",
            "text_threshold",
            "num_boxes",
            "best_score",
            "boxes_xyxy",
            "foreground_ratio",
        ],
        log_rows,
    )


def import_internal_masks() -> None:
    """Copy project-internal method masks into S23's shared naming scheme."""
    prepare_subset()
    rows: list[dict[str, object]] = []
    for item in read_index_rows():
        for method_name in ("SAM3_P2", "RAP-FSAM3-v2"):
            source_dir = INTERNAL_MASK_SOURCES[(method_name, item.sample)]
            source_path = source_dir / f"mask_{item.frame}.png"
            if not source_path.exists():
                raise FileNotFoundError(f"Missing internal mask: {source_path}")
            out_path = BENCH_ROOT / "method_masks" / method_name / f"mask_{item.stem}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, out_path)
            rows.append(
                {
                    "method": method_name,
                    "sample": item.sample,
                    "frame": item.frame,
                    "source_mask": str(source_path),
                    "imported_mask": str(out_path),
                }
            )

    write_csv(
        BENCH_ROOT / "method_masks/internal_import_log.csv",
        ["method", "sample", "frame", "source_mask", "imported_mask"],
        rows,
    )


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if shape is not None and img.shape != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img > 127


def confusion(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    tn = int(np.logical_and(~gt, ~pred).sum(dtype=np.int64))
    return tp, fp, fn, tn


def pixel_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_fg": iou_fg,
        "miou": (iou_fg + iou_bg) / 2.0,
    }


def boundary_map(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return np.logical_and(m > 0, eroded == 0)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance_px: int) -> float:
    gt_b = boundary_map(gt)
    pr_b = boundary_map(pred)
    gt_n = int(gt_b.sum(dtype=np.int64))
    pr_n = int(pr_b.sum(dtype=np.int64))
    if gt_n == 0 and pr_n == 0:
        return 1.0
    if gt_n == 0 or pr_n == 0:
        return 0.0
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1)
    )
    gt_d = cv2.dilate(gt_b.astype(np.uint8), k, iterations=1) > 0
    pr_d = cv2.dilate(pr_b.astype(np.uint8), k, iterations=1) > 0
    precision = safe_div(int(np.logical_and(pr_b, gt_d).sum(dtype=np.int64)), pr_n)
    recall = safe_div(int(np.logical_and(gt_b, pr_d).sum(dtype=np.int64)), gt_n)
    return safe_div(2 * precision * recall, precision + recall)


def hd95(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_b = boundary_map(gt)
    pr_b = boundary_map(pred)
    if not gt_b.any() and not pr_b.any():
        return 0.0
    diag = float(np.hypot(gt.shape[0], gt.shape[1]))
    if not gt_b.any() or not pr_b.any():
        return diag
    dist_to_pr = cv2.distanceTransform((~pr_b).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_gt = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    d1 = dist_to_pr[gt_b]
    d2 = dist_to_gt[pr_b]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def component_count(mask: np.ndarray, min_area_ratio: float = 0.0005) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0
    min_area = max(1, int(mask.size * min_area_ratio))
    return int(sum(int(stats[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, n)))


def evaluate_method(method_name: str, mask_dir: Path, boundary_tol: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    gt_dir = BENCH_ROOT / "gt_masks"
    gt_paths = sorted(gt_dir.glob("mask_*.png"))
    frame_rows: list[dict[str, object]] = []
    tp = fp = fn = tn = 0
    areas: list[float] = []
    comps: list[int] = []
    temporal_ious: list[float] = []
    hd95_values: list[float] = []
    bf_values: list[float] = []
    missing: list[str] = []
    prev_pred: np.ndarray | None = None

    for gt_path in gt_paths:
        stem = gt_path.stem.removeprefix("mask_")
        pred_path = mask_dir / f"mask_{stem}.png"
        gt = read_mask(gt_path)
        if not pred_path.exists():
            missing.append(stem)
            continue
        pred = read_mask(pred_path, gt.shape)
        c = confusion(gt, pred)
        tp += c[0]
        fp += c[1]
        fn += c[2]
        tn += c[3]
        mm = pixel_metrics(*c)
        h = hd95(gt, pred)
        b = boundary_f1(gt, pred, boundary_tol)
        area_ratio = float(pred.sum() / pred.size)
        comp = component_count(pred)
        temporal_iou = ""
        if prev_pred is not None:
            union = np.logical_or(prev_pred, pred).sum()
            temporal_iou = safe_div(int(np.logical_and(prev_pred, pred).sum()), int(union))
            temporal_ious.append(float(temporal_iou))
        prev_pred = pred
        outside_nonblack = safe_div(c[1], c[1] + c[3])
        leakage_energy = safe_div(c[1], gt.size)
        areas.append(area_ratio)
        comps.append(comp)
        hd95_values.append(h)
        bf_values.append(b)
        frame_rows.append(
            {
                "method": method_name,
                "frame": stem,
                "mask_path": str(pred_path),
                "gt_path": str(gt_path),
                "precision": mm["precision"],
                "recall": mm["recall"],
                "f1": mm["f1"],
                "miou": mm["miou"],
                "iou_fg": mm["iou_fg"],
                "hd95_px": h,
                "boundary_f1": b,
                "area_ratio": area_ratio,
                "component_count": comp,
                "temporal_iou": temporal_iou,
                "outside_nonblack_ratio": outside_nonblack,
                "leakage_energy": leakage_energy,
            }
        )

    summary_metrics = pixel_metrics(tp, fp, fn, tn)
    gt_shape = read_mask(gt_paths[0]).shape if gt_paths else (0, 0)
    summary = {
        "method": method_name,
        "mask_dir": str(mask_dir),
        "gt_frames": len(gt_paths),
        "eval_frames": len(frame_rows),
        "missing_frames": ";".join(missing),
        "precision": summary_metrics["precision"],
        "recall": summary_metrics["recall"],
        "f1": summary_metrics["f1"],
        "miou": summary_metrics["miou"],
        "iou_fg": summary_metrics["iou_fg"],
        "hd95_px": float(np.mean(hd95_values)) if hd95_values else "",
        "boundary_f1": float(np.mean(bf_values)) if bf_values else "",
        "temporal_iou": float(np.mean(temporal_ious)) if temporal_ious else "",
        "area_cv": safe_div(float(np.std(areas)), float(np.mean(areas))) if areas else "",
        "component_count_mean": float(np.mean(comps)) if comps else "",
        "outside_nonblack_ratio": safe_div(fp, fp + tn),
        "leakage_energy": safe_div(fp, len(frame_rows) * gt_shape[0] * gt_shape[1]) if frame_rows else "",
    }
    return summary, frame_rows


def evaluate_all(boundary_tol: int = 3) -> None:
    prepare_subset()
    method_root = BENCH_ROOT / "method_masks"
    method_dirs = sorted([p for p in method_root.iterdir() if p.is_dir()]) if method_root.exists() else []
    summary_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for method_dir in method_dirs:
        summary, rows = evaluate_method(method_dir.name, method_dir, boundary_tol)
        summary_rows.append(summary)
        frame_rows.extend(rows)

    summary_fields = [
        "method",
        "mask_dir",
        "gt_frames",
        "eval_frames",
        "missing_frames",
        "precision",
        "recall",
        "f1",
        "miou",
        "iou_fg",
        "hd95_px",
        "boundary_f1",
        "temporal_iou",
        "area_cv",
        "component_count_mean",
        "outside_nonblack_ratio",
        "leakage_energy",
    ]
    frame_fields = [
        "method",
        "frame",
        "mask_path",
        "gt_path",
        "precision",
        "recall",
        "f1",
        "miou",
        "iou_fg",
        "hd95_px",
        "boundary_f1",
        "area_ratio",
        "component_count",
        "temporal_iou",
        "outside_nonblack_ratio",
        "leakage_energy",
    ]
    metrics_dir = BENCH_ROOT / "metrics"
    write_csv(metrics_dir / "summary_metrics.csv", summary_fields, summary_rows)
    write_csv(metrics_dir / "frame_metrics.csv", frame_fields, frame_rows)
    (metrics_dir / "summary_metrics.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Evaluated {len(summary_rows)} methods. Output: {metrics_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment-one external baselines.")
    parser.add_argument(
        "--method",
        choices=[
            "prepare",
            "clipseg_p2",
            "florence2_res",
            "sam2_oracle_box",
            "grounded_sam",
            "grounded_sam2",
            "import_internal",
            "evaluate",
        ],
        required=True,
    )
    parser.add_argument("--clipseg-threshold", type=float, default=0.5)
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--boundary-tol", type=int, default=3)
    args = parser.parse_args()

    if args.method == "prepare":
        prepare_subset()
        print(f"Prepared subset under {BENCH_ROOT}")
    elif args.method == "clipseg_p2":
        run_clipseg(args.clipseg_threshold)
    elif args.method == "florence2_res":
        run_florence2_res()
    elif args.method == "sam2_oracle_box":
        run_sam2_oracle_box()
    elif args.method == "grounded_sam":
        run_grounded_sam(args.box_threshold, args.text_threshold)
    elif args.method == "grounded_sam2":
        run_grounded_sam2(args.box_threshold, args.text_threshold)
    elif args.method == "import_internal":
        import_internal_masks()
    elif args.method == "evaluate":
        evaluate_all(args.boundary_tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
