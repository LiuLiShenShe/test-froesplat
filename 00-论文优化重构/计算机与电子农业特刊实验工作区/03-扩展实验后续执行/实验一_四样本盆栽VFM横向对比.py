#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Four-sample potted-plant benchmark for experiment one.

This script builds a self-contained 4-sample x 5-frame benchmark under the
paper workspace. It does not overwrite the older S23 six-frame benchmark.
The GT target follows the existing LabelMe GT masks used by the project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
PAPER_ROOT = WORKSPACE_DIR.parent
PROJECT_ROOT = PAPER_ROOT.parent
THIRD_PARTY = PROJECT_ROOT / "第三方源码"

BENCH_ROOT = WORKSPACE_DIR / "06-实验一四样本代表集/E1_Representative4_PottedPlant_VFM"
GT_ROOT = PROJECT_ROOT / "03-GT"
SAM_MASK_ROOT = PROJECT_ROOT / "03-SAM"
SEEM_MASK_ROOT = PROJECT_ROOT / "03-SEEM"
SAM2_ROOT = THIRD_PARTY / "SAM2"
SAM1_ROOT = PROJECT_ROOT / "segment-anything"
SAM1_CHECKPOINT = SAM1_ROOT / "checkpoints/sam_vit_h_4b8939.pth"

SAMPLES = ["KongQueZhuYu", "DouBanLv1", "ChangShouHua2", "CaoMei1"]
FRAMES = ["0000", "0025", "0050", "0075", "0100"]

PROMPT_POTTED = "whole potted plant including flowerpot"
GROUNDING_PROMPT = "potted plant."
FLORENCE2_TASK = "<REFERRING_EXPRESSION_SEGMENTATION>"
TARGET_DEFINITION = "project LabelMe GT foreground: union of closed target-object linestrips"

METHOD_ORDER = [
    "SEEM_existing",
    "CLIPSeg_potted",
    "GroundedSAM1_potted",
    "GroundedSAM2_potted",
    "Florence2_potted",
    "SAM2_oracle_box",
    "SAM3_P2",
    "RAP-FSAM3-v2",
    "UNet_fewshot_seqcv",
    "DeepLabV3PlusLite_fewshot_seqcv",
    "SAM_existing",
]

DISPLAY_NAME = {
    "SEEM_existing": "SEEM",
    "SAM_existing": "SAM existing",
    "CLIPSeg_potted": "CLIPSeg",
    "GroundedSAM1_potted": "Grounded-SAM",
    "GroundedSAM2_potted": "Grounded-SAM2",
    "Florence2_potted": "Florence-2",
    "SAM2_oracle_box": "SAM2 oracle",
    "SAM3_P2": "SAM3 single prompt",
    "RAP-FSAM3-v2": "RAP-FSAM3-v2",
    "UNet_fewshot_seqcv": "U-Net",
    "DeepLabV3PlusLite_fewshot_seqcv": "DeepLabv3+ lite",
}

METHOD_COLORS = {
    "SEEM_existing": (196, 78, 82),
    "SAM_existing": (156, 106, 222),
    "CLIPSeg_potted": (242, 177, 52),
    "GroundedSAM1_potted": (86, 145, 207),
    "GroundedSAM2_potted": (74, 164, 154),
    "Florence2_potted": (224, 122, 95),
    "SAM2_oracle_box": (47, 109, 181),
    "SAM3_P2": (115, 101, 191),
    "RAP-FSAM3-v2": (65, 150, 86),
    "UNet_fewshot_seqcv": (186, 120, 64),
    "DeepLabV3PlusLite_fewshot_seqcv": (128, 128, 128),
}

METHOD_PROTOCOL = {
    "SEEM_existing": "existing project output",
    "SAM_existing": "existing project output; historical reference only",
    "CLIPSeg_potted": "zero-shot text prompt",
    "GroundedSAM1_potted": "GroundingDINO text box + SAM1",
    "GroundedSAM2_potted": "GroundingDINO text box + SAM2",
    "Florence2_potted": "referring-expression segmentation",
    "SAM2_oracle_box": "oracle GT bounding-box prompt",
    "SAM3_P2": "SAM3 P2 single prompt",
    "RAP-FSAM3-v2": "ours: multi-prompt selection + refinement + geometry correction",
    "UNet_fewshot_seqcv": "leave-one-sequence-out few-shot training",
    "DeepLabV3PlusLite_fewshot_seqcv": "leave-one-sequence-out few-shot training",
}

MAIN_TABLE_METHODS = {
    "SEEM_existing",
    "CLIPSeg_potted",
    "GroundedSAM1_potted",
    "GroundedSAM2_potted",
    "Florence2_potted",
    "SAM2_oracle_box",
    "SAM3_P2",
    "RAP-FSAM3-v2",
    "UNet_fewshot_seqcv",
    "DeepLabV3PlusLite_fewshot_seqcv",
}

GT_COLOR = (0, 214, 201)
FP_COLOR = (222, 77, 77)
FN_COLOR = (62, 126, 214)
TP_COLOR = (75, 170, 90)
TEXT_COLOR = (34, 34, 34)
GRID_COLOR = (216, 216, 216)
_TORCHVISION_NMS_LIB = None


def ensure_torchvision_nms_stub() -> None:
    """Define torchvision::nms for mismatched torch/torchvision installs.

    The current workstation can import torch, but torchvision's fake-kernel
    registration fails if the compiled nms operator is absent. Transformers
    imports torchvision from its image utilities, so define the operator schema
    before those lazy imports are resolved.
    """
    global _TORCHVISION_NMS_LIB
    if _TORCHVISION_NMS_LIB is not None:
        return
    try:
        import torch

        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
    except RuntimeError:
        try:
            from torch.library import Library

            _TORCHVISION_NMS_LIB = Library("torchvision", "DEF")
            _TORCHVISION_NMS_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except RuntimeError:
            pass
    except Exception:
        pass


@dataclass(frozen=True)
class FrameRecord:
    sample: str
    frame: str
    image: Path
    gt_json: Path
    gt_mask: Path

    @property
    def stem(self) -> str:
        return f"{self.sample}_{self.frame}"


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def shape_to_mask(shape: dict[str, object], height: int, width: int) -> np.ndarray:
    points = shape.get("points", [])
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(points, list) or len(points) < 3:
        return mask
    pts = np.asarray(points, dtype=np.float32)
    pts = np.rint(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def convert_labelme(json_path: Path) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    height = int(data["imageHeight"])
    width = int(data["imageWidth"])
    mask = np.zeros((height, width), dtype=np.uint8)
    shape_rows: list[dict[str, object]] = []
    for idx, shape in enumerate(data.get("shapes", []), start=1):
        sm = shape_to_mask(shape, height, width)
        mask = np.maximum(mask, sm)
        points = shape.get("points", [])
        pts = np.asarray(points, dtype=np.float32) if isinstance(points, list) else np.zeros((0, 2))
        close_dist = ""
        if len(pts) >= 2:
            close_dist = float(np.linalg.norm(pts[0] - pts[-1]))
        area_px = int((sm > 0).sum())
        bbox = ["", "", "", ""]
        if area_px > 0:
            ys, xs = np.where(sm > 0)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        shape_rows.append(
            {
                "shape_index": idx,
                "label": shape.get("label", ""),
                "shape_type": shape.get("shape_type", ""),
                "points": len(points) if isinstance(points, list) else 0,
                "close_distance_px": close_dist,
                "area_px": area_px,
                "bbox_x0": bbox[0],
                "bbox_y0": bbox[1],
                "bbox_x1": bbox[2],
                "bbox_y1": bbox[3],
            }
        )
    meta = {
        "width": width,
        "height": height,
        "image_path": data.get("imagePath", json_path.with_suffix(".jpg").name),
        "shape_count": len(shape_rows),
        "gt_area_ratio": float((mask > 0).sum() / mask.size),
    }
    return mask, meta, shape_rows


def prepare_dataset() -> None:
    selected_dir = BENCH_ROOT / "selected_frames"
    gt_dir = BENCH_ROOT / "gt_masks"
    selected_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    shape_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        for frame in FRAMES:
            source_image = GT_ROOT / sample / f"{frame}.jpg"
            gt_json = GT_ROOT / sample / f"{frame}.json"
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            if not gt_json.exists():
                raise FileNotFoundError(gt_json)
            image_out = selected_dir / f"{sample}_{frame}.jpg"
            gt_out = gt_dir / f"mask_{sample}_{frame}.png"
            shutil.copy2(source_image, image_out)
            mask, meta, rows = convert_labelme(gt_json)
            cv2.imwrite(str(gt_out), mask)
            manifest_rows.append(
                {
                    "sample": sample,
                    "frame": frame,
                    "image": str(image_out),
                    "gt_mask": str(gt_out),
                    "source_image": str(source_image),
                    "gt_json": str(gt_json),
                    "width": meta["width"],
                    "height": meta["height"],
                    "gt_area_ratio": meta["gt_area_ratio"],
                    "num_shapes": meta["shape_count"],
                    "target_definition": TARGET_DEFINITION,
                }
            )
            for row in rows:
                shape_rows.append({"sample": sample, "frame": frame, **row})

    write_csv(
        BENCH_ROOT / "manifest.csv",
        [
            "sample",
            "frame",
            "image",
            "gt_mask",
            "source_image",
            "gt_json",
            "width",
            "height",
            "gt_area_ratio",
            "num_shapes",
            "target_definition",
        ],
        manifest_rows,
    )
    write_csv(
        BENCH_ROOT / "gt_shape_summary.csv",
        [
            "sample",
            "frame",
            "shape_index",
            "label",
            "shape_type",
            "points",
            "close_distance_px",
            "area_px",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
        ],
        shape_rows,
    )


def load_manifest() -> list[FrameRecord]:
    prepare_dataset()
    rows = read_csv(BENCH_ROOT / "manifest.csv")
    return [
        FrameRecord(
            sample=row["sample"],
            frame=row["frame"],
            image=Path(row["image"]),
            gt_json=Path(row["gt_json"]),
            gt_mask=Path(row["gt_mask"]),
        )
        for row in rows
    ]


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, shape: tuple[int, int] | None = None, threshold: int = 127) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            gray = np.max(img, axis=2)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if shape is not None and gray.shape != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return gray > threshold


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def import_existing_masks() -> None:
    prepare_dataset()
    rows: list[dict[str, object]] = []
    for record in load_manifest():
        gt = read_mask(record.gt_mask)
        sources = {
            "SAM_existing": SAM_MASK_ROOT / record.sample / f"mask_{record.frame}.png",
            "SEEM_existing": SEEM_MASK_ROOT / record.sample / f"{record.frame}.jpg",
        }
        for method, src in sources.items():
            out = BENCH_ROOT / "method_masks" / method / f"mask_{record.stem}.png"
            if not src.exists():
                rows.append(
                    {
                        "method": method,
                        "sample": record.sample,
                        "frame": record.frame,
                        "source_mask": str(src),
                        "imported_mask": "",
                        "status": "missing_source",
                    }
                )
                continue
            try:
                mask = read_mask(src, gt.shape, threshold=127)
            except FileNotFoundError:
                rows.append(
                    {
                        "method": method,
                        "sample": record.sample,
                        "frame": record.frame,
                        "source_mask": str(src),
                        "imported_mask": "",
                        "status": "unreadable_source",
                    }
                )
                continue
            save_mask(mask, out)
            rows.append(
                {
                    "method": method,
                    "sample": record.sample,
                    "frame": record.frame,
                    "source_mask": str(src),
                    "imported_mask": str(out),
                    "status": "ok",
                }
            )
    write_csv(
        BENCH_ROOT / "method_masks/existing_import_log.csv",
        ["method", "sample", "frame", "source_mask", "imported_mask", "status"],
        rows,
    )


def import_rap_runs() -> None:
    """Import freshly generated SAM3/RAP masks into the shared benchmark layout."""
    prepare_dataset()
    source_specs = {
        "SAM3_P2": ("候选掩膜/P2_整株去花盆", "SAM3 P2 candidate mask"),
        "RAP-FSAM3-v2": ("最终掩膜", "RAP-FSAM3-v2 final mask"),
    }
    rows: list[dict[str, object]] = []
    for record in load_manifest():
        gt = read_mask(record.gt_mask)
        for method, (subdir, source_stage) in source_specs.items():
            src = BENCH_ROOT / "rap_runs" / method / record.sample / subdir / f"mask_{record.frame}.png"
            out = BENCH_ROOT / "method_masks" / method / f"mask_{record.stem}.png"
            if not src.exists():
                rows.append(
                    {
                        "method": method,
                        "sample": record.sample,
                        "frame": record.frame,
                        "source_stage": source_stage,
                        "source_mask": str(src),
                        "imported_mask": "",
                        "status": "missing_source",
                    }
                )
                continue
            try:
                mask = read_mask(src, gt.shape, threshold=127)
            except FileNotFoundError:
                rows.append(
                    {
                        "method": method,
                        "sample": record.sample,
                        "frame": record.frame,
                        "source_stage": source_stage,
                        "source_mask": str(src),
                        "imported_mask": "",
                        "status": "unreadable_source",
                    }
                )
                continue
            save_mask(mask, out)
            rows.append(
                {
                    "method": method,
                    "sample": record.sample,
                    "frame": record.frame,
                    "source_stage": source_stage,
                    "source_mask": str(src),
                    "imported_mask": str(out),
                    "status": "ok",
                }
            )
    write_csv(
        BENCH_ROOT / "method_masks/rap_import_log.csv",
        ["method", "sample", "frame", "source_stage", "source_mask", "imported_mask", "status"],
        rows,
    )


def run_clipseg(threshold: float = 0.5) -> None:
    ensure_torchvision_nms_stub()
    import torch
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    out_dir = BENCH_ROOT / "method_masks/CLIPSeg_potted"
    log_rows: list[dict[str, object]] = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()
    for record in load_manifest():
        image = Image.open(record.image).convert("RGB")
        inputs = processor(text=[PROMPT_POTTED], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            probs = torch.sigmoid(model(**inputs).logits)[0].detach().float().cpu().numpy()
        probs = cv2.resize(probs, image.size, interpolation=cv2.INTER_LINEAR)
        mask = probs >= threshold
        out_path = out_dir / f"mask_{record.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "sample": record.sample,
                "frame": record.frame,
                "image": str(record.image),
                "mask": str(out_path),
                "prompt": PROMPT_POTTED,
                "threshold": threshold,
                "foreground_ratio": float(mask.mean()),
            }
        )
    write_csv(
        out_dir / "run_log.csv",
        ["sample", "frame", "image", "mask", "prompt", "threshold", "foreground_ratio"],
        log_rows,
    )


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


def run_florence2() -> None:
    ensure_torchvision_nms_stub()
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Florence-2-base-ft"
    out_dir = BENCH_ROOT / "method_masks/Florence2_potted"
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
    for record in load_manifest():
        image = Image.open(record.image).convert("RGB")
        prompt = FLORENCE2_TASK + PROMPT_POTTED
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
        out_path = out_dir / f"mask_{record.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "sample": record.sample,
                "frame": record.frame,
                "image": str(record.image),
                "mask": str(out_path),
                "model_id": model_id,
                "task": FLORENCE2_TASK,
                "prompt": PROMPT_POTTED,
                "polygon_count": polygon_count,
                "coord_count": coord_count,
                "foreground_ratio": float(mask.mean()),
            }
        )
    write_csv(
        out_dir / "run_log.csv",
        [
            "sample",
            "frame",
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


def import_sam2() -> None:
    if str(SAM2_ROOT) not in sys.path:
        sys.path.insert(0, str(SAM2_ROOT))


def get_sam2_predictor():
    ensure_torchvision_nms_stub()
    import torch

    import_sam2()
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    return predictor, device


def run_sam2_oracle_box() -> None:
    ensure_torchvision_nms_stub()
    import torch

    predictor, device = get_sam2_predictor()
    out_dir = BENCH_ROOT / "method_masks/SAM2_oracle_box"
    log_rows: list[dict[str, object]] = []
    for record in load_manifest():
        gt = read_mask(record.gt_mask)
        box = mask_bbox(gt)
        image = read_rgb(record.image)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        best = int(np.argmax(scores))
        mask = masks[best] > 0
        out_path = out_dir / f"mask_{record.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "sample": record.sample,
                "frame": record.frame,
                "image": str(record.image),
                "mask": str(out_path),
                "prompt_source": "manual_gt_bbox_oracle",
                "box_xyxy": " ".join(f"{float(x):.1f}" for x in box),
                "score": float(scores[best]),
                "foreground_ratio": float(mask.mean()),
            }
        )
    write_csv(
        out_dir / "run_log.csv",
        ["sample", "frame", "image", "mask", "prompt_source", "box_xyxy", "score", "foreground_ratio"],
        log_rows,
    )


def get_grounding_dino_detector():
    ensure_torchvision_nms_stub()
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    detector_id = "IDEA-Research/grounding-dino-base"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(detector_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device)
    detector.eval()
    return detector_id, processor, detector, device


def import_sam1() -> None:
    if str(SAM1_ROOT) not in sys.path:
        sys.path.insert(0, str(SAM1_ROOT))


def detect_grounding_boxes(
    *,
    processor,
    detector,
    device: str,
    pil_image: Image.Image,
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


def run_grounded_sam1(box_threshold: float = 0.2, text_threshold: float = 0.2) -> None:
    ensure_torchvision_nms_stub()
    import torch

    if not SAM1_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing SAM1 checkpoint: {SAM1_CHECKPOINT}")
    import_sam1()
    from segment_anything import SamPredictor, sam_model_registry

    detector_id, processor, detector, device = get_grounding_dino_detector()
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM1_CHECKPOINT)).to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    out_dir = BENCH_ROOT / "method_masks/GroundedSAM1_potted"
    log_rows: list[dict[str, object]] = []
    for record in load_manifest():
        pil_image = Image.open(record.image).convert("RGB")
        boxes, scores = detect_grounding_boxes(
            processor=processor,
            detector=detector,
            device=device,
            pil_image=pil_image,
            prompt=GROUNDING_PROMPT,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        image = read_rgb(record.image)
        if len(boxes) == 0:
            mask = np.zeros(image.shape[:2], dtype=bool)
            chosen = ""
            best_score: float | str = ""
        else:
            union = np.zeros(image.shape[:2], dtype=bool)
            predictor.set_image(image)
            with torch.inference_mode():
                for box in boxes:
                    masks, mask_scores, _ = predictor.predict(
                        box=box.astype(np.float32),
                        multimask_output=True,
                    )
                    union |= masks[int(np.argmax(mask_scores))] > 0
            mask = union
            chosen = ";".join(" ".join(f"{float(x):.1f}" for x in box) for box in boxes)
            best_score = float(scores.max())
        out_path = out_dir / f"mask_{record.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "sample": record.sample,
                "frame": record.frame,
                "image": str(record.image),
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
            "sample",
            "frame",
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


def run_grounded_sam2(box_threshold: float = 0.2, text_threshold: float = 0.2) -> None:
    ensure_torchvision_nms_stub()
    import torch

    predictor, sam_device = get_sam2_predictor()
    detector_id, processor, detector, detector_device = get_grounding_dino_detector()
    out_dir = BENCH_ROOT / "method_masks/GroundedSAM2_potted"
    log_rows: list[dict[str, object]] = []
    for record in load_manifest():
        pil_image = Image.open(record.image).convert("RGB")
        boxes, scores = detect_grounding_boxes(
            processor=processor,
            detector=detector,
            device=detector_device,
            pil_image=pil_image,
            prompt=GROUNDING_PROMPT,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        image = read_rgb(record.image)
        if len(boxes) == 0:
            mask = np.zeros(image.shape[:2], dtype=bool)
            chosen = ""
            best_score: float | str = ""
        else:
            union = np.zeros(image.shape[:2], dtype=bool)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=sam_device.startswith("cuda"),
            ):
                predictor.set_image(image)
                for box in boxes:
                    masks, mask_scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
                    union |= masks[int(np.argmax(mask_scores))] > 0
            mask = union
            chosen = ";".join(" ".join(f"{float(x):.1f}" for x in box) for box in boxes)
            best_score = float(scores.max())
        out_path = out_dir / f"mask_{record.stem}.png"
        save_mask(mask, out_path)
        log_rows.append(
            {
                "sample": record.sample,
                "frame": record.frame,
                "image": str(record.image),
                "mask": str(out_path),
                "detector_id": detector_id,
                "sam_model": "facebook/sam2.1-hiera-large",
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
            "sample",
            "frame",
            "image",
            "mask",
            "detector_id",
            "sam_model",
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


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
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
    records = load_manifest()
    frame_rows: list[dict[str, object]] = []
    tp = fp = fn = tn = 0
    areas: list[float] = []
    comps: list[int] = []
    temporal_ious: list[float] = []
    hd95_values: list[float] = []
    bf_values: list[float] = []
    missing: list[str] = []
    prev_by_sample: dict[str, np.ndarray] = {}

    for record in records:
        stem = record.stem
        pred_path = mask_dir / f"mask_{stem}.png"
        gt = read_mask(record.gt_mask)
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
        temporal_iou: float | str = ""
        prev_pred = prev_by_sample.get(record.sample)
        if prev_pred is not None:
            union = np.logical_or(prev_pred, pred).sum()
            temporal_iou = safe_div(int(np.logical_and(prev_pred, pred).sum()), int(union))
            temporal_ious.append(float(temporal_iou))
        prev_by_sample[record.sample] = pred
        outside_nonblack = safe_div(c[1], c[1] + c[3])
        leakage_energy = safe_div(c[1], gt.size)
        areas.append(area_ratio)
        comps.append(comp)
        hd95_values.append(h)
        bf_values.append(b)
        frame_rows.append(
            {
                "method": method_name,
                "sample": record.sample,
                "frame_id": record.frame,
                "frame": stem,
                "mask_path": str(pred_path),
                "gt_path": str(record.gt_mask),
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
    gt_shape = read_mask(records[0].gt_mask).shape if records else (0, 0)
    summary = {
        "method": method_name,
        "display_name": DISPLAY_NAME.get(method_name, method_name),
        "protocol": METHOD_PROTOCOL.get(method_name, ""),
        "main_table_candidate": "yes" if method_name in MAIN_TABLE_METHODS else "no",
        "mask_dir": str(mask_dir),
        "target_definition": TARGET_DEFINITION,
        "sample_count": len({r.sample for r in records}),
        "gt_frames": len(records),
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
    prepare_dataset()
    method_root = BENCH_ROOT / "method_masks"
    method_dirs = [method_root / method for method in METHOD_ORDER if (method_root / method).is_dir()]
    extra_dirs = sorted([p for p in method_root.iterdir() if p.is_dir() and p.name not in METHOD_ORDER]) if method_root.exists() else []
    method_dirs.extend(extra_dirs)
    summary_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for method_dir in method_dirs:
        summary, rows = evaluate_method(method_dir.name, method_dir, boundary_tol)
        summary_rows.append(summary)
        frame_rows.extend(rows)

    summary_fields = [
        "method",
        "display_name",
        "protocol",
        "main_table_candidate",
        "mask_dir",
        "target_definition",
        "sample_count",
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
        "sample",
        "frame_id",
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


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_SMALL = font(18)
FONT_MED = font(24)
FONT_MED_BOLD = font(24, bold=True)


def contours(mask: np.ndarray, thickness: int = 4) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, found, -1, (255, 255, 255), thickness)
    return out > 0


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contour(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 4) -> np.ndarray:
    out = rgb.copy()
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, found, -1, color[::-1], thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    height = max(1, int(round(h * width / w)))
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, width: int) -> np.ndarray:
    h, w = mask.shape[:2]
    height = max(1, int(round(h * width / w)))
    out = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return out > 0


def add_label(rgb: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    img = Image.fromarray(rgb)
    pad = 16
    label_h = 72 if subtitle else 48
    canvas = Image.new("RGB", (img.width, img.height + label_h), (255, 255, 255))
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), title, fill=TEXT_COLOR, font=FONT_MED_BOLD)
    if subtitle:
        draw.text((pad, 40), subtitle, fill=(88, 88, 88), font=FONT_SMALL)
    return canvas


def make_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, method: str) -> np.ndarray:
    out = blend_mask(rgb, pred, METHOD_COLORS.get(method, (92, 142, 205)), 0.42)
    out = draw_contour(out, gt, GT_COLOR, thickness=5)
    out = draw_contour(out, pred, METHOD_COLORS.get(method, (92, 142, 205)), thickness=3)
    return out


def make_gt_overlay(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = blend_mask(rgb, gt, (92, 181, 125), 0.28)
    return draw_contour(out, gt, GT_COLOR, thickness=5)


def make_error_map(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = (rgb.astype(np.float32) * 0.42 + 255 * 0.58).astype(np.uint8)
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    out = blend_mask(base, tp, TP_COLOR, 0.38)
    out = blend_mask(out, fp, FP_COLOR, 0.78)
    out = blend_mask(out, fn, FN_COLOR, 0.78)
    out = draw_contour(out, gt, GT_COLOR, thickness=3)
    return out


def make_contact_sheet(images: list[Image.Image], cols: int, gap: int = 14) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap), (255, 255, 255))
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = gap + col * (cell_w + gap) + (cell_w - img.width) // 2
        y = gap + row * (cell_h + gap) + (cell_h - img.height) // 2
        canvas.paste(img, (x, y))
    return canvas


def save_image(img: Image.Image, path: Path, *, tiff: bool = False, pdf: bool = False) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path]
    img.save(path, dpi=(600, 600))
    if tiff:
        tif = path.with_suffix(".tif")
        img.save(tif, dpi=(600, 600), compression="tiff_lzw")
        written.append(tif)
    if pdf:
        pdf_path = path.with_suffix(".pdf")
        img.save(pdf_path, resolution=600.0)
        written.append(pdf_path)
    return written


def load_frame_metrics() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(BENCH_ROOT / "metrics/frame_metrics.csv")
    return {(row["method"], row["frame"]): row for row in rows}


def load_summary() -> dict[str, dict[str, str]]:
    rows = read_csv(BENCH_ROOT / "metrics/summary_metrics.csv")
    return {row["method"]: row for row in rows}


def metric_subtitle(metric: dict[str, str]) -> str:
    return (
        f"F1 {float(metric['f1']):.3f} | mIoU {float(metric['miou']):.3f} | "
        f"leak {float(metric['leakage_energy']):.4f}"
    )


def generate_visual_assets(panel_width: int = 520) -> None:
    evaluate_all()
    out_root = BENCH_ROOT / "visual_assets"
    metrics = load_frame_metrics()
    summary = load_summary()
    asset_rows: list[dict[str, object]] = []
    records = load_manifest()
    present_methods = [method for method in METHOD_ORDER if method in summary]

    # Source data archive.
    source_data = out_root / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)
    for src in [BENCH_ROOT / "manifest.csv", BENCH_ROOT / "gt_shape_summary.csv", BENCH_ROOT / "metrics/summary_metrics.csv", BENCH_ROOT / "metrics/frame_metrics.csv", BENCH_ROOT / "metrics/summary_metrics.json"]:
        dst = source_data / src.name
        shutil.copy2(src, dst)
        asset_rows.append({"asset_type": "source_data", "method": "all", "frame": "all", "path": str(dst)})

    paper_rows = []
    for method in present_methods:
        row = summary[method]
        paper_rows.append(
            {
                "method": DISPLAY_NAME.get(method, method),
                "method_dir": method,
                "protocol": row["protocol"],
                "main_table_candidate": row["main_table_candidate"],
                "target": TARGET_DEFINITION,
                "sample_count": row["sample_count"],
                "gt_frames": row["gt_frames"],
                "eval_frames": row["eval_frames"],
                "F1": f"{float(row['f1']):.4f}",
                "mIoU": f"{float(row['miou']):.4f}",
                "HD95_px": f"{float(row['hd95_px']):.2f}",
                "boundary_F1": f"{float(row['boundary_f1']):.4f}",
                "leakage_energy": f"{float(row['leakage_energy']):.6f}",
                "precision": f"{float(row['precision']):.4f}",
                "recall": f"{float(row['recall']):.4f}",
            }
        )
    write_csv(
        source_data / "paper_summary_metrics.csv",
        [
            "method",
            "method_dir",
            "protocol",
            "main_table_candidate",
            "target",
            "sample_count",
            "gt_frames",
            "eval_frames",
            "F1",
            "mIoU",
            "HD95_px",
            "boundary_F1",
            "leakage_energy",
            "precision",
            "recall",
        ],
        paper_rows,
    )
    asset_rows.append({"asset_type": "paper_table", "method": "all", "frame": "summary", "path": str(source_data / "paper_summary_metrics.csv")})

    for record in records:
        rgb_full = read_rgb(record.image)
        gt_full = read_mask(record.gt_mask, rgb_full.shape[:2])
        rgb = resize_rgb(rgb_full, panel_width)
        gt = resize_mask(gt_full, panel_width)
        panels = [add_label(make_gt_overlay(rgb, gt), "Image + GT", record.stem)]
        gt_path = out_root / "gt_overlays" / f"{record.stem}_gt_overlay.png"
        save_image(panels[0], gt_path)
        asset_rows.append({"asset_type": "gt_overlay", "method": "GT", "frame": record.stem, "path": str(gt_path)})
        for method in present_methods:
            pred_path = BENCH_ROOT / "method_masks" / method / f"mask_{record.stem}.png"
            if not pred_path.exists():
                continue
            pred = resize_mask(read_mask(pred_path, rgb_full.shape[:2]), panel_width)
            subtitle = metric_subtitle(metrics[(method, record.stem)])
            overlay = add_label(make_overlay(rgb, gt, pred, method), DISPLAY_NAME.get(method, method), subtitle)
            error = add_label(make_error_map(rgb, gt, pred), f"{DISPLAY_NAME.get(method, method)} error", "red FP | blue FN | green TP")
            overlay_path = out_root / "overlays" / method / f"{record.stem}_{method}_overlay.png"
            error_path = out_root / "error_maps" / method / f"{record.stem}_{method}_error.png"
            save_image(overlay, overlay_path)
            save_image(error, error_path)
            asset_rows.append({"asset_type": "overlay", "method": method, "frame": record.stem, "path": str(overlay_path)})
            asset_rows.append({"asset_type": "error_map", "method": method, "frame": record.stem, "path": str(error_path)})
            panels.append(overlay)
        sheet = make_contact_sheet(panels, cols=3)
        sheet_path = out_root / "contact_sheets" / f"{record.stem}_method_overlay_grid.png"
        for item in save_image(sheet, sheet_path, tiff=True):
            asset_rows.append({"asset_type": "contact_sheet", "method": "all", "frame": record.stem, "path": str(item)})

    metric_panel = make_metric_bar_figure(summary, present_methods)
    for item in save_image(metric_panel, out_root / "figures/Fig_E1_rep4_metric_bars.png", tiff=True, pdf=True):
        asset_rows.append({"asset_type": "metric_bars", "method": "all", "frame": "summary", "path": str(item)})

    write_csv(out_root / "figure_asset_index.csv", ["asset_type", "method", "frame", "path"], asset_rows)
    write_readme(out_root, present_methods, summary)


def make_metric_bar_figure(summary: dict[str, dict[str, str]], methods: list[str]) -> Image.Image:
    width, height = 1700, 960
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    panels = [
        ("F1", "f1", 1.0, False),
        ("mIoU", "miou", 1.0, False),
        ("HD95 px", "hd95_px", max(float(summary[m]["hd95_px"]) for m in methods) * 1.05, True),
        ("Leakage energy", "leakage_energy", max(float(summary[m]["leakage_energy"]) for m in methods) * 1.10, True),
    ]
    positions = [(50, 40), (890, 40), (50, 500), (890, 500)]
    for (title, key, maxv, lower), (x, y) in zip(panels, positions):
        draw.rectangle([x, y, x + 800, y + 400], fill=(255, 255, 255), outline=(226, 226, 226), width=1)
        draw.text((x + 16, y + 12), title, fill=TEXT_COLOR, font=FONT_MED_BOLD)
        draw.text((x + 16, y + 42), "lower better" if lower else "higher better", fill=(100, 100, 100), font=FONT_SMALL)
        bar_x = x + 185
        bar_w = 550
        top = y + 90
        step = max(38, (330 // max(1, len(methods))))
        for idx, method in enumerate(methods):
            value = float(summary[method][key])
            yy = top + idx * step
            color = METHOD_COLORS.get(method, (92, 142, 205))
            bw = 0 if maxv <= 0 else int(round(bar_w * value / maxv))
            draw.text((x + 16, yy - 3), DISPLAY_NAME.get(method, method), fill=TEXT_COLOR, font=FONT_SMALL)
            draw.line([bar_x, yy + 10, bar_x + bar_w, yy + 10], fill=GRID_COLOR, width=1)
            draw.rectangle([bar_x, yy, bar_x + bw, yy + 20], fill=color)
            label = f"{value:.4f}" if value < 0.1 else f"{value:.3f}"
            draw.text((bar_x + bar_w + 10, yy - 2), label, fill=TEXT_COLOR, font=FONT_SMALL)
    return img


def write_readme(out_root: Path, methods: list[str], summary: dict[str, dict[str, str]]) -> None:
    lines = [
        "# 实验一四样本代表集 VFM 横向对比",
        "",
        "## 口径",
        "",
        "- 样本：`KongQueZhuYu`、`DouBanLv1`、`ChangShouHua2`、`CaoMei1`。",
        "- 帧数：4 个样本 x 5 帧 = 20 帧 GT。",
        f"- GT 定义：{TARGET_DEFINITION}。",
        "- 输出目录全部位于本工作区内，未覆盖旧 S23 六帧结果。",
        "",
        "## 汇总结果",
        "",
        "| Method | F1 | mIoU | HD95 px | Boundary F1 | Leakage energy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = summary[method]
        lines.append(
            f"| {DISPLAY_NAME.get(method, method)} | {float(row['f1']):.4f} | "
            f"{float(row['miou']):.4f} | {float(row['hd95_px']):.2f} | "
            f"{float(row['boundary_f1']):.4f} | {float(row['leakage_energy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            "- `manifest.csv`：四样本 20 帧图像、GT JSON 和 GT mask 索引。",
            "- `gt_shape_summary.csv`：每个闭合 linestrip 的面积和 bbox，用于审计花盆/主体轮廓。",
            "- `method_masks/`：各方法二值预测 mask。",
            "- `metrics/summary_metrics.csv` 与 `metrics/frame_metrics.csv`：汇总与逐帧指标。",
            "- `visual_assets/`：overlay、error map、contact sheet、指标柱状图和 source data。",
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def clean_outputs() -> None:
    if BENCH_ROOT.exists():
        shutil.rmtree(BENCH_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment one representative-4 potted-plant VFM benchmark.")
    parser.add_argument(
        "--step",
        choices=[
            "prepare",
            "import_existing",
            "import_rap",
            "clipseg",
            "grounded_sam1",
            "grounded_sam2",
            "florence2",
            "sam2_oracle",
            "evaluate",
            "visuals",
            "clean",
        ],
        required=True,
    )
    parser.add_argument("--clipseg-threshold", type=float, default=0.5)
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--boundary-tol", type=int, default=3)
    parser.add_argument("--panel-width", type=int, default=520)
    args = parser.parse_args()

    start = time.time()
    if args.step == "clean":
        clean_outputs()
    elif args.step == "prepare":
        prepare_dataset()
    elif args.step == "import_existing":
        import_existing_masks()
    elif args.step == "import_rap":
        import_rap_runs()
    elif args.step == "clipseg":
        run_clipseg(args.clipseg_threshold)
    elif args.step == "grounded_sam1":
        run_grounded_sam1(args.box_threshold, args.text_threshold)
    elif args.step == "grounded_sam2":
        run_grounded_sam2(args.box_threshold, args.text_threshold)
    elif args.step == "florence2":
        run_florence2()
    elif args.step == "sam2_oracle":
        run_sam2_oracle_box()
    elif args.step == "evaluate":
        evaluate_all(args.boundary_tol)
    elif args.step == "visuals":
        generate_visual_assets(args.panel_width)
    print(f"{args.step} finished in {time.time() - start:.1f}s; output={BENCH_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
