"""
DeepLabv3+ (ResNet-101) Segmentation Baseline
Deep learning semantic segmentation using pretrained weights (no finetuning).

Uses torchvision's DeepLabv3 with ResNet-101 backbone pretrained on COCO.
Pascal VOC class 16 = "pottedplant" is used to extract plant regions.
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
from torchvision import transforms

# ============ Configuration ============
INPUT_BASE_DIR = r"D:\CAAS\02-FFT"
OUTPUT_BASE_DIR = r"D:\CAAS\03-DeepLabv3+  U-Net"

# Pascal VOC class indices for plants and related objects
# 0: background, 16: pottedplant
# We'll also consider class 6 (bus) -> NO, only pottedplant
PLANT_CLASSES = [16]  # pottedplant

# Additional trick: also detect class 7 (car->no), 9 (chair) if plant is on chair
# For robustness, only use class 16 (pottedplant)

# Morphological post-processing
CLOSING_KERNEL_SIZE = 15
FILL_HOLES = True
MIN_REGION_RATIO = 0.001

# If pottedplant detection fails, fall back to any non-background class
FALLBACK_TO_ANY = True


def load_model():
    """Load pretrained DeepLabv3+ (ResNet-101) model."""
    print("Loading DeepLabv3+ (ResNet-101) pretrained on COCO...")
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    print("Model loaded successfully!")
    return model


def get_transform():
    """Get preprocessing transform for DeepLabv3+."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])


def keep_largest_component(mask_u8):
    """Keep only the largest connected component. Input/output: uint8 0/255."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argmax(areas) + 1
    return ((labels == largest) * 255).astype(np.uint8)


def postprocess_mask(mask_bool, closing_size=15, fill_holes=True, min_region_ratio=0.001):
    """Post-process binary mask with morphological operations using OpenCV."""
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    total_pixels = mask_u8.size
    min_size = int(total_pixels * min_region_ratio)

    # Morphological closing
    if closing_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # Fill holes using flood fill
    if fill_holes:
        h, w = mask_u8.shape
        filled = mask_u8.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(filled, ff_mask, (0, 0), 255)
        mask_u8 = mask_u8 | cv2.bitwise_not(filled)

    # Remove small components  
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels > 1:
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask_u8[labels == i] = 0

    return mask_u8


def segment_image(model, preprocess, image_path):
    """Segment a single image using DeepLabv3+."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred = output.argmax(1).squeeze().cpu().numpy()

    # Extract plant mask (pottedplant = class 16)
    plant_mask = np.zeros((h, w), dtype=bool)
    detected_classes = set(np.unique(pred).tolist())
    
    for cls_id in PLANT_CLASSES:
        if cls_id in detected_classes:
            plant_mask |= (pred == cls_id)

    n_plant_pixels = plant_mask.sum()
    
    # Fallback: if no plant detected, try using any non-background class as plant
    fallback_used = False
    if n_plant_pixels == 0 and FALLBACK_TO_ANY:
        # Use all non-background predictions as potential plant
        non_bg_mask = pred > 0
        if non_bg_mask.sum() > 0:
            plant_mask = non_bg_mask
            fallback_used = True
            n_plant_pixels = plant_mask.sum()

    raw_coverage = plant_mask.sum() / plant_mask.size * 100

    # Post-process (returns uint8 0/255)
    mask_u8 = postprocess_mask(plant_mask, CLOSING_KERNEL_SIZE, FILL_HOLES, MIN_REGION_RATIO)

    # Keep largest component
    mask_u8 = keep_largest_component(mask_u8)

    clean_coverage = np.count_nonzero(mask_u8) / mask_u8.size * 100

    # Count final regions
    n_final_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    n_final = max(n_final_labels - 1, 0)

    voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    detected_names = [voc_classes[c] for c in detected_classes if c < len(voc_classes)]

    info = {
        "detected_classes": list(detected_classes),
        "detected_class_names": detected_names,
        "fallback_used": fallback_used,
        "raw_coverage_pct": round(raw_coverage, 2),
        "clean_coverage_pct": round(clean_coverage, 2),
        "final_regions": n_final,
    }

    return mask_u8, info


def save_results(image_path, mask_u8, output_dir, filename):
    """Save binary mask and cropped image (matching SAM output format)."""
    os.makedirs(output_dir, exist_ok=True)

    # Save binary mask (already uint8 0/255)
    mask_path = os.path.join(output_dir, f"mask_{filename}")
    cv2.imwrite(mask_path, mask_u8)

    # Save cropped original (bg = black)
    original = cv2.imread(image_path)
    cropped = original.copy()
    cropped[mask_u8 == 0] = 0
    cropped_path = os.path.join(output_dir, f"crop_{filename}")
    cv2.imwrite(cropped_path, cropped)


def process_folder(model, preprocess, folder_name, force=False):
    """Process all images in a folder."""
    input_dir = os.path.join(INPUT_BASE_DIR, folder_name)
    output_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])

    # Skip if already completed
    log_path = os.path.join(output_dir, "segmentation_log.json")
    if not force and os.path.exists(log_path):
        with open(log_path, 'r') as f:
            existing_log = json.load(f)
        if len(existing_log) >= len(images):
            print(f"\n[SKIP] {folder_name}: already processed ({len(existing_log)} images)")
            return

    print(f"\nProcessing {folder_name}: {len(images)} images")

    log = []
    t_start = time.time()
    skipped = 0

    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_dir, img_name)
        out_filename = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
        mask_out = os.path.join(output_dir, f"mask_{out_filename}")
        crop_out = os.path.join(output_dir, f"crop_{out_filename}")

        # Image-level resume: skip if both outputs exist
        if not force and os.path.exists(mask_out) and os.path.exists(crop_out):
            skipped += 1
            # Add placeholder log entry
            log.append({"image": img_name, "skipped": True, "time_sec": 0})
            continue

        t0 = time.time()

        mask_u8, info = segment_image(model, preprocess, img_path)
        save_results(img_path, mask_u8, output_dir, out_filename)

        elapsed = time.time() - t0
        info["image"] = img_name
        info["time_sec"] = round(elapsed, 3)
        log.append(info)

        if (idx + 1) % 20 == 0 or idx == 0 or idx == len(images) - 1:
            print(f"  [{idx+1}/{len(images)}] cov={info['raw_coverage_pct']:.1f}%->{info['clean_coverage_pct']:.1f}% t={elapsed:.2f}s")
            sys.stdout.flush()

    total_time = time.time() - t_start
    print(f"  Done {folder_name}: {total_time:.1f}s ({skipped} skipped, {len(images)-skipped} processed)")
    sys.stdout.flush()

    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {log_path}")


def get_all_folders():
    """Get all subfolders in INPUT_BASE_DIR that contain images."""
    folders = []
    for name in sorted(os.listdir(INPUT_BASE_DIR)):
        folder_path = os.path.join(INPUT_BASE_DIR, name)
        if os.path.isdir(folder_path):
            imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
            if imgs:
                folders.append(name)
    return folders


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepLabv3+ Segmentation Baseline")
    parser.add_argument("--folder", type=str, default=None, help="Process specific folder")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    args = parser.parse_args()

    t_global = time.time()

    model = load_model()
    preprocess = get_transform()

    if args.folder:
        folders = [args.folder]
    else:
        folders = get_all_folders()

    print(f"\nDeepLabv3+ Segmentation Baseline")
    print(f"Input: {INPUT_BASE_DIR}")
    print(f"Output: {OUTPUT_BASE_DIR}")
    print(f"Folders: {len(folders)}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    for i, folder in enumerate(folders):
        print(f"\n{'='*60}")
        print(f"  Folder [{i+1}/{len(folders)}]: {folder}")
        print(f"{'='*60}")
        process_folder(model, preprocess, folder, force=args.force)

    total_global = time.time() - t_global
    print(f"\n\nAll done! Processed {len(folders)} folders in {total_global:.1f}s")

    # Save global summary
    summary = {
        "method": "DeepLabv3+ (ResNet-101) pretrained on COCO",
        "total_folders": len(folders),
        "total_time_sec": round(total_global, 1),
        "parameters": {
            "plant_classes": PLANT_CLASSES,
            "closing_kernel_size": CLOSING_KERNEL_SIZE,
            "fill_holes": FILL_HOLES,
            "min_region_ratio": MIN_REGION_RATIO,
            "fallback_to_any": FALLBACK_TO_ANY,
        }
    }
    summary_path = os.path.join(OUTPUT_BASE_DIR, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
