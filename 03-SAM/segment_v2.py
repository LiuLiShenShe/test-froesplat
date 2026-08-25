"""
SAM3 Segmentation Script v2 - with morphological post-processing
Segments potted plants and blue calibration blocks from images.
Cleans up masks to remove disconnected fragments and fill holes.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion, binary_closing, binary_opening

sys.path.insert(0, r"D:\CAAS\sam3-main")

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============ Configuration ============
CHECKPOINT_PATH = r"D:\CAAS\sam3\sam3.pt"
INPUT_BASE_DIR = r"D:\CAAS\02-FFT"
OUTPUT_BASE_DIR = r"D:\CAAS\03-SAM"

# Text prompts
PLANT_PROMPT = "a potted plant"
BLOCK_PROMPT = "a small blue square block"

# Confidence thresholds
PLANT_CONFIDENCE = 0.3
BLOCK_CONFIDENCE = 0.3

# Post-processing parameters
CLOSING_KERNEL_SIZE = 15  # morphological closing kernel size
FILL_HOLES = True         # fill holes inside mask


def load_model():
    """Load SAM3 model with local checkpoint."""
    print("Loading SAM3 model...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT_PATH,
        load_from_HF=False,
        device="cuda",
    )
    print("Model loaded successfully!")
    return model


def keep_largest_component(mask):
    """Keep only the largest connected component in a binary mask."""
    labeled, n_regions = ndimage.label(mask)
    if n_regions <= 1:
        return mask
    region_sizes = ndimage.sum(mask, labeled, range(1, n_regions + 1))
    largest_id = np.argmax(region_sizes) + 1
    return (labeled == largest_id)


def postprocess_plant_mask(mask, closing_size=15, fill_holes=True):
    """
    Post-process the PLANT mask only (before merging with block):
    1. Keep only the largest connected component (= target plant, discard other plants)
    2. Morphological closing to smooth edges
    3. Fill holes inside the mask
    """
    # Step 1: Keep only the largest connected component
    # This removes other plants that were captured in background
    mask = keep_largest_component(mask)

    # Step 2: Morphological closing to smooth edges and bridge small leaf gaps
    if closing_size > 0:
        struct = np.ones((closing_size, closing_size))
        mask = binary_closing(mask, structure=struct, iterations=1)

    # Step 3: Fill holes
    if fill_holes:
        mask = binary_fill_holes(mask)

    return mask


def postprocess_block_mask(mask, min_size_ratio=0.0001):
    """
    Post-process the BLOCK mask:
    1. Remove noise (tiny regions)
    2. Keep valid block regions
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    min_size = int(total_pixels * min_size_ratio)

    labeled, n_regions = ndimage.label(mask)
    if n_regions > 0:
        region_sizes = ndimage.sum(mask, labeled, range(1, n_regions + 1))
        for region_id, size in enumerate(region_sizes, start=1):
            if size < min_size:
                mask[labeled == region_id] = False
    return mask


def segment_image(processor, image_path, plant_prompt, block_prompt,
                  plant_conf=0.3, block_conf=0.3):
    """
    Segment one image with two prompts.
    For plant: keep only the LARGEST connected component (target plant).
    For block: keep all valid detections.
    Then merge both masks.
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # --- Segment plant + pot ---
    processor.confidence_threshold = plant_conf
    state = processor.set_image(image)
    state = processor.set_text_prompt(state=state, prompt=plant_prompt)

    plant_masks = state.get("masks", None)
    plant_scores = state.get("scores", None)

    plant_mask = np.zeros((h, w), dtype=bool)
    n_plant = 0
    if plant_masks is not None and len(plant_masks) > 0:
        for i in range(len(plant_masks)):
            mask_np = plant_masks[i].squeeze().cpu().numpy().astype(bool)
            plant_mask = plant_mask | mask_np
            n_plant += 1

    # Post-process plant: keep only the largest component (= target plant)
    raw_plant_coverage = plant_mask.sum() / plant_mask.size * 100
    plant_mask = postprocess_plant_mask(
        plant_mask,
        closing_size=CLOSING_KERNEL_SIZE,
        fill_holes=FILL_HOLES,
    )
    clean_plant_coverage = plant_mask.sum() / plant_mask.size * 100

    # --- Segment blue block ---
    processor.confidence_threshold = block_conf
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=block_prompt)

    block_masks = state.get("masks", None)
    block_scores = state.get("scores", None)

    block_mask = np.zeros((h, w), dtype=bool)
    n_block = 0
    if block_masks is not None and len(block_masks) > 0:
        for i in range(len(block_masks)):
            mask_np = block_masks[i].squeeze().cpu().numpy().astype(bool)
            block_mask = block_mask | mask_np
            n_block += 1

    # Post-process block: remove noise
    block_mask = postprocess_block_mask(block_mask)

    # --- Merge plant + block ---
    combined_mask = plant_mask | block_mask

    # Count final connected regions
    labeled_final, n_final = ndimage.label(combined_mask)
    processed_coverage = combined_mask.sum() / combined_mask.size * 100

    info = {
        "n_plant_detections": n_plant,
        "n_block_detections": n_block,
        "plant_scores": [s.item() for s in plant_scores] if plant_scores is not None and len(plant_scores) > 0 else [],
        "block_scores": [s.item() for s in block_scores] if block_scores is not None and len(block_scores) > 0 else [],
        "raw_plant_coverage_pct": round(raw_plant_coverage, 2),
        "clean_plant_coverage_pct": round(clean_plant_coverage, 2),
        "processed_coverage_pct": round(processed_coverage, 2),
        "final_regions": n_final,
    }

    return combined_mask, info


def save_results(image_path, combined_mask, output_dir, filename):
    """Save binary mask and cropped image."""
    os.makedirs(output_dir, exist_ok=True)

    # Save binary mask
    mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
    mask_path = os.path.join(output_dir, f"mask_{filename}")
    mask_img.save(mask_path)

    # Save cropped original (bg = black)
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original)
    cropped = original_np.copy()
    cropped[~combined_mask] = 0
    cropped_img = Image.fromarray(cropped)
    cropped_path = os.path.join(output_dir, f"crop_{filename}")
    cropped_img.save(cropped_path)

    return mask_path, cropped_path


def process_folder(model, folder_name, limit=None, force=False):
    """Process all images in a folder."""
    input_dir = os.path.join(INPUT_BASE_DIR, folder_name)
    output_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    # Skip if already completed (log exists with matching count)
    log_path = os.path.join(output_dir, "segmentation_log.json")
    if not force and os.path.exists(log_path):
        with open(log_path, 'r') as f:
            existing_log = json.load(f)
        if len(existing_log) >= len(images):
            print(f"\n[SKIP] {folder_name}: already processed ({len(existing_log)} images)")
            return
    
    if limit:
        images = images[:limit]

    print(f"\nProcessing {folder_name}: {len(images)} images")
    print(f"  Plant prompt: '{PLANT_PROMPT}' (conf={PLANT_CONFIDENCE})")
    print(f"  Block prompt: '{BLOCK_PROMPT}' (conf={BLOCK_CONFIDENCE})")
    print(f"  Post-processing: closing={CLOSING_KERNEL_SIZE}, keep_largest_plant=True, fill_holes={FILL_HOLES}")

    processor = Sam3Processor(model, confidence_threshold=PLANT_CONFIDENCE)

    log = []
    t_start = time.time()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, img_name in enumerate(images):
            img_path = os.path.join(input_dir, img_name)
            t0 = time.time()

            combined_mask, info = segment_image(
                processor, img_path,
                PLANT_PROMPT, BLOCK_PROMPT,
                PLANT_CONFIDENCE, BLOCK_CONFIDENCE
            )

            mask_path, crop_path = save_results(
                img_path, combined_mask, output_dir,
                img_name.replace('.jpg', '.png')
            )

            elapsed = time.time() - t0
            info["image"] = img_name
            info["time_sec"] = round(elapsed, 2)
            log.append(info)

            print(f"  [{idx+1}/{len(images)}] {img_name}: "
                  f"plants={info['n_plant_detections']}, "
                  f"blocks={info['n_block_detections']}, "
                  f"regions={info['final_regions']}, "
                  f"plant={info['raw_plant_coverage_pct']:.1f}%->{info['clean_plant_coverage_pct']:.1f}%, "
                  f"total={info['processed_coverage_pct']:.1f}%, "
                  f"time={elapsed:.1f}s")

    total_time = time.time() - t_start
    print(f"\nDone! Total: {total_time:.1f}s, Avg: {total_time/len(images):.1f}s/img")

    log_path = os.path.join(output_dir, "segmentation_log.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {log_path}")


def get_all_folders():
    """Get all subfolders in INPUT_BASE_DIR that contain .jpg images."""
    folders = []
    for name in sorted(os.listdir(INPUT_BASE_DIR)):
        folder_path = os.path.join(INPUT_BASE_DIR, name)
        if os.path.isdir(folder_path):
            jpgs = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if jpgs:
                folders.append(name)
    return folders


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SAM3 Segmentation")
    parser.add_argument("--folder", type=str, default=None,
                        help="Process specific folder only (e.g. BaiZhang)")
    parser.add_argument("--all", action="store_true",
                        help="Process all folders")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images per folder")
    args = parser.parse_args()

    model = load_model()

    if args.folder:
        folders = [args.folder]
    elif args.all:
        folders = get_all_folders()
    else:
        folders = get_all_folders()

    print(f"\nFolders to process: {folders}")
    print(f"Total: {len(folders)} folders\n")

    for i, folder in enumerate(folders):
        print(f"\n{'='*60}")
        print(f"  Folder [{i+1}/{len(folders)}]: {folder}")
        print(f"{'='*60}")
        process_folder(model, folder, limit=args.limit)

    print(f"\n\nAll done! Processed {len(folders)} folders.")
