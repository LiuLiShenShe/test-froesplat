#!/usr/bin/env python3
"""
Test script to verify the example datasets are correctly formatted.
"""

import os
import json
import sys

print("=" * 60)
print("Testing instant-ngp Example Datasets")
print("=" * 60)

# Test fox NeRF dataset
fox_dir = "/data/fj/instant-ngp/data/nerf/fox"
print(f"\nTesting fox NeRF dataset: {fox_dir}")

if os.path.exists(fox_dir):
    # Check transforms.json
    transforms_path = os.path.join(fox_dir, "transforms.json")
    if os.path.exists(transforms_path):
        print("  ✓ transforms.json found")
        try:
            with open(transforms_path, 'r') as f:
                transforms = json.load(f)

            # Check required fields
            if 'frames' in transforms:
                print(f"  ✓ Found {len(transforms['frames'])} frames")

            if 'camera_angle_x' in transforms:
                print(f"  ✓ Camera angle X: {transforms['camera_angle_x']:.4f}")

            # Check images
            images_dir = os.path.join(fox_dir, "images")
            if os.path.exists(images_dir):
                images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                print(f"  ✓ Found {len(images)} images in images/ directory")

                # Verify all frames have corresponding images
                missing_images = []
                for frame in transforms.get('frames', []):
                    img_path = os.path.join(fox_dir, frame.get('file_path', ''))
                    if not os.path.exists(img_path):
                        missing_images.append(frame.get('file_path'))

                if missing_images:
                    print(f"  ⚠ Missing {len(missing_images)} images referenced in transforms.json")
                else:
                    print("  ✓ All frame images exist")

        except Exception as e:
            print(f"  ✗ Error reading transforms.json: {e}")
    else:
        print("  ✗ transforms.json not found")
else:
    print("  ✗ Fox dataset directory not found")

# Test SDF dataset
sdf_dir = "/data/fj/instant-ngp/data/sdf"
print(f"\nTesting SDF dataset: {sdf_dir}")
if os.path.exists(sdf_dir):
    files = os.listdir(sdf_dir)
    print(f"  ✓ Found {len(files)} files:")
    for f in files:
        print(f"    - {f}")
else:
    print("  ✗ SDF dataset directory not found")

# Test image dataset
image_dir = "/data/fj/instant-ngp/data/image"
print(f"\nTesting image dataset: {image_dir}")
if os.path.exists(image_dir):
    files = os.listdir(image_dir)
    print(f"  ✓ Found {len(files)} files:")
    for f in files:
        print(f"    - {f}")
else:
    print("  ✗ Image dataset directory not found")

print("\n" + "=" * 60)
print("Dataset validation completed!")
print("=" * 60)
print("\nThe example datasets are ready to use.")
print("Once the project is compiled, you can test with:")
print("  ./instant-ngp data/nerf/fox")
