#!/usr/bin/env python3
"""
Test script for instant-ngp environment setup.
Verifies that all dependencies are correctly installed.
"""

import sys
print("=" * 60)
print("Testing instant-ngp Python Environment")
print("=" * 60)

# Test 1: Check Python version
print(f"\n✓ Python version: {sys.version}")

# Test 2: Import required packages
dependencies = [
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('imageio', 'ImageIO'),
    ('cv2', 'OpenCV'),
    ('tqdm', 'TQDM'),
    ('commentjson', 'CommentJSON'),
    ('pyquaternion', 'PyQuaternion'),
    ('pybind11', 'PyBind11'),
]

print("\nTesting Python dependencies:")
for module, name in dependencies:
    try:
        __import__(module)
        print(f"  ✓ {name} imported successfully")
    except ImportError as e:
        print(f"  ✗ {name} import failed: {e}")
        sys.exit(1)

# Test 3: Check numpy version
import numpy as np
print(f"\n✓ NumPy version: {np.__version__}")

# Test 4: Check if data directory exists
import os
data_dir = "/data/fj/instant-ngp/data"
if os.path.exists(data_dir):
    print(f"✓ Data directory exists: {data_dir}")
    print(f"  Available datasets:")
    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        if os.path.isdir(dataset_path):
            print(f"    - {dataset}")
else:
    print(f"✗ Data directory not found: {data_dir}")

# Test 5: Test importing project scripts
print("\nTesting project scripts:")
try:
    sys.path.insert(0, '/data/fj/instant-ngp/scripts')
    from common import *
    print("  ✓ common.py imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import common.py: {e}")

# Test 6: Check if pyngp module is available (requires compilation)
print("\nChecking pyngp module (requires compilation):")
try:
    import pyngp as ngp
    print("  ✓ pyngp module is available")
except ImportError:
    print("  ⚠ pyngp module not found - project needs to be compiled")
    print("    To compile, you need:")
    print("    1. CMake 3.21 or higher")
    print("    2. CUDA 10.2 or higher (detected: 12.6)")
    print("    3. C++14 compiler (detected: GCC 13.3.0)")
    print("    Then run:")
    print("      cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo")
    print("      cmake --build build --config RelWithDebInfo -j")

print("\n" + "=" * 60)
print("Environment test completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Install CMake: sudo apt-get install cmake")
print("2. Compile the project (see commands above)")
print("3. Run: ./instant-ngp data/nerf/fox")
print("   Or use Python: python scripts/run.py --scene data/nerf/fox")
