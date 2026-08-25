"""
用清理后的点云重跑 BaiZhang 的 SuGaR 管线（Steps 2-5）。
前提: vanilla 3DGS 的 point_cloud.ply 已替换为 clean_v2 版本。
"""
import os
import sys
import time
import shutil

SUGAR_ROOT = r"D:\CAAS\SuGaR-main"
os.chdir(SUGAR_ROOT)
sys.path.insert(0, SUGAR_ROOT)

SCENE_NAME = "BaiZhang"
SCENE_PATH = rf"D:\CAAS\04-COLMAP\{SCENE_NAME}"
CHECKPOINT_PATH = os.path.join(SUGAR_ROOT, "output", "vanilla_gs", SCENE_NAME)
GS_ITERATIONS = 7000
GS_OUT = rf"D:\CAAS\07-SuGaR-GS\{SCENE_NAME}"
MESH_OUT = rf"D:\CAAS\07-SuGaR-Mesh\{SCENE_NAME}"

print("=" * 60)
print(f"重跑 SuGaR 管线 (使用清废光后的点云)")
print(f"场景: {SCENE_NAME}")
print(f"点云: {CHECKPOINT_PATH}")
print("=" * 60)

# 确认点云已被替换
ply_path = os.path.join(CHECKPOINT_PATH, "point_cloud", f"iteration_{GS_ITERATIONS}", "point_cloud.ply")
orig_path = ply_path.replace("point_cloud.ply", "point_cloud_original.ply")
if os.path.exists(orig_path):
    clean_size = os.path.getsize(ply_path) / 1024 / 1024
    orig_size = os.path.getsize(orig_path) / 1024 / 1024
    print(f"✓ 点云已替换: {orig_size:.1f}MB -> {clean_size:.1f}MB")
else:
    print("⚠ 未找到 _original.ply 备份，可能尚未执行清废光！")

# --- 运行 SuGaR 管线 (Steps 2-5) ---
print("\n[Steps 2-5] SuGaR: coarse → mesh → refine → texture")
t0 = time.time()

# 使用 train.py 来运行完整 SuGaR 管线
from train import *  # noqa
import argparse

# 构造参数
sugar_args = argparse.Namespace(
    scene_path=SCENE_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    iteration_to_load=GS_ITERATIONS,
    regularization_type="dn_consistency",
    surface_level=0.3,
    n_vertices_in_mesh=1_000_000,
    n_gaussians_per_surface_triangle=1,
    refinement_iterations=15_000,
    export_obj=True,
    export_ply=True,
    square_size=8,
    eval=False,
    gpu=0,
    postprocess_mesh=False,
    postprocess_density_threshold=0.1,
    postprocess_iterations=5,
    mesh_output_dir=None,
)

# 这里直接调用 subprocess 更安全（避免 import 冲突）
import subprocess

cmd = [
    sys.executable,
    os.path.join(SUGAR_ROOT, "train.py"),
    "-s", SCENE_PATH,
    "-c", CHECKPOINT_PATH,
    "-i", str(GS_ITERATIONS),
    "-r", "dn_consistency",
    "-l", "0.3",
    "-v", "1000000",
    "-g", "1",
    "-f", "15000",
    "-t", "True",
    "--square_size", "8",
    "--export_ply", "True",
    "--eval", "False",
    "--gpu", "0",
]

print(f"运行命令: {' '.join(cmd[:6])} ...")
proc = subprocess.Popen(
    cmd,
    cwd=SUGAR_ROOT,
    stdout=sys.stdout,
    stderr=subprocess.STDOUT,
)
proc.wait()

elapsed = time.time() - t0
if proc.returncode != 0:
    print(f"\n✗ SuGaR 管线失败 (exit {proc.returncode}) after {elapsed:.0f}s")
    sys.exit(1)

print(f"\n✓ SuGaR 管线完成，耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")

# --- 收集输出 ---
print("\n[Collect] 收集输出...")
sugar_output = os.path.join(SUGAR_ROOT, "output")
os.makedirs(GS_OUT, exist_ok=True)
os.makedirs(MESH_OUT, exist_ok=True)

# Copy SuGaR intermediates
for subdir in ["coarse", "coarse_mesh", "refined"]:
    src = os.path.join(sugar_output, subdir, SCENE_NAME)
    if os.path.isdir(src):
        dst = os.path.join(GS_OUT, subdir)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  Copied {subdir}")

# Copy refined mesh
refined_mesh_src = os.path.join(sugar_output, "refined_mesh", SCENE_NAME)
if os.path.isdir(refined_mesh_src):
    for f in os.listdir(refined_mesh_src):
        src_file = os.path.join(refined_mesh_src, f)
        dst_file = os.path.join(MESH_OUT, f)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"  Copied {f} -> {MESH_OUT}")
    dst_gs = os.path.join(GS_OUT, "refined_mesh")
    if os.path.exists(dst_gs):
        shutil.rmtree(dst_gs)
    shutil.copytree(refined_mesh_src, dst_gs)

# Copy GS model checkpoint
gs_dst = os.path.join(GS_OUT, "vanilla_gs")
if os.path.exists(gs_dst):
    shutil.rmtree(gs_dst)
shutil.copytree(CHECKPOINT_PATH, gs_dst)
print(f"  Copied vanilla_gs")

print("\n" + "=" * 60)
print(f"BaiZhang 重跑完成! (使用清废光后的点云)")
print(f"  GS 输出: {GS_OUT}")
print(f"  Mesh 输出: {MESH_OUT}")
print("=" * 60)
