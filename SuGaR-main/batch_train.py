"""
Batch training script for SuGaR pipeline on all COLMAP scenes.

Pipeline per scene:
  1. Vanilla 3DGS training (7k iterations)
  2. Coarse SuGaR training (dn_consistency regularization)
  3. Coarse mesh extraction
  4. Refined SuGaR training (15k iterations)
  5. Textured mesh extraction (.obj)

Outputs:
  - GS + SuGaR intermediate: D:\CAAS\07-SuGaR-GS\{scene}\
  - Final meshes:            D:\CAAS\07-SuGaR-Mesh\{scene}\
  - Logs:                    D:\CAAS\07-SuGaR-GS\{scene}\log.txt
"""

import os
import sys
import time
import shutil
import logging
import traceback
import subprocess
import atexit
from pathlib import Path

# 清废光模块
from clean_3dgs_ply_v2 import clean_scene_checkpoint

# ── Paths ──────────────────────────────────────────────────────────
COLMAP_ROOT  = r"D:\CAAS\04-COLMAP"
GS_OUT_ROOT  = r"D:\CAAS\07-SuGaR-GS"
MESH_OUT_ROOT = r"D:\CAAS\07-SuGaR-Mesh"
SUGAR_ROOT   = r"D:\CAAS\SuGaR-main"

# ── Training parameters ───────────────────────────────────────────
GS_ITERATIONS       = 7_000       # vanilla 3DGS iterations
REGULARIZATION      = "dn_consistency"
SURFACE_LEVEL       = 0.3
N_VERTICES           = 1_000_000   # high_poly
GAUSSIANS_PER_TRI   = 1           # high_poly
REFINEMENT_ITERS    = 15_000
EXPORT_OBJ          = True
EXPORT_PLY          = True
SQUARE_SIZE          = 8
GPU                  = 0


def get_scene_list():
    """返回 COLMAP_ROOT 下可用场景名称的排序列表。
    跳过无 sparse 数据或图片数不足的场景。"""
    MIN_IMAGES = 10
    scenes = []
    for name in sorted(os.listdir(COLMAP_ROOT)):
        scene_dir = os.path.join(COLMAP_ROOT, name)
        sparse_dir = os.path.join(scene_dir, "sparse", "0")
        images_dir = os.path.join(scene_dir, "images")
        if not os.path.isdir(scene_dir) or not os.path.isdir(sparse_dir):
            continue
        if not os.path.isdir(images_dir):
            continue
        n_imgs = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        if n_imgs < MIN_IMAGES:
            print(f"  跳过 {name}：图片数不足 ({n_imgs} < {MIN_IMAGES})")
            continue
        scenes.append(name)
    return scenes


def setup_logger(scene_name):
    """Create a file+console logger for a scene."""
    log_dir = os.path.join(GS_OUT_ROOT, scene_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")

    logger = logging.getLogger(scene_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


def _run_subprocess_with_live_log(cmd, logger, step_label, cwd=None):
    """运行子进程，实时将 stdout/stderr 写入 logger（文件+控制台）。"""
    logger.info(f"运行命令: {' '.join(cmd[:4])} ...")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or SUGAR_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # 行缓冲
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU)},
    )
    # 实时逐行读取并写入日志
    for line in proc.stdout:
        line = line.rstrip('\n\r')
        if line:
            logger.info(f"  [{step_label}] {line}")
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.error(f"{step_label} FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
        raise RuntimeError(f"{step_label} failed")
    logger.info(f"{step_label} 完成，耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return elapsed


def run_vanilla_gs(scene_path, gs_model_dir, logger):
    """Train vanilla 3DGS for 7k iterations via subprocess."""
    # Check if already completed
    ckpt_path = os.path.join(gs_model_dir, "point_cloud", f"iteration_{GS_ITERATIONS}", "point_cloud.ply")
    if os.path.exists(ckpt_path):
        logger.info(f"[Step 1/5] Vanilla 3DGS - SKIPPED (checkpoint exists: {ckpt_path})")
        return gs_model_dir

    logger.info("=" * 60)
    logger.info(f"[Step 1/5] Vanilla 3DGS training  ({GS_ITERATIONS} iters)")
    logger.info(f"  Source:  {scene_path}")
    logger.info(f"  Output:  {gs_model_dir}")

    cmd = [
        sys.executable,
        os.path.join(SUGAR_ROOT, "gaussian_splatting", "train.py"),
        "-s", scene_path,
        "-m", gs_model_dir,
        "--iterations", str(GS_ITERATIONS),
    ]

    _run_subprocess_with_live_log(cmd, logger, "Step1-3DGS")
    logger.info(f"[Step 1/5] Done")
    return gs_model_dir


def clean_point_cloud(gs_model_dir, logger):
    """Step 1.5: Clean waste light from vanilla 3DGS point cloud."""
    logger.info("=" * 60)
    logger.info(f"[Step 1.5/5] 清废光 (clean_3dgs_ply_v2)")

    # Check if already cleaned (backup original exists)
    orig_backup = os.path.join(gs_model_dir, "point_cloud", f"iteration_{GS_ITERATIONS}",
                               "point_cloud_original.ply")
    if os.path.exists(orig_backup):
        logger.info(f"[Step 1.5/5] SKIPPED (already cleaned, backup exists: {orig_backup})")
        return

    result = clean_scene_checkpoint(gs_model_dir, iteration=GS_ITERATIONS, verbose=True)
    if result is None:
        logger.warning("  PLY not found, skipping clean step")
        return

    n_orig, n_clean = result
    pct = 100 * (n_orig - n_clean) / n_orig
    logger.info(f"  清废光完成: {n_orig} -> {n_clean} (去除 {pct:.1f}%)")


def run_sugar_pipeline(scene_path, gs_model_dir, scene_name, logger):
    """Run steps 2-5: coarse → mesh → refine → textured mesh via train.py."""
    logger.info("=" * 60)
    logger.info("[Steps 2-5] SuGaR pipeline (coarse → mesh → refine → texture)")

    checkpoint_path = gs_model_dir

    cmd = [
        sys.executable,
        os.path.join(SUGAR_ROOT, "train.py"),
        "-s", scene_path,
        "-c", checkpoint_path,
        "-i", str(GS_ITERATIONS),
        "-r", REGULARIZATION,
        "-l", str(SURFACE_LEVEL),
        "-v", str(N_VERTICES),
        "-g", str(GAUSSIANS_PER_TRI),
        "-f", str(REFINEMENT_ITERS),
        "-t", str(EXPORT_OBJ),
        "--square_size", str(SQUARE_SIZE),
        "--export_ply", str(EXPORT_PLY),
        "--eval", "False",
        "--gpu", str(GPU),
    ]

    _run_subprocess_with_live_log(cmd, logger, "Steps2-5-SuGaR")
    logger.info(f"[Steps 2-5] Done")


def collect_outputs(scene_name, gs_model_dir, logger):
    """
    Copy/move outputs to the final directories.

    SuGaR puts intermediate outputs under SUGAR_ROOT/output/:
      - coarse/{scene}/        → coarse SuGaR checkpoints
      - coarse_mesh/{scene}/   → coarse .ply mesh
      - refined/{scene}/       → refined SuGaR checkpoints
      - refined_mesh/{scene}/  → textured .obj mesh

    We collect everything:
      - GS model + SuGaR intermediates → GS_OUT_ROOT/{scene}/
      - Final meshes (.obj, .mtl, .png, .ply) → MESH_OUT_ROOT/{scene}/
    """
    logger.info("=" * 60)
    logger.info("[Collect] Copying outputs to final directories")

    sugar_output = os.path.join(SUGAR_ROOT, "output")
    gs_out = os.path.join(GS_OUT_ROOT, scene_name)
    mesh_out = os.path.join(MESH_OUT_ROOT, scene_name)
    os.makedirs(gs_out, exist_ok=True)
    os.makedirs(mesh_out, exist_ok=True)

    # Copy SuGaR intermediate outputs to GS output
    for subdir in ["coarse", "coarse_mesh", "refined"]:
        src = os.path.join(sugar_output, subdir, scene_name)
        if os.path.isdir(src):
            dst = os.path.join(gs_out, subdir)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"  Copied {src} -> {dst}")

    # Copy refined mesh (textured .obj) to mesh output
    refined_mesh_src = os.path.join(sugar_output, "refined_mesh", scene_name)
    if os.path.isdir(refined_mesh_src):
        for f in os.listdir(refined_mesh_src):
            src_file = os.path.join(refined_mesh_src, f)
            dst_file = os.path.join(mesh_out, f)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                logger.info(f"  Copied {src_file} -> {dst_file}")
        # Also copy to GS output for completeness
        dst_gs = os.path.join(gs_out, "refined_mesh")
        if os.path.exists(dst_gs):
            shutil.rmtree(dst_gs)
        shutil.copytree(refined_mesh_src, dst_gs)

    # Copy GS model checkpoint
    if os.path.isdir(gs_model_dir):
        gs_dst = os.path.join(gs_out, "vanilla_gs")
        if os.path.exists(gs_dst):
            shutil.rmtree(gs_dst)
        shutil.copytree(gs_model_dir, gs_dst)
        logger.info(f"  Copied GS checkpoint -> {gs_dst}")

    logger.info("[Collect] Done")


def cleanup_sugar_output(scene_name, logger):
    """Remove scene-specific intermediate files from SUGAR_ROOT/output/ to save disk space."""
    sugar_output = os.path.join(SUGAR_ROOT, "output")
    for subdir in ["vanilla_gs", "coarse", "coarse_mesh", "refined", "refined_mesh"]:
        src = os.path.join(sugar_output, subdir, scene_name)
        if os.path.isdir(src):
            shutil.rmtree(src)
            logger.info(f"  Cleaned up {src}")


def process_scene(scene_name):
    """Full pipeline for a single scene. Returns (success, timings_dict)."""
    scene_path = os.path.join(COLMAP_ROOT, scene_name)
    gs_model_dir = os.path.join(SUGAR_ROOT, "output", "vanilla_gs", scene_name)

    logger, log_path = setup_logger(scene_name)
    logger.info(f"Processing scene: {scene_name}")
    logger.info(f"Log file: {log_path}")

    timings = {}
    t_total = time.time()

    try:
        # Step 1: Vanilla 3DGS
        t0 = time.time()
        run_vanilla_gs(scene_path, gs_model_dir, logger)
        timings['step1_vanilla_gs'] = time.time() - t0

        # Step 1.5: 清废光
        t0 = time.time()
        clean_point_cloud(gs_model_dir, logger)
        timings['step1_5_clean'] = time.time() - t0

        # Steps 2-5: SuGaR pipeline
        t0 = time.time()
        run_sugar_pipeline(scene_path, gs_model_dir, scene_name, logger)
        timings['steps2_5_sugar'] = time.time() - t0

        # Collect outputs
        t0 = time.time()
        collect_outputs(scene_name, gs_model_dir, logger)
        timings['collect'] = time.time() - t0

        # Cleanup intermediates to save space
        cleanup_sugar_output(scene_name, logger)

        elapsed_total = time.time() - t_total
        timings['total'] = elapsed_total

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"场景 {scene_name} 耗时统计:")
        logger.info(f"  Step 1   Vanilla 3DGS:  {timings['step1_vanilla_gs']/60:7.1f} min")
        logger.info(f"  Step 1.5 清废光:        {timings['step1_5_clean']/60:7.1f} min")
        logger.info(f"  Steps 2-5 SuGaR:       {timings['steps2_5_sugar']/60:7.1f} min")
        logger.info(f"  Collect outputs:        {timings['collect']/60:7.1f} min")
        logger.info(f"  ───────────────────────────────")
        logger.info(f"  总计:                   {elapsed_total/60:7.1f} min ({elapsed_total/3600:.2f} h)")
        logger.info(f"{'='*60}")

        return True, timings

    except Exception as e:
        elapsed_total = time.time() - t_total
        timings['total'] = elapsed_total
        logger.error(f"Scene {scene_name} FAILED after {elapsed_total:.0f}s: {e}")
        logger.error(traceback.format_exc())
        return False, timings


LOCK_FILE = os.path.join(GS_OUT_ROOT, ".batch_lock")

def acquire_lock():
    """防止批处理脚本多次运行冲突"""
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            old_pid = f.read().strip()
        # 检查该进程是否仍在运行
        try:
            import signal
            os.kill(int(old_pid), 0)
            print(f"错误：另一个批处理进程已在运行 (PID {old_pid})，退出。")
            sys.exit(1)
        except (OSError, ValueError):
            print(f"清理旧锁文件 (PID {old_pid} 已不存在)")
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))
    atexit.register(release_lock)

def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


def main():
    os.makedirs(GS_OUT_ROOT, exist_ok=True)
    os.makedirs(MESH_OUT_ROOT, exist_ok=True)
    acquire_lock()

    scenes = get_scene_list()
    print(f"找到 {len(scenes)} 个可用场景: {scenes}")

    results = {}
    all_timings = {}
    scene_durations = []  # 用于计算平均耗时
    t_start = time.time()

    # 计算需要处理的场景数
    to_process = []
    for s in scenes:
        mesh_out_dir = os.path.join(MESH_OUT_ROOT, s)
        if os.path.isdir(mesh_out_dir) and any(f.endswith('.obj') for f in os.listdir(mesh_out_dir)):
            continue
        to_process.append(s)
    print(f"需要处理 {len(to_process)}/{len(scenes)} 个场景")
    print(f"已跳过 (有.obj): {[s for s in scenes if s not in to_process]}")

    for idx, scene_name in enumerate(scenes, 1):
        # Check if already completed (has mesh output)
        mesh_out = os.path.join(MESH_OUT_ROOT, scene_name)
        if os.path.isdir(mesh_out) and any(f.endswith('.obj') for f in os.listdir(mesh_out)):
            print(f"[{idx}/{len(scenes)}] {scene_name} - SKIPPED (already has .obj)")
            results[scene_name] = "skipped"
            continue

        # 计算 ETA
        processed_idx = len(scene_durations)
        remaining = len(to_process) - processed_idx
        if scene_durations:
            avg_min = sum(scene_durations) / len(scene_durations) / 60
            eta_min = avg_min * remaining
            eta_str = f"  | 平均 {avg_min:.1f}min/场景, 预计剩余 {eta_min:.0f}min ({eta_min/60:.1f}h)"
        else:
            eta_str = ""

        batch_elapsed = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(scenes)}] Processing: {scene_name}  ({processed_idx+1}/{len(to_process)})")
        print(f"  批处理已运行 {batch_elapsed/60:.1f}min ({batch_elapsed/3600:.2f}h){eta_str}")
        print(f"{'='*70}")

        ok, timings = process_scene(scene_name)
        results[scene_name] = "OK" if ok else "FAILED"
        all_timings[scene_name] = timings
        if 'total' in timings:
            scene_durations.append(timings['total'])

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"批处理全部完成  总耗时: {elapsed/60:.1f}min ({elapsed/3600:.2f}h)")
    print(f"{'='*70}")
    print(f"\n场景耗时明细:")
    print(f"{'场景':<20s} {'状态':<8s} {'3DGS':>8s} {'清废光':>8s} {'SuGaR':>8s} {'总计':>8s}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, status in results.items():
        marker = "✓" if status == "OK" else ("⊘" if status == "skipped" else "✗")
        if name in all_timings:
            t = all_timings[name]
            s1 = f"{t.get('step1_vanilla_gs',0)/60:.1f}"
            s15 = f"{t.get('step1_5_clean',0)/60:.1f}"
            s25 = f"{t.get('steps2_5_sugar',0)/60:.1f}"
            tot = f"{t.get('total',0)/60:.1f}"
            print(f"  {marker} {name:<18s} {status:<8s} {s1:>7s}m {s15:>7s}m {s25:>7s}m {tot:>7s}m")
        else:
            print(f"  {marker} {name:<18s} {status:<8s}")

    if scene_durations:
        avg = sum(scene_durations) / len(scene_durations)
        print(f"\n平均每场景: {avg/60:.1f}min ({avg/3600:.2f}h)")
        print(f"处理场景数: {len(scene_durations)} / 总计: {len(scenes)}")

    failed = [n for n, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\nFailed scenes ({len(failed)}): {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
