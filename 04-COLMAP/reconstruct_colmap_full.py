#!/usr/bin/env python3
"""
COLMAP 完整重建批处理
1. 使用原始图像（full frames）重新运行 COLMAP
2. 保持原始文件夹结构
3. 输出结果到 11-COLMAP-fuse
"""
import os
import shutil
from pathlib import Path

# 配置
SOURCE_DIR = "/data/fj/04-COLMAP"
OUTPUT_DIR = "/data/fj/11-COLMAP-fuse"

# 完整的重建流程
# 1. 特征提取 (feature extractor)
# 2. 特征匹配 (sequential/exhaustive matcher)
# 3. 稀疏重建 (mapper)
# 4. 导出为文本格式
# 5. 可选：图像去模糊/下采样

# 列表中的所有植物
PLANTS = [
    "BaiZhang", "CaoMei1", "ChangShouHua1", "ChangShouHua2", "ChangShouHua3",
    "DouBanLv1", "DouBanLv2", "DouBanLv3",
    "HongZhang", "KongQueZhuYu",
    "WangWenCao1", "WangWenCao2",
    "WanNianQing1", "WanNianQing2",
    "XiangPiShu1", "XiangPiShu2", "XiangKeLai1", "XianKeLai2",
    "XianKeLai3"
]

# 稠密重建参数
MATCHER = "sequential"  # 连续视频使用 sequential
MARCHING_CUBES_RES = 512  # Marching cubes 分辨率
DENSITY_THRESH = 2.5
MAX_RETRIES = 3

print("=" * 80)
print("COLMAP 完整重建批处理工具")
print("=" * 80)
print(f"源目录: {SOURCE_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"植物数量: {len(PLANTS)}")
print(f"匹配器: {MATCHER}")
print(f"重建分辨率: {MARCHING_CUBES_RES}³")
print(f"密度阈值: {DENSITY_THRESH}")
print("=" * 80)

# 创建输出主目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志文件
log_file = open(f"{OUTPUT_DIR}/reconstruction.log", "a", encoding="utf-8")

def log(message):
    log_file.write(f"[{message}]\n")

def run_colmap_command(command, cwd):
    """运行 COLMAP 命令"""
    log(f"执行: {command}")
    import subprocess
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    return result

def reconstruct_plant(plant_name):
    """重建单个植物"""
    plant_dir = Path(SOURCE_DIR) / plant_name
    images_dir = plant_dir / "images"  # images 文件夹（原始帧）
    database = plant_dir / "colmap.db"

    if not images_dir.exists():
        log(f"[{plant_name}] ⚠️ images 目录不存在，跳过")
        return False

    # 确保 COLMAP 数据库不存在（从头开始）
    if database.exists():
        log(f"[{plant_name}] 移除旧数据库...")
        os.remove(database)

    # 统计图像数量
    image_files = sorted(list(images_dir.glob("*.png")))
    num_images = len(image_files)
    log(f"[{plant_name}] 找到 {num_images} 张图像")

    if num_images == 0:
        log(f"[{plant_name}] ⚠️ 没有图像，跳过")
        return False

    # 步骤 1: 特征提取
    log(f"[{plant_name}] [1/5] 特征提取 (feature extractor)...")
    result = run_colmap_command(
        f"colmap feature_extractor "
        f"--database_path {database} "
        f"--image_path {images_dir} "
        f"--ImageReader.camera_model OPENCV "
        f"--ImageReader.single_camera 1 "
        f"--SiftExtraction.estimate_affine_shape true "
        f"--SiftExtraction.domain_size_pooling true "
        f"--SiftExtraction.use_gpu 0 "
        f"--SiftExtraction.num_threads 8",
        cwd=str(plant_dir)
    )
    if result.returncode != 0:
        log(f"[{plant_name}] 特征提取失败")
        return False

    # 步骤 2: 特征匹配
    log(f"[{plant_name}] [2/5] 特征匹配 ({MATCHER} matcher)...")
    result = run_colmap_command(
        f"colmap {MATCHER}_matcher "
        f"--database_path {database} "
        f"--SiftMatching.guided_matching true "
        f"--SiftMatching.use_gpu 0 "
        f"--SiftMatching.num_threads 8",
        cwd=str(plant_dir)
    )
    if result.returncode != 0:
        log(f"[{plant_name}] 特征匹配失败")
        return False

    # 步骤 3: 稀疏重建
    log(f"[{plant_name}] [3/5] 稀疏重建 (mapper)...")
    sparse_dir = plant_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    result = run_colmap_command(
        f"colmap mapper "
        f"--database_path {database} "
        f"--image_path {images_dir} "
        f"--output_path {sparse_dir} "
        f"--Mapper.ba_refine_principal_point 1 "
        f"--Mapper.ba_refine_focal_length 1 "
        f"--Mapper.num_threads 8",
        cwd=str(plant_dir)
    )
    if result.returncode != 0:
        log(f"[{plant_name}] 稀疏重建失败")
        return False

    # 步骤 4: Bundle Adjustment（可选）
    sparse_0 = sparse_dir / "0"
    result = run_colmap_command(
        f"colmap bundle_adjuster "
        f"--input_path {sparse_0} "
        f"--output_path {sparse_0} "
        f"--BundleAdjustment.refine_principal_point 1 "
        f"--BundleAdjuster.num_threads 8",
        cwd=str(plant_dir)
    )

    # 步骤 5: 导出文本格式
    text_dir = plant_dir / "sparse_txt"
    text_dir.mkdir(exist_ok=True)

    result = run_colmap_command(
        f"colmap model_converter "
        f"--input_path {sparse_0} "
        f"--output_path {text_dir} "
        f"--output_type TXT",
        cwd=str(plant_dir)
    )

    # 统计输出文件
    sparse_files = list(sparse_dir.rglob("*.*"))
    log(f"[{plant_name}] ✓ 稀疏重建完成，生成 {len(sparse_files)} 个文件")

    # 创建输出子文件夹结构
    output_plant_dir = OUTPUT_DIR / plant_name
    output_plant_dir.mkdir(exist_ok=True)

    # 验证输出目录
    sparse_backup = output_plant_dir / "sparse"
    if sparse_backup.exists():
        shutil.rmtree(sparse_backup)

    # 将 sparse 内容复制到输出（保持原始结构）
    shutil.copytree(sparse_dir, output_plant_dir, symlinks=True)

    sparse_size = sum(f.stat().st_size for f in sparse_dir.rglob("*.*") if f.is_file())
    log(f"[{plant_name}] ✓ 输出大小: {sparse_size / 1024 / 1024:.2f} KB")

    log(f"[{plant_name}] ✓ 输出到: {output_plant_dir}")

    return True

# 主处理循环
log("")
log("开始批量重建...")
log("")

success_count = 0
failed_plants = []

for plant in PLANTS:
    try:
        if reconstruct_plant(plant):
            success_count += 1
            log(f"[{plant}] ✓ 完成")
        else:
            failed_plants.append(plant)
    except Exception as e:
        log(f"[{plant}] ✗ 失败: {e}")

# 最终总结
log("")
log("=" * 80)
log("批量重建完成")
log("=" * 80)
log(f"成功: {success_count}/{len(PLANTS)}")
if failed_plants:
    log("失败:")
    for plant in failed_plants:
        log(f"  - {plant}")
log("")
log("=" * 80)
log(f"输出目录: {OUTPUT_DIR}")
log("")
log("每个植物的文件夹结构:")
log(f"  <output>")
log(f"      ├── images/     # 原始图像")
log(f"      ├── sparse/       # 稀疏重建结果")
log(f"      └── transforms.json (如果有，需要从 04-COLMAP 复制)")
log("")
log("稠密重建说明:")
log("当前执行的是稀疏重建（稀疏点云）")
log("如需稠密重建（泊松表面或网格），可使用:")
log("  1. COLMAP stereo_fusion - 立体匹配和稠密化")
log("   2. 使用 MeshLab - 导入稀疏点云，应用稠密算法")
log("  3. Python PCL 库 - 实现去噪、稠密重建算法")
log("")
log("下一步：")
log("1. 使用 sparse/ 文件夹的文本格式点云在 CloudCompare 中查看")
log("2. 或使用 Python PCL 库进行进一步处理")
log("3. 或将稀疏点云导入 MeshLab 进行表面重建")

log_file.close()

print("")
print("COLMAP 完整重建批处理已启动！")
print("日志文件:", f"{OUTPUT_DIR}/reconstruction.log")
print("")
print("⏱ 预计时间：")
print("  - 单个植物：约 5-15 分钟（取决于图像数量）")
print(f"  - 总计：约 {len(PLANTS)*10} 分钟到 {len(PLANTS)*30} 分钟")
print("")
print("执行位置:", f"{SOURCE_DIR}/reconstruct_colmap_full.py")
print("")
print("📊 查看实时日志:")
print("  tail -100", f"{OUTPUT_DIR}/reconstruction.log")
