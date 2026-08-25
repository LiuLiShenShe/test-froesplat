#!/bin/bash
# 批量运行 COLMAP - 纯CPU模式（无OpenGL依赖）
# 适用于无显示器的远程服务器

OUTPUT_DIR="/data/fj/10-Instant-NGP"
INSTANT_NGP_DIR="/data/fj/instant-ngp"
AABB_SCALE=32

# 环境变量：强制CPU模式
export QT_QPA_PLATFORM=offscreen
export COLMAP_CPU_THREADS=8  # 根据CPU核心数调整

echo "============================================================"
echo "批量运行 COLMAP（纯CPU模式）"
echo "============================================================"
echo "输出目录: $OUTPUT_DIR"
echo "AABB Scale: $AABB_SCALE"
echo "CPU线程数: $COLMAP_CPU_THREADS"
echo ""

# 激活Python环境
source "$INSTANT_NGP_DIR/venv/bin/activate"

# 获取所有文件夹
cd $OUTPUT_DIR
folders=($(ls -d */ | sed 's|/||'))

echo "找到 ${#folders[@]} 个文件夹"
echo ""

success_count=0
failed_folders=()

for folder in "${folders[@]}"; do
    echo "============================================================"
    echo "[$(($success_count+1))/${#folders[@]}] 处理: $folder"
    echo "============================================================"

    folder_path="$OUTPUT_DIR/$folder"
    cd "$folder_path"

    # 检查images文件夹
    if [ ! -d "images" ]; then
        echo "⚠️ images 文件夹不存在，跳过"
        failed_folders+=("$folder")
        continue
    fi

    # 检查是否已有transforms.json
    if [ -f "transforms.json" ]; then
        echo "✓ transforms.json 已存在，跳过"
        ((success_count++))
        continue
    fi

    # 统计图像数量
    png_count=$(ls images/*.png 2>/dev/null | wc -l)
    echo "找到 $png_count 张 PNG 图像"

    # 步骤1: 特征提取（CPU模式）
    echo ""
    echo "步骤1: 特征提取..."
    colmap feature_extractor \
        --database_path colmap.db \
        --image_path images \
        --ImageReader.camera_model OPENCV \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 0 \
        --SiftExtraction.num_threads $COLMAP_CPU_THREADS

    if [ $? -ne 0 ]; then
        echo "❌ 特征提取失败"
        failed_folders+=("$folder")
        continue
    fi

    # 步骤2: 特征匹配（CPU模式）
    echo ""
    echo "步骤2: 特征匹配..."
    colmap sequential_matcher \
        --database_path colmap.db \
        --SiftMatching.use_gpu 0 \
        --SiftMatching.num_threads $COLMAP_CPU_THREADS

    if [ $? -ne 0 ]; then
        echo "❌ 特征匹配失败"
        failed_folders+=("$folder")
        continue
    fi

    # 步骤3: 稀疏重建
    echo ""
    echo "步骤3: 稀疏重建..."
    mkdir -p colmap_sparse
    colmap mapper \
        --database_path colmap.db \
        --image_path images \
        --output_path colmap_sparse \
        --Mapper.ba_refine_principal_point 1

    if [ $? -ne 0 ]; then
        echo "⚠️ 稀疏重建失败，但继续尝试生成transforms.json"
    fi

    # 步骤4: Bundle Adjustment
    if [ -d "colmap_sparse/0" ]; then
        echo ""
        echo "步骤4: Bundle Adjustment..."
        colmap bundle_adjuster \
            --input_path colmap_sparse/0 \
            --output_path colmap_sparse/0 \
            --BundleAdjustment.refine_principal_point 1
    fi

    # 步骤5: 转换为文本格式
    echo ""
    echo "步骤5: 转换为文本格式..."
    mkdir -p colmap_text
    colmap model_converter \
        --input_path colmap_sparse/0 \
        --output_path colmap_text \
        --output_type TXT

    # 步骤6: 生成transforms.json
    echo ""
    echo "步骤6: 生成 transforms.json..."
    python "$INSTANT_NGP_DIR/scripts/colmap2nerf.py" \
        --colmap_db colmap.db \
        --images images \
        --text colmap_text \
        --aabb_scale $AABB_SCALE \
        --out transforms.json

    # 检查结果
    if [ -f "transforms.json" ]; then
        echo ""
        echo "✓ 完成: $folder"
        ((success_count++))
    else
        echo ""
        echo "⚠️ transforms.json 未生成: $folder"
        failed_folders+=("$folder")
    fi

    echo ""
done

# 总结
cd "$OUTPUT_DIR"
echo "============================================================"
echo "处理完成"
echo "============================================================"
echo "✓ 成功: $success_count/${#folders[@]}"

if [ ${#failed_folders[@]} -gt 0 ]; then
    echo "✗ 失败: ${#failed_folders[@]}"
    for folder in "${failed_folders[@]}"; do
        echo "  - $folder"
    done
fi

completed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)
echo ""
echo "生成的 transforms.json 文件数: $completed"

echo ""
echo "输出位置: $OUTPUT_DIR"
echo ""
echo "查看结果:"
echo "  ls $OUTPUT_DIR/*/transforms.json"
echo ""
echo "训练 NeRF:"
echo "  ./instant-ngp $OUTPUT_DIR/CaoMei1"
