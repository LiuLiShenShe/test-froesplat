#!/bin/bash
# 批量为已转换的图像运行 COLMAP（使用CPU模式避免OpenGL依赖）
# 每个文件夹单独处理

OUTPUT_DIR="/data/fj/10-Instant-NGP"
INSTANT_NGP_DIR="/data/fj/instant-ngp"
AABB_SCALE=32

# 设置环境变量
export QT_QPA_PLATFORM=offscreen

echo "============================================================"
echo "批量运行 COLMAP（CPU模式）"
echo "============================================================"
echo "输出目录: $OUTPUT_DIR"
echo "AABB Scale: $AABB_SCALE"
echo ""

# 获取所有文件夹
cd $OUTPUT_DIR
folders=($(ls -d */ | sed 's|/||'))

echo "找到 ${#folders[@]} 个文件夹"
echo ""

success_count=0
failed_folders=()

for folder in "${folders[@]}"; do
    echo "============================================================"
    echo "[$((${success_count}+1))/${#folders[@]}] 处理: $folder"
    echo "============================================================"

    cd "$OUTPUT_DIR/$folder"

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

    # 运行COLMAP
    echo "运行 colmap2nerf.py..."
    python "$INSTANT_NGP_DIR/scripts/colmap2nerf.py" \
        --images images \
        --run_colmap \
        --colmap_matcher sequential \
        --aabb_scale $AABB_SCALE \
        --overwrite

    # 检查结果
    if [ -f "transforms.json" ]; then
        echo "✓ 完成: $folder"
        ((success_count++))
    else
        echo "⚠️ transforms.json 未生成: $folder"
        failed_folders+=("$folder")
    fi

    echo ""
done

# 总结
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

echo ""
echo "输出位置: $OUTPUT_DIR"
echo ""
echo "后续步骤:"
echo "1. 检查 transforms.json 文件:"
echo "   ls $OUTPUT_DIR/*/transforms.json"
echo "2. 训练 NeRF:"
echo "   ./instant-ngp $OUTPUT_DIR/CaoMei1"
