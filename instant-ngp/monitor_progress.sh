#!/bin/bash
# 监控批量处理进度

OUTPUT_DIR="/data/fj/10-Instant-NGP"
SOURCE_DIR="/data/fj/03-SAM"

echo "======================================"
echo "批量处理进度监控"
echo "======================================"
echo ""

# 统计已处理的文件夹
processed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)
total=$(ls -d $SOURCE_DIR/*/ 2>/dev/null | wc -l)

echo "总体进度: $processed / $total 个文件夹"
echo ""

# 显示每个文件夹的进度
echo "详细进度:"
echo "--------------------------------------"

for dir in $SOURCE_DIR/*/; do
    folder_name=$(basename "$dir")
    output_folder="$OUTPUT_DIR/$folder_name"

    if [ -f "$output_folder/transforms.json" ]; then
        # 已完成
        images=$(ls "$output_folder/images/"*.png 2>/dev/null | wc -l)
        echo "✅ $folder_name: 完成 ($images 张图像)"
    elif [ -d "$output_folder/images" ]; then
        # 进行中
        converted=$(ls "$output_folder/images/"*.png 2>/dev/null | wc -l)
        total_images=$(ls "$dir"*.png 2>/dev/null | wc -l)
        percent=$((converted * 100 / total_images))
        echo "🔄 $folder_name: 转换中 ($converted / $total_images) [$percent%]"
    else
        # 未开始
        echo "⏳ $folder_name: 等待中"
    fi
done

echo ""
echo "======================================"
echo "按 Ctrl+C 退出监控"
echo "======================================"
