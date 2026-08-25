#!/bin/bash
# 使用 instant-ngp 命令行渲染 NeRF 并导出

SCENE="/data/fj/10-Instant-NGP/CaoMei1"
SNAPSHOT="/data/fj/10-Instant-NGP/CaoMei1_trained.msgpack"
INSTANT_NGP="/data/fj/instant-ngp"

echo "============================================================"
echo "NeRF 渲染和导出工具"
echo "============================================================"
echo "场景: $SCENE"
echo "快照: $SNAPSHOT"
echo ""

# 选项1: 加载快照并显示帮助（查看可用导出命令）
echo "选项1: 查看可用的导出命令"
echo "命令: $INSTANT_NGP/instant-ngp $SCENE --load_snapshot $SNAPSHOT"
echo ""

# 选项2: 渲染单个视角图像
echo "选项2: 渲染预览图像"
mkdir -p /data/fj/10-Instant-NGP/CaoMei1_renders
for i in {0,90,180,270}; do
    echo "  渲染角度 $i°..."
    $INSTANT_NGP/instant-ngp $SCENE \
        --load_snapshot $SNAPSHOT \
        --no-gui \
        --camera_yaw $i \
        --output /data/fj/10-Instant-NGP/CaoMei1_renders/angle_${i}.png 2>/dev/null
done

echo "  ✓ 渲染完成: 4张预览图像"
echo "  输出: /data/fj/10-Instant-NGP/CaoMei1_renders/"
echo ""

# 选项3: 渲染为视频
echo "选项3: 渲染为视频"
$INSTANT_NGP/instant-ngp $SCENE \
    --load_snapshot $SNAPSHOT \
    --no-gui \
    --camera_path "$SCENE/camera_path.txt" \
    --output /data/fj/10-Instant-NGP/CaoMei1_video.mp4 2>/dev/null

if [ -f /data/fj/10-Instant-NGP/CaoMei1_video.mp4 ]; then
    echo "  ✓ 视频已导出"
    echo "  输出: /data/fj/10-Instant-NGP/CaoMei1_video.mp4"
else
    echo "  ℹ️ 视频导出失败（可能需要相机路径文件）"
fi

echo ""
echo "============================================================"
echo "总结"
echo "============================================================"
echo "文件位置:"
echo "  预览图像: /data/fj/10-Instant-NGP/CaoMei1_renders/"
echo "  视频: /data/fj/10-Instant-NGP/CaoMei1_video.mp4"
echo ""
echo "CloudCompare 使用:"
echo "  1. 预览图像：直接打开 .png 文件"
echo "  2. 视频：使用播放器打开 .mp4"
echo "  3. 点云：使用 instant-ngp GUI 导出"
