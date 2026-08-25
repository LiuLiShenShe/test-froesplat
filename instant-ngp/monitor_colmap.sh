#!/bin/bash
# 监控批处理COLMAP进度

OUTPUT_DIR="/data/fj/10-Instant-NGP"
LOG_FILE="/tmp/batch_colmap.log"

echo "========================================"
echo "COLMAP 批处理进度监控"
echo "========================================"
echo "开始时间: $(date)"
echo ""

while true; do
    # 检查进程是否还在运行
    if ! pgrep -f "batch_run_colmap.py" > /dev/null; then
        echo ""
        echo "========================================"
        echo "批处理已完成"
        echo "完成时间: $(date)"
        echo "========================================"

        # 统计结果
        completed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)
        echo ""
        echo "✓ 成功生成 transforms.json: $completed / 19"

        if [ $completed -gt 0 ]; then
            echo ""
            echo "已完成的文件夹:"
            find $OUTPUT_DIR -name "transforms.json" | while read f; do
                folder=$(basename $(dirname $(dirname $f)))
                echo "  - $folder"
            done
        fi

        echo ""
        echo "日志最后30行:"
        tail -30 $LOG_FILE
        break
    fi

    # 统计进度
    completed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)
    total=19

    # 获取当前处理的文件夹
    current=$(grep -oP "处理: \K[^ ]+" $LOG_FILE | tail -1)

    clear
    echo "========================================"
    echo "COLMAP 批处理进度监控"
    echo "========================================"
    echo "当前时间: $(date '+%H:%M:%S')"
    echo ""
    echo "进度: $completed / $total"
    echo "当前处理: $current"
    echo ""
    echo "按 Ctrl+C 退出监控（不会中断处理）"
    echo "========================================"

    # 等待2分钟
    sleep 120
done
