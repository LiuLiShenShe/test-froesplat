#!/bin/bash
# 实时监控COLMAP批处理进度

LOG_FILE="/tmp/batch_colmap_cpu.log"
OUTPUT_DIR="/data/fj/10-Instant-NGP"

echo "============================================================"
echo "COLMAP 批处理进度监控（CPU模式）"
echo "============================================================"
echo "日志文件: $LOG_FILE"
echo "开始监控时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "按 Ctrl+C 退出（不会中断处理）"
echo "============================================================"
echo ""

# 持续监控
while true; do
    # 检查进程
    if ! pgrep -f "batch_run_colmap_cpu_only.sh" > /dev/null; then
        echo ""
        echo "============================================================"
        echo "批处理已完成"
        echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"

        # 最终统计
        completed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)
        echo ""
        echo "✓ 成功生成: $completed / 19"

        if [ $completed -gt 0 ]; then
            echo ""
            echo "已完成的文件夹:"
            find $OUTPUT_DIR -name "transforms.json" | sort | while read f; do
                folder=$(basename $(dirname $(dirname $f)))
                echo "  ✓ $folder"
            done
        fi

        # 显示失败信息
        if grep -q "✗ 失败" $LOG_FILE; then
            echo ""
            echo "失败信息:"
            grep -A20 "✗ 失败" $LOG_FILE | tail -20
        fi

        echo ""
        echo "日志最后30行:"
        tail -30 $LOG_FILE
        break
    fi

    # 获取当前状态
    current_step=$(tail -100 $LOG_FILE | grep -E "步骤[0-9]:" | tail -1)
    current_folder=$(tail -100 $LOG_FILE | grep "处理:" | tail -1 | sed 's/.*处理: //')

    # 统计进度
    completed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)

    # 清屏并显示
    clear
    echo "============================================================"
    echo "COLMAP 批处理进度监控（CPU模式）"
    echo "============================================================"
    echo "当前时间: $(date '+%H:%M:%S')"
    echo ""
    echo "总进度: $completed / 19"
    echo "当前文件夹: $current_folder"
    echo "当前步骤: $current_step"
    echo ""
    echo "最近日志:"
    tail -10 $LOG_FILE
    echo ""
    echo "============================================================"
    echo "按 Ctrl+C 退出（不会中断处理）"
    echo "============================================================"

    # 等待2分钟
    sleep 120
done
