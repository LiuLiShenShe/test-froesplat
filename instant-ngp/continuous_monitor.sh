#!/bin/bash
# 持续监控批处理进度，每5分钟检查一次

LOG_FILE="/tmp/batch_process_v2.log"
OUTPUT_DIR="/data/fj/10-Instant-NGP"
MONITOR_LOG="/tmp/nerf_batch_monitor.log"

echo "开始监控批处理进度..." | tee -a $MONITOR_LOG
echo "日志文件: $LOG_FILE" | tee -a $MONITOR_LOG
echo "监控开始时间: $(date)" | tee -a $MONITOR_LOG
echo "----------------------------------------" | tee -a $MONITOR_LOG

while true; do
    # 检查进程是否还在运行
    if ! pgrep -f "batch_process_sam_to_nerf.py" > /dev/null; then
        echo "" | tee -a $MONITOR_LOG
        echo "==========================================" | tee -a $MONITOR_LOG
        echo "批处理进程已结束" | tee -a $MONITOR_LOG
        echo "结束时间: $(date)" | tee -a $MONITOR_LOG
        echo "==========================================" | tee -a $MONITOR_LOG

        # 显示最终统计
        completed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)
        echo "完成文件夹数: $completed / 19" | tee -a $MONITOR_LOG

        # 显示日志最后50行
        echo "" | tee -a $MONITOR_LOG
        echo "日志最后50行:" | tee -a $MONITOR_LOG
        tail -50 $LOG_FILE | tee -a $MONITOR_LOG

        break
    fi

    # 统计进度
    completed=$(find $OUTPUT_DIR -name "transforms.json" 2>/dev/null | wc -l)
    processing=$(find $OUTPUT_DIR -type d -name "images" 2>/dev/null | wc -l)

    echo "" | tee -a $MONITOR_LOG
    echo "[$(date '+%H:%M:%S')] 进度更新:" | tee -a $MONITOR_LOG
    echo "  完成: $completed / 19" | tee -a $MONITOR_LOG
    echo "  处理中: $processing" | tee -a $MONITOR_LOG

    # 检查是否有错误
    errors=$(grep -c "❌ 处理失败" $LOG_FILE 2>/dev/null || echo "0")
    if [ "$errors" -gt 0 ]; then
        echo "  ⚠️ 失败: $errors 个文件夹" | tee -a $MONITOR_LOG
    fi

    # 等待5分钟
    sleep 300
done

echo "" | tee -a $MONITOR_LOG
echo "监控结束" | tee -a $MONITOR_LOG
