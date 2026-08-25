# 监控 FFmepg 文件夹下各子文件夹的图片生成进度
# 每 10 秒刷新一次，写入每个文件夹的 progress.log

$outputDir = "D:\CAAS\FFmepg"

while ($true) {
    $dirs = Get-ChildItem $outputDir -Directory | Where-Object { $_.Name -ne "ffmpeg_bin" }
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    foreach ($dir in $dirs) {
        $count = (Get-ChildItem $dir.FullName -File -Filter "*.png" -ErrorAction SilentlyContinue).Count
        $logFile = Join-Path $dir.FullName "progress.log"
        $line = "$timestamp | $($dir.Name) | $count images"
        Add-Content -Path $logFile -Value $line -Encoding UTF8
    }
    
    # 同时打印到控制台
    Write-Host "--- $timestamp ---"
    foreach ($dir in $dirs) {
        $count = (Get-ChildItem $dir.FullName -File -Filter "*.png" -ErrorAction SilentlyContinue).Count
        Write-Host "  $($dir.Name): $count images"
    }
    
    # 检查 ffmpeg 是否还在运行
    $ffmpegProcs = Get-Process -Name ffmpeg -ErrorAction SilentlyContinue
    if (-not $ffmpegProcs) {
        Write-Host "`nAll ffmpeg processes finished."
        # 最终统计
        foreach ($dir in $dirs) {
            $count = (Get-ChildItem $dir.FullName -File -Filter "*.png" -ErrorAction SilentlyContinue).Count
            $logFile = Join-Path $dir.FullName "progress.log"
            Add-Content -Path $logFile -Value "=== DONE | $count total images ===" -Encoding UTF8
        }
        break
    }
    
    Start-Sleep -Seconds 10
}
