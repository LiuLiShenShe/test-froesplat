<# 
  SuGaR 批训练监控脚本
  用法: powershell -File D:\CAAS\SuGaR-main\monitor.ps1
  或在终端中: .\monitor.ps1
#>

Write-Host "========== SuGaR 批训练监控 ==========" -ForegroundColor Cyan
Write-Host "时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# 1. 检查 Python 进程
Write-Host "--- Python 进程 ---" -ForegroundColor Yellow
$pyProcs = Get-Process python* -ea 0 | Where-Object {$_.WS -gt 50MB}
if ($pyProcs) {
    $pyProcs | Select Id,
        @{N='内存(MB)';E={[math]::Round($_.WS/1MB,1)}},
        @{N='CPU(s)';E={[math]::Round($_.CPU,0)}} | Format-Table -Auto
} else {
    Write-Host "  无活跃 Python 进程 (训练可能已全部完成或未启动)" -ForegroundColor Red
}

# 2. GPU 状态
Write-Host "--- GPU 状态 ---" -ForegroundColor Yellow
$gpuInfo = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null
if ($gpuInfo) {
    Write-Host "  $gpuInfo"
} else {
    Write-Host "  nvidia-smi 不可用"
}

# 3. 检查 SuGaR output 目录
Write-Host ""
Write-Host "--- SuGaR 中间输出 (SuGaR-main/output/) ---" -ForegroundColor Yellow
foreach ($sub in @("coarse","coarse_mesh","refined","refined_mesh")) {
    $dir = "D:\CAAS\SuGaR-main\output\$sub"
    $files = Get-ChildItem $dir -Recurse -File -ea 0
    $count = if($files) { $files.Count } else { 0 }
    $size = if($files) { [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 1) } else { 0 }
    $scenes = Get-ChildItem $dir -Directory -ea 0 | Select -ExpandProperty Name
    $sceneList = if($scenes) { $scenes -join ", " } else { "-" }
    Write-Host "  $sub : $count 文件 ($size MB) | 场景: $sceneList"
}

# 4. 检查最终输出
Write-Host ""
Write-Host "--- 最终输出 ---" -ForegroundColor Yellow
Write-Host "  07-SuGaR-GS:" -ForegroundColor Green
$gsScenes = Get-ChildItem "D:\CAAS\07-SuGaR-GS" -Directory -ea 0 | Where-Object { $_.Name -ne ".batch_lock" }
foreach ($s in $gsScenes) {
    $fileCount = (Get-ChildItem $s.FullName -Recurse -File -ea 0).Count
    $hasLog = Test-Path (Join-Path $s.FullName "log.txt")
    $logStatus = if($hasLog) { "有log" } else { "无log" }
    Write-Host "    $($s.Name): $fileCount 文件 ($logStatus)"
}

Write-Host "  07-SuGaR-Mesh:" -ForegroundColor Green
$meshScenes = Get-ChildItem "D:\CAAS\07-SuGaR-Mesh" -Directory -ea 0
foreach ($s in $meshScenes) {
    $objFiles = Get-ChildItem $s.FullName -Filter "*.obj" -ea 0
    $objStatus = if($objFiles) { "有 .obj" } else { "无 .obj" }
    $fileCount = (Get-ChildItem $s.FullName -File -ea 0).Count
    Write-Host "    $($s.Name): $fileCount 文件 ($objStatus)"
}
if (-not $meshScenes) {
    Write-Host "    (空)" -ForegroundColor DarkGray
}

# 5. 最新日志
Write-Host ""
Write-Host "--- batch_train.log 最后 20 行 ---" -ForegroundColor Yellow
$logPath = "D:\CAAS\07-SuGaR-GS\batch_train.log"
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 20
} else {
    Write-Host "  日志文件不存在"
}

# 6. 当前场景日志
Write-Host ""
Write-Host "--- 各场景 log.txt 最后更新 ---" -ForegroundColor Yellow
$sceneLogs = Get-ChildItem "D:\CAAS\07-SuGaR-GS\*\log.txt" -ea 0
foreach ($log in $sceneLogs) {
    $lastLine = Get-Content $log.FullName -Tail 1 -ea 0
    $lastWrite = $log.LastWriteTime.ToString("HH:mm:ss")
    $sceneName = $log.Directory.Name
    Write-Host "  $sceneName ($lastWrite): $lastLine"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
