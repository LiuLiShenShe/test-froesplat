<#
  Retry 4 failed scenes: DouBanLv1, DouBanLv3, HongZhang, WanNianQing1
  Fixes applied:
    1. dataset_readers.py patched: auto-resolve .jpg/.png extension mismatch
    2. DouBanLv3 points3D.ply deleted (missing normals), will regenerate from .bin
#>

$trainOutBase  = "D:\CAAS\05-2DGS-new"
$meshOutBase   = "D:\CAAS\06-MESH-new"
$repoDir       = "D:\CAAS\2d-gaussian-splatting-main"
$colmapBase    = "D:\CAAS\04-COLMAP"
$venvActivate  = "D:\CAAS\2d-gaussian-splatting-great-again-dev\.venv_uv\Scripts\Activate.ps1"
$cleanScript   = "$trainOutBase\clean_ply_v3.py"
$masterLog     = "$trainOutBase\master_log_retry.txt"
$iteration     = 30000
$basePort      = 6090

$scenes = @("DouBanLv1", "DouBanLv3", "HongZhang", "WanNianQing1")

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
& $venvActivate

$totalScenes = $scenes.Count
$globalStart = Get-Date

function Log {
    param([string]$msg)
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $masterLog -Value $line
}

function FormatElapsed {
    param($elapsed)
    $h = [math]::Floor($elapsed.TotalHours)
    $m = $elapsed.Minutes
    $s = $elapsed.Seconds
    if ($h -gt 0) { return "${h}h${m}m${s}s" }
    else          { return "${m}m${s}s" }
}

Log "========================================================================"
Log "  RETRY: 4 Failed Scenes"
Log "========================================================================"

# ---- PHASE 1: TRAINING ----
Log ""
Log "  PHASE 1: TRAINING"
$trainResults = @{}
$phase1Start = Get-Date

Push-Location $repoDir

for ($idx = 0; $idx -lt $totalScenes; $idx++) {
    $scene = $scenes[$idx]
    $num = $idx + 1
    $srcDir  = "$colmapBase\$scene"
    $outDir  = "$trainOutBase\$scene"
    $logFile = "$trainOutBase\${scene}_train.log"
    $port    = $basePort + $num
    $plyPath = "$outDir\point_cloud\iteration_$iteration\point_cloud.ply"

    Log ""
    Log ("---- [" + $num + "/" + $totalScenes + "] TRAIN: " + $scene + " ----")

    if (Test-Path $plyPath) {
        $szMB = "{0:N1}" -f ((Get-Item $plyPath).Length / 1MB)
        Log ("  SKIP: Already completed (PLY=" + $szMB + " MB)")
        $trainResults[$scene] = "EXIST"
        continue
    }

    if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }

    $sceneStart = Get-Date
    Log ("  Started at " + (Get-Date -Format 'HH:mm:ss'))

    cmd /c "python train.py -s `"$srcDir`" -m `"$outDir`" --images images_rgba -r 2 --depth_ratio 1 --save_iterations 7000 $iteration --port $port > `"$logFile`" 2>&1"

    $sceneElapsed = (Get-Date) - $sceneStart
    $elStr = FormatElapsed $sceneElapsed

    if (Test-Path $plyPath) {
        $szMB = "{0:N1}" -f ((Get-Item $plyPath).Length / 1MB)
        Log ("  SUCCESS: " + $scene + " in " + $elStr + " (PLY=" + $szMB + " MB)")
        $trainResults[$scene] = "OK | " + $elStr + " | " + $szMB + " MB"
    } else {
        Log ("  FAILED: " + $scene + " after " + $elStr + " -- check " + $logFile)
        $trainResults[$scene] = "FAIL | " + $elStr
    }
}

Pop-Location

$phase1Elapsed = (Get-Date) - $phase1Start
Log ""
Log ("---- PHASE 1 SUMMARY (" + (FormatElapsed $phase1Elapsed) + ") ----")
foreach ($scene in $scenes) { Log ("  " + $scene + " : " + $trainResults[$scene]) }

# ---- PHASE 2: V3 CLEANING ----
Log ""
Log "  PHASE 2: V3 CLEANING"
$phase2Start = Get-Date
$sceneList = ($scenes -join " ")
$cleanLog = "$trainOutBase\clean_v3_retry.log"

cmd /c "python `"$cleanScript`" --base_dir `"$trainOutBase`" --scenes $sceneList --iterations $iteration > `"$cleanLog`" 2>&1"

$phase2Elapsed = (Get-Date) - $phase2Start
Log ("  V3 Cleaning done in " + (FormatElapsed $phase2Elapsed))
Get-Content $cleanLog -Tail 10 | ForEach-Object { Log ("    " + $_) }

# ---- PHASE 3: MESH EXTRACTION ----
Log ""
Log "  PHASE 3: MESH EXTRACTION"
$phase3Start = Get-Date
$meshResults = @{}

Push-Location $repoDir

for ($idx = 0; $idx -lt $totalScenes; $idx++) {
    $scene = $scenes[$idx]
    $num = $idx + 1
    $pcDir     = "$trainOutBase\$scene\point_cloud\iteration_$iteration"
    $origPly   = "$pcDir\point_cloud.ply"
    $v3Ply     = "$pcDir\point_cloud_clean_v3.ply"
    $backupPly = "$pcDir\point_cloud_orig.ply"
    $trainDir  = "$trainOutBase\$scene\train\ours_$iteration"

    Log ""
    Log ("---- [" + $num + "/" + $totalScenes + "] MESH: " + $scene + " ----")

    if (-not (Test-Path $v3Ply)) {
        Log "  SKIP: V3 PLY not found"
        $meshResults[$scene] = "SKIP"
        continue
    }

    $meshStart = Get-Date

    # Swap v3 PLY
    if (-not (Test-Path $backupPly)) { Copy-Item $origPly $backupPly -Force }
    Copy-Item $v3Ply $origPly -Force
    Log "  Swapped V3 PLY"

    if (Test-Path $trainDir) { Remove-Item "$trainDir\fuse*.ply" -Force -ErrorAction SilentlyContinue }

    $meshLog = "$meshOutBase\${scene}_mesh.log"
    cmd /c "python render.py -m `"$trainOutBase\$scene`" -s `"$colmapBase\$scene`" --depth_ratio 1 --num_cluster 50 --mesh_res 1024 --iteration $iteration --skip_test --skip_train > `"$meshLog`" 2>&1"

    # Copy results
    $outMeshDir = "$meshOutBase\$scene"
    New-Item -ItemType Directory -Path $outMeshDir -Force | Out-Null
    $fusePly  = "$trainDir\fuse.ply"
    $fusePost = "$trainDir\fuse_post.ply"
    $ok = $true
    if (Test-Path $fusePly)  { Copy-Item $fusePly  $outMeshDir -Force; $fuseSzMB = "{0:N1}" -f ((Get-Item $fusePly).Length / 1MB) } else { $fuseSzMB = "MISSING"; $ok = $false }
    if (Test-Path $fusePost) { Copy-Item $fusePost $outMeshDir -Force; $postSzMB = "{0:N1}" -f ((Get-Item $fusePost).Length / 1MB) } else { $postSzMB = "MISSING"; $ok = $false }

    # Restore original
    if (Test-Path $backupPly) { Copy-Item $backupPly $origPly -Force }

    $meshElapsed = (Get-Date) - $meshStart
    $meshElStr = FormatElapsed $meshElapsed

    if ($ok) {
        Log ("  SUCCESS: " + $scene + " in " + $meshElStr + " (fuse=" + $fuseSzMB + " MB, post=" + $postSzMB + " MB)")
        $meshResults[$scene] = "OK | " + $meshElStr + " | post=" + $postSzMB + " MB"
    } else {
        Log ("  FAILED: " + $scene + " in " + $meshElStr + " -- check " + $meshLog)
        $meshResults[$scene] = "FAIL | " + $meshElStr
    }
}

Pop-Location

$phase3Elapsed = (Get-Date) - $phase3Start
Log ""
Log ("---- PHASE 3 SUMMARY (" + (FormatElapsed $phase3Elapsed) + ") ----")
foreach ($scene in $scenes) { Log ("  " + $scene + " : " + $meshResults[$scene]) }

# ---- FINAL ----
$globalElapsed = (Get-Date) - $globalStart
Log ""
Log "========================================================================"
Log ("  RETRY COMPLETE - Total: " + (FormatElapsed $globalElapsed))
Log "========================================================================"
foreach ($scene in $scenes) { Log ("  " + $scene + " : Train=" + $trainResults[$scene] + " | Mesh=" + $meshResults[$scene]) }
Log "Done."
