<#
  Standard 2DGS: Training -> V3 Cleaning -> Mesh Extraction
  
  Using the OFFICIAL 2DGS repo (D:\CAAS\2d-gaussian-splatting-main)
  NOT the "great-again" fork.
  
  Training params (all defaults, plus depth_ratio=1 for close-up plants):
    --iterations 30000         (default)
    --densify_until_iter 15000 (default)
    --densify_grad_threshold 0.0002 (default)
    --lambda_normal 0.05       (default)
    --lambda_dist 0            (default, distortion reg)
    --depth_ratio 1            (median depth, better for bounded scenes)
    -r 2                       (half resolution)
    --images images_rgba       (use SAM-masked RGBA images)
  
  Data source: D:\CAAS\04-COLMAP\{scene} (reusing existing COLMAP poses)
  Training output: D:\CAAS\05-2DGS-new\{scene}
  Mesh output:     D:\CAAS\06-MESH-new\{scene}
#>

# ---- Config ----
$trainOutBase  = "D:\CAAS\05-2DGS-new"
$meshOutBase   = "D:\CAAS\06-MESH-new"
$repoDir       = "D:\CAAS\2d-gaussian-splatting-main"
$colmapBase    = "D:\CAAS\04-COLMAP"
$venvActivate  = "D:\CAAS\2d-gaussian-splatting-great-again-dev\.venv_uv\Scripts\Activate.ps1"
$cleanScript   = "$trainOutBase\clean_ply_v3.py"
$masterLog     = "$trainOutBase\master_log.txt"
$iteration     = 30000
$basePort      = 6080

$scenes = @(
    "BaiZhang",
    "CaoMei2",
    "DouBanLv1",
    "DouBanLv3",
    "HongZhang",
    "KongQueZhuYu",
    "WangWenCao1",
    "WanNianQing1",
    "XiangPiShu1",
    "XianKeLai1",
    "XianKeLai3"
)

# ---- Init ----
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
& $venvActivate

New-Item -ItemType Directory -Path $trainOutBase -Force | Out-Null
New-Item -ItemType Directory -Path $meshOutBase  -Force | Out-Null

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
Log "  Standard 2DGS Full Pipeline"
Log ("  Repo: " + $repoDir)
Log "  Scenes: $totalScenes"
Log "  Training iterations: $iteration"
Log "  Params: depth_ratio=1, lambda_normal=0.05, lambda_dist=0 (defaults)"
Log ("  Training output: " + $trainOutBase)
Log ("  Mesh output:     " + $meshOutBase)
Log "========================================================================"

# ================================================================
#  PHASE 1: TRAINING
# ================================================================
Log ""
Log "================================================================"
Log ("  PHASE 1 / 3 : TRAINING (" + $totalScenes + " scenes x " + $iteration + " iters)")
Log "================================================================"

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

    # Check images_rgba
    $rgbaDir = "$srcDir\images_rgba"
    if (-not (Test-Path $rgbaDir) -or (Get-ChildItem $rgbaDir -File).Count -eq 0) {
        Log "  SKIP: No RGBA images"
        $trainResults[$scene] = "SKIP"
        continue
    }

    # Check if already completed
    if (Test-Path $plyPath) {
        $szMB = "{0:N1}" -f ((Get-Item $plyPath).Length / 1MB)
        Log ("  SKIP: Already completed (PLY=" + $szMB + " MB)")
        $trainResults[$scene] = "EXIST"
        continue
    }

    # Clean old output
    if (Test-Path $outDir) {
        Remove-Item $outDir -Recurse -Force
    }

    $sceneStart = Get-Date
    Log ("  Started at " + (Get-Date -Format 'HH:mm:ss'))

    # Standard 2DGS training command - all default params
    # depth_ratio=1 for median depth (better for bounded close-up scenes)
    # --images images_rgba to use SAM-masked RGBA (alpha channel = mask)
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

$phase1Elapsed = (Get-Date) - $phase1Start
$p1Str = FormatElapsed $phase1Elapsed
Log ""
Log ("---- PHASE 1 SUMMARY (" + $p1Str + ") ----")
foreach ($scene in $scenes) {
    $r = $trainResults[$scene]
    Log ("  " + $scene + " : " + $r)
}

$trainFails = @($trainResults.Values | Where-Object { $_ -like "FAIL*" })
if ($trainFails.Count -gt 0) {
    Log ("WARNING: " + $trainFails.Count + " scene(s) failed training.")
}

# ================================================================
#  PHASE 2: V3 CLEANING
# ================================================================
Log ""
Log "================================================================"
Log "  PHASE 2 / 3 : V3 CLEANING"
Log "================================================================"

$phase2Start = Get-Date

Log ("  Running clean_ply_v3.py --base_dir " + $trainOutBase + " --iterations " + $iteration)
$cleanLog = "$trainOutBase\clean_v3.log"

cmd /c "python `"$cleanScript`" --base_dir `"$trainOutBase`" --iterations $iteration > `"$cleanLog`" 2>&1"
$cleanExit = $LASTEXITCODE

$phase2Elapsed = (Get-Date) - $phase2Start
$p2Str = FormatElapsed $phase2Elapsed

if ($cleanExit -eq 0) {
    Log ("  V3 Cleaning completed in " + $p2Str)
    $lastLines = Get-Content $cleanLog -Tail 15
    foreach ($line in $lastLines) {
        Log ("    " + $line)
    }
} else {
    Log ("  V3 Cleaning FAILED (exit=" + $cleanExit + ") in " + $p2Str)
}

# ================================================================
#  PHASE 3: MESH EXTRACTION
# ================================================================
Log ""
Log "================================================================"
Log "  PHASE 3 / 3 : MESH EXTRACTION (TSDF + Marching Cubes)"
Log "================================================================"

$phase3Start = Get-Date
$meshResults = @{}

# Need to go back to standard 2DGS repo for render.py
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

    # Check v3 PLY
    if (-not (Test-Path $v3Ply)) {
        Log "  SKIP: V3 PLY not found"
        $meshResults[$scene] = "SKIP"
        continue
    }

    # Check if already completed
    $outMeshDir = "$meshOutBase\$scene"
    if (Test-Path "$outMeshDir\fuse_post.ply") {
        $szMB = "{0:N1}" -f ((Get-Item "$outMeshDir\fuse_post.ply").Length / 1MB)
        Log ("  SKIP: Already completed (fuse_post=" + $szMB + " MB)")
        $meshResults[$scene] = "EXIST"
        continue
    }

    $meshStart = Get-Date

    # Step 1: Backup + Swap v3 PLY
    if (-not (Test-Path $backupPly)) {
        Copy-Item $origPly $backupPly -Force
    }
    Copy-Item $v3Ply $origPly -Force
    Log "  Swapped V3 PLY as active point cloud"

    # Step 2: Clean old mesh output
    if (Test-Path $trainDir) {
        Remove-Item "$trainDir\fuse*.ply" -Force -ErrorAction SilentlyContinue
    }

    # Step 3: Extract mesh using standard 2DGS render.py
    # For bounded scenes: no --unbounded, use default auto-estimated params
    $meshLog = "$meshOutBase\${scene}_mesh.log"
    Log "  Running render.py (mesh_res=1024, num_cluster=50, depth_ratio=1)..."

    cmd /c "python render.py -m `"$trainOutBase\$scene`" -s `"$colmapBase\$scene`" --depth_ratio 1 --num_cluster 50 --mesh_res 1024 --iteration $iteration --skip_test --skip_train > `"$meshLog`" 2>&1"
    $meshExit = $LASTEXITCODE

    # Step 4: Copy results
    New-Item -ItemType Directory -Path $outMeshDir -Force | Out-Null
    $fusePly  = "$trainDir\fuse.ply"
    $fusePost = "$trainDir\fuse_post.ply"

    $ok = $true
    if (Test-Path $fusePly) {
        Copy-Item $fusePly $outMeshDir -Force
        $fuseSzMB = "{0:N1}" -f ((Get-Item $fusePly).Length / 1MB)
    } else {
        $fuseSzMB = "MISSING"; $ok = $false
    }
    if (Test-Path $fusePost) {
        Copy-Item $fusePost $outMeshDir -Force
        $postSzMB = "{0:N1}" -f ((Get-Item $fusePost).Length / 1MB)
    } else {
        $postSzMB = "MISSING"; $ok = $false
    }

    # Step 5: Restore original PLY
    if (Test-Path $backupPly) {
        Copy-Item $backupPly $origPly -Force
    }

    $meshElapsed = (Get-Date) - $meshStart
    $meshElStr = FormatElapsed $meshElapsed

    if ($ok) {
        Log ("  SUCCESS: " + $scene + " in " + $meshElStr + " (fuse=" + $fuseSzMB + " MB, post=" + $postSzMB + " MB)")
        $meshResults[$scene] = "OK | " + $meshElStr + " | post=" + $postSzMB + " MB"
    } else {
        Log ("  FAILED: " + $scene + " in " + $meshElStr + " (exit=" + $meshExit + ") -- check " + $meshLog)
        $meshResults[$scene] = "FAIL | " + $meshElStr
    }
}

Pop-Location

$phase3Elapsed = (Get-Date) - $phase3Start
$p3Str = FormatElapsed $phase3Elapsed

Log ""
Log ("---- PHASE 3 SUMMARY (" + $p3Str + ") ----")
foreach ($scene in $scenes) {
    $r = $meshResults[$scene]
    Log ("  " + $scene + " : " + $r)
}

Pop-Location

# ================================================================
#  FINAL SUMMARY
# ================================================================
$globalElapsed = (Get-Date) - $globalStart
$totalStr = FormatElapsed $globalElapsed

Log ""
Log "================================================================"
Log "  ALL PHASES COMPLETE"
Log "================================================================"
Log ""
$p1s = FormatElapsed $phase1Elapsed
$p2s = FormatElapsed $phase2Elapsed
$p3s = FormatElapsed $phase3Elapsed
Log ("Phase 1 (Training):        " + $p1s)
Log ("Phase 2 (V3 Cleaning):     " + $p2s)
Log ("Phase 3 (Mesh Extraction): " + $p3s)
Log "--------------------------------------"
Log ("Total elapsed:             " + $totalStr)
Log ""
Log ("Training results (" + $trainOutBase + "):")
foreach ($scene in $scenes) {
    $r = $trainResults[$scene]
    Log ("  " + $scene + " : " + $r)
}
Log ""
Log ("Mesh results (" + $meshOutBase + "):")
foreach ($scene in $scenes) {
    $r = $meshResults[$scene]
    Log ("  " + $scene + " : " + $r)
}
Log ""
Log ("Master log: " + $masterLog)
Log "Done."
