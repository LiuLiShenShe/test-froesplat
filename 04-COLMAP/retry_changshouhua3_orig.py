import shutil, os, subprocess, struct, glob

folder = 'D:/CAAS/04-COLMAP/ChangShouHua3'
colmap = 'D:/CAAS/04-COLMAP/colmap_bin/bin/colmap.exe'
sam_dir = 'D:/CAAS/03-SAM/ChangShouHua3'
fft_dir = 'D:/CAAS/02-FFT/ChangShouHua3'
orig_dir = 'D:/CAAS/01-FFmepg/ChangShouHua3'

# Clear old results
for d in ['distorted', 'sparse', 'images', 'stereo']:
    p = os.path.join(folder, d)
    if os.path.exists(p):
        shutil.rmtree(p)
for f in ['run-colmap-geometric.sh', 'run-colmap-photometric.sh']:
    p = os.path.join(folder, f)
    if os.path.exists(p):
        os.remove(p)

# Clear and repopulate input/ with ORIGINAL images (not crop)
input_dir = os.path.join(folder, 'input')
if os.path.exists(input_dir):
    shutil.rmtree(input_dir)
os.makedirs(input_dir)

# Get FFT-kept filenames, copy originals
fft_files = [f for f in os.listdir(fft_dir) if f.endswith('.jpg')]
for f in fft_files:
    src = os.path.join(orig_dir, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(input_dir, f))

n_input = len(os.listdir(input_dir))
print(f'Input images (originals): {n_input}')

# Run COLMAP
convert_py = 'D:/CAAS/05-2d-gaussian-splatting-great-again-dev/convert.py'
python_exe = 'D:/CAAS/05-2d-gaussian-splatting-great-again-dev/.venv_uv/Scripts/python.exe'
cmd = [python_exe, convert_py, '-s', folder, '--colmap_executable', colmap, '--no_gpu']
print('Running COLMAP with original images...')
result = subprocess.run(cmd, capture_output=True, text=True)
stdout_tail = result.stdout[-500:] if result.stdout else 'None'
stderr_tail = result.stderr[-500:] if result.stderr else 'None'
print('STDOUT:', stdout_tail)
print('STDERR:', stderr_tail)
print('Return code:', result.returncode)

# Check result
sparse_images = os.path.join(folder, 'sparse/0/images.bin')
sparse_pts = os.path.join(folder, 'sparse/0/points3D.bin')
if os.path.exists(sparse_images):
    with open(sparse_images, 'rb') as fh:
        n_reg = struct.unpack('<Q', fh.read(8))[0]
    with open(sparse_pts, 'rb') as fh:
        n_pts = struct.unpack('<Q', fh.read(8))[0]
    pct = n_reg / n_input * 100
    print(f'Result: {n_reg}/{n_input} ({pct:.1f}%), {n_pts} pts')
else:
    print('FAILED - no sparse output')
