import shutil, os, subprocess, struct

folder = 'D:/CAAS/04-COLMAP/ChangShouHua3'
colmap = 'D:/CAAS/04-COLMAP/colmap_bin/bin/colmap.exe'

# Clear old results but keep input/
for d in ['distorted', 'sparse', 'images', 'stereo']:
    p = os.path.join(folder, d)
    if os.path.exists(p):
        shutil.rmtree(p)
for f in ['run-colmap-geometric.sh', 'run-colmap-photometric.sh']:
    p = os.path.join(folder, f)
    if os.path.exists(p):
        os.remove(p)

# input/ already has crop images from rerun_crop.py, keep as is
input_dir = os.path.join(folder, 'input')
n_input = len(os.listdir(input_dir))
print(f'Input images: {n_input}')

# Run convert.py
convert_py = 'D:/CAAS/05-2d-gaussian-splatting-great-again-dev/convert.py'
cmd = [
    'D:/CAAS/05-2d-gaussian-splatting-great-again-dev/.venv_uv/Scripts/python.exe',
    convert_py,
    '-s', folder,
    '--colmap_executable', colmap,
    '--no_gpu'
]
print('Running COLMAP...')
result = subprocess.run(cmd, capture_output=True, text=True)
print('STDOUT:', result.stdout[-500:] if result.stdout else 'None')
print('STDERR:', result.stderr[-500:] if result.stderr else 'None')
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
