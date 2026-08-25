import struct, os

# Correct number of params per COLMAP camera model
COLMAP_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,   # RADIAL: f, cx, cy, k1, k2
    4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
    6: 14,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}

MODEL_NAMES = {0:'SIMPLE_PINHOLE',1:'PINHOLE',2:'SIMPLE_RADIAL',3:'RADIAL',4:'OPENCV',5:'OPENCV_FISHEYE',6:'FULL_OPENCV'}

def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack('<i', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            np_ = COLMAP_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack(f'<{np_}d', f.read(8*np_))
            cameras[cid] = {'model': MODEL_NAMES.get(model_id, f'M{model_id}'), 'model_id': model_id, 'w': w, 'h': h, 'params': params}
    return cameras

rerun_folders = ['CaoMei1','ChangShouHua2','DouBanLv1','DouBanLv2','HongZhang','WanNianQing1','WanNianQing2','WangWenCao2','XianKeLai2']

for folder in rerun_folders:
    dist_path = f'D:\\CAAS\\04-COLMAP\\{folder}\\distorted\\sparse\\0\\cameras.bin'
    und_path = f'D:\\CAAS\\04-COLMAP\\{folder}\\sparse\\0\\cameras.bin'
    
    print(f"\n{'='*60}")
    print(f"{folder}:")
    
    if os.path.exists(dist_path):
        cams = read_cameras_bin(dist_path)
        for cid, cam in cams.items():
            print(f"  DIST: {cam['model']} {cam['w']}x{cam['h']}")
            if cam['model'] == 'OPENCV':
                fx, fy, cx, cy, k1, k2, p1, p2 = cam['params']
                print(f"    fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
                print(f"    k1={k1:.6f} k2={k2:.6f} p1={p1:.6f} p2={p2:.6f}")
            else:
                print(f"    params={cam['params']}")
    
    if os.path.exists(und_path):
        cams = read_cameras_bin(und_path)
        for cid, cam in cams.items():
            print(f"  UNDIST: {cam['model']} {cam['w']}x{cam['h']}")
            print(f"    params={cam['params']}")
