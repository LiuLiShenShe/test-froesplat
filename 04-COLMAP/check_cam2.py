import struct, os

def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack('<i', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            np_ = {0:3,1:4,2:4,3:5,4:4,5:5,6:4,7:8,8:12,9:13}.get(model_id,4)
            params = struct.unpack(f'<{np_}d', f.read(8*np_))
            model_name = {0:'SIMPLE_PINHOLE',1:'PINHOLE',2:'SIMPLE_RADIAL',3:'RADIAL',4:'OPENCV',5:'OPENCV_FISHEYE'}.get(model_id, f'MODEL_{model_id}')
            cameras[cid] = {'model': model_name, 'model_id': model_id, 'w': w, 'h': h, 'params': params}
    return cameras

# Check BOTH distorted and undistorted for CaoMei1
for label, subdir in [('DISTORTED', 'distorted/sparse/0'), ('UNDISTORTED', 'sparse/0')]:
    path = f'D:\\CAAS\\04-COLMAP\\CaoMei1\\{subdir}\\cameras.bin'
    if os.path.exists(path):
        cams = read_cameras_bin(path)
        for cid, cam in cams.items():
            print(f"{label}: cam{cid} model={cam['model']}(id={cam['model_id']}) {cam['w']}x{cam['h']}")
            print(f"  params: {cam['params']}")
    else:
        print(f'{label}: NOT FOUND')
