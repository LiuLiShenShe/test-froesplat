"""Independent pixel-centre mapping fix; original frontend remains untouched.

Use with saved standard VGGT K_model and original RGB/depth. This does not
change focal predictions, camera poses, model weights, or image content.
"""
import numpy as np

def resize_affine(source_wh, target_wh):
    sx,sy=np.array(target_wh,dtype=float)/np.array(source_wh,dtype=float)
    return np.array([[sx,0,.5*sx-.5],[0,sy,.5*sy-.5],[0,0,1.]])

def square_model_affine(original_wh, model_resolution=518):
    w,h=map(int,original_wh);side=max(w,h)
    pad=np.array([[1,0,(side-w)//2],[0,1,(side-h)//2],[0,0,1.]])
    # Two align_corners=False resize operations compose to the same affine.
    return resize_affine((side,side),(model_resolution,model_resolution))@pad

def restore_intrinsics(K_model, original_wh, model_resolution=518):
    return np.linalg.inv(square_model_affine(original_wh,model_resolution))@np.asarray(K_model)

def training_intrinsics(K_original, original_wh, actual_training_wh):
    return resize_affine(original_wh,actual_training_wh)@np.asarray(K_original)

def projection_from_intrinsics(K,width,height,znear=.01,zfar=100.):
    """Row-independent mathematical P matching CUDA integer pixel centres.

    GGGS's ray-plane CUDA kernels still hardcode centred rays. This P is a
    projection test helper, not a claim of full GGGS compatibility or repair.
    """
    K=np.asarray(K);P=np.zeros((4,4))
    P[0,0]=2*K[0,0]/width;P[1,1]=2*K[1,1]/height
    P[0,2]=(2*K[0,2]+1)/width-1;P[1,2]=(2*K[1,2]+1)/height-1
    P[3,2]=1;P[2,2]=zfar/(zfar-znear);P[2,3]=-zfar*znear/(zfar-znear)
    return P
