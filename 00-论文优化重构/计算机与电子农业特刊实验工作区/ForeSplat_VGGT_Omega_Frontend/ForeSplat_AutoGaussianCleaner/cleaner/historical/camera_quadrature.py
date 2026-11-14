"""Representative cameras retain the population weight of the full trajectory."""
import numpy as np


def representative_weights(scene,indices):
    """Voronoi cell masses in the same pose features used by camera FPS.

    Farthest-point sampling covers rare viewpoints deliberately. Equal weights
    would also amplify their contribution mass, which changes the historical
    foreground/background ratio. Assign every original camera to its closest
    representative instead; no renderer, image, point ID or plant ROI is used.
    """
    indices=np.asarray(indices,dtype=np.int64)
    if not len(indices):return np.empty(0,np.float64)
    count=len(scene.views)
    if (len(np.unique(indices))!=len(indices) or indices.min()<0 or indices.max()>=count):
        raise ValueError('Representative camera indices must be distinct valid scene rows')
    if len(indices)==count:return np.full(count,1/count)
    E=np.array([v['E'] for v in scene.views])
    centers=np.linalg.inv(E)[:,:3,3]
    extent=max(float(np.linalg.norm(np.ptp(centers,axis=0))),1e-9)
    features=np.column_stack([centers/extent,E[:,2,:3]])
    distances=((features[:,None,:]-features[indices][None,:,:])**2).sum(2)
    closest=distances.argmin(1)
    return np.bincount(closest,minlength=len(indices)).astype(np.float64)/count
