"""Pixel-scaled contour and skeleton protection; no density-based deletion."""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def regions(mask, relative_width):
    radius = max(1, round(relative_width * np.sqrt(mask.size)))
    inner_distance = ndimage.distance_transform_edt(mask)
    outer_distance = ndimage.distance_transform_edt(~mask)
    boundary = mask & (inner_distance <= radius)
    skeleton = skeletonize(mask)
    # Low local thickness selects narrow regions; includes thin leaves and petiole proxies.
    thickness = inner_distance[skeleton]
    threshold = max(2 * radius, float(np.quantile(thickness, .25))) if len(thickness) else 2 * radius
    thin = ndimage.binary_dilation(skeleton & (inner_distance <= threshold), iterations=radius) & mask
    return dict(fg=mask, bg=outer_distance > radius, boundary=boundary, thin=thin,
        skeleton=skeleton, radius=radius, thin_threshold_px=threshold)
