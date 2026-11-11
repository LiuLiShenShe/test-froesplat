"""Foreground photometry and structural proxies, explicitly distinct from semantic ground truth."""
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


def boundary_f1(prediction, mask, radius):
    a = prediction ^ ndimage.binary_erosion(prediction)
    b = mask ^ ndimage.binary_erosion(mask)
    if not a.any() or not b.any():
        return float(not a.any() and not b.any())
    precision = (ndimage.distance_transform_edt(~b)[a] <= radius).mean()
    recall = (ndimage.distance_transform_edt(~a)[b] <= radius).mean()
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def photometry(rgb, target, mask, r, alpha):
    mse = float(np.mean((rgb[mask]-target[mask])**2)) if mask.any() else 0.
    _, ssim_map = structural_similarity(rgb, target, data_range=1., channel_axis=2, full=True)
    thin = r['thin']
    dark = mask & (target @ np.array([.2126, .7152, .0722]) < .12)
    return dict(fg_mse=mse, fg_psnr=float(-10*np.log10(max(mse, 1e-12))),
        fg_ssim=float(ssim_map[mask].mean()) if mask.any() else 1.,
        fg_alpha=float(alpha[mask].mean()) if mask.any() else 1.,
        fg_coverage=float((alpha[mask] > .5).mean()) if mask.any() else 1.,
        boundary_coverage=float((alpha[r['boundary']] > .5).mean()) if r['boundary'].any() else 1.,
        skeleton_coverage=float((alpha[r['skeleton']] > .5).mean()) if r['skeleton'].any() else 1.,
        boundary_f1=boundary_f1(alpha > .5, mask, r['radius']),
        thin_coverage=float((alpha[thin] > .5).mean()) if thin.any() else 1.,
        dark_foreground_alpha=float(alpha[dark].mean()) if dark.any() else None,
        dark_foreground_mse=float(((rgb[dark]-target[dark])**2).mean()) if dark.any() else None,
        dark_foreground_pixels=int(dark.sum()))
