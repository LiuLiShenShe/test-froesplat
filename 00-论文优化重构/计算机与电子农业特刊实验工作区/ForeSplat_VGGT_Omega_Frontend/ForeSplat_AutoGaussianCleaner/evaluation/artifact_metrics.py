import numpy as np
from scipy import ndimage


def artifact_metrics(alpha, rgb, regions):
    bg = regions['bg']
    occupied = bg & (alpha > .5)
    components, n = ndimage.label(occupied)
    sizes = np.bincount(components.ravel())[1:]
    return dict(bg_alpha=float(alpha[bg].mean()) if bg.any() else 0.,
        bg_alpha_mass_per_fg=float(alpha[bg].sum()/max(regions['fg'].sum(), 1)),
        bg_high_alpha_coverage=float(occupied.sum()/max(bg.sum(), 1)),
        bg_high_alpha_pixels=int(occupied.sum()),
        largest_bg_component_fraction=float(sizes.max()/alpha.size) if len(sizes) else 0.,
        dark_bg_high_alpha_pixels=int((occupied & (rgb @ np.array([.2126, .7152, .0722]) < .12)).sum()),
        dark_bg_high_alpha_fraction=float((occupied & (rgb @ np.array([.2126, .7152, .0722]) < .12)).sum()/alpha.size))
