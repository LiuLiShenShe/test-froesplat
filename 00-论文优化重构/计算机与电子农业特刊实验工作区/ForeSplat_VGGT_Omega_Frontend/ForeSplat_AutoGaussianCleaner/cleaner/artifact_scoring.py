"""Shared color-independent evidence rules for A1--A4."""
import numpy as np


def classify(e, config):
    fg, bg = e['fg'], e['bg']
    harm = np.divide(bg, fg+bg, out=np.zeros_like(bg), where=fg+bg > 0)
    observed = e['visible_views'] >= 2
    meaningful = bg > 1e-7
    protected = (e['boundary']+e['thin']) / np.maximum(fg, 1e-15) > .2
    # Any measured contour/thin support contradicts a whole-point invalidity classification.
    # Route these Gaussians to constrained covariance refinement even when background dominates.
    a2 = observed & meaningful & (harm > config['prune_background_fraction']) & ~protected
    if config['geometry']:
        a2 &= (e['geometry_views'] < 2) & (e['occlusion_views'] == 0)
        a2 |= observed & meaningful & (harm > config['geometry_prune_background_fraction']) & (e['geometry_views'] == 0) & (e['occlusion_views'] == 0) & ~protected
    a3 = observed & meaningful & (fg > 1e-7) & (harm > config['refine_background_fraction']) & ~a2
    a1 = (e['fg_views'] >= 2) & (harm < config['refine_background_fraction'])
    labels = np.full(len(fg), 4, np.uint8)
    labels[a1], labels[a2], labels[a3] = 1, 2, 3
    # Rank pruning by foreground sacrifice per eliminated background contribution.
    ids = np.flatnonzero(a2)
    ids = ids[np.argsort(fg[ids] / np.maximum(bg[ids], 1e-15), kind='stable')]
    budget = config['prune_foreground_budget'] * fg.sum()
    ids = ids[np.cumsum(fg[ids]) <= budget]
    return dict(labels=labels, harm=harm, prune=ids, mixed=np.flatnonzero(a3),
        counts={f'A{i}': int((labels == i).sum()) for i in range(1, 5)},
        protected=protected, foreground_budget=budget)
