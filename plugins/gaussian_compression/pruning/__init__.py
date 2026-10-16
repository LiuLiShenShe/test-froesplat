"""Score-independent deterministic Gaussian deletion."""
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import numpy as np


def removal_budget(count, prune_ratio):
    if isinstance(prune_ratio, bool):
        raise ValueError('prune_ratio must be numeric, not boolean')
    try:
        ratio = Decimal(str(prune_ratio))
    except (InvalidOperation, ValueError) as error:
        raise ValueError('Invalid prune_ratio') from error
    if not ratio.is_finite() or not (0 <= ratio < 1):
        raise ValueError('prune_ratio must be finite and in [0,1)')
    if count < 1:
        raise ValueError('Empty input model')
    return int((ratio * count).to_integral_value(rounding=ROUND_FLOOR))


def rank_and_select(scores, prune_ratio):
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype.kind not in 'fiu' or not np.isfinite(scores).all():
        raise ValueError('One finite real importance score per original Gaussian is required')
    removed_count = removal_budget(len(scores), prune_ratio)
    # Stable descending order resolves equal scores by ascending original index.
    key = np.bitwise_not(scores) if scores.dtype.kind in 'iu' else -scores
    order = np.argsort(key, kind='stable')
    retained = np.sort(order[:len(scores)-removed_count]).astype(np.int64)
    removed = np.sort(order[len(scores)-removed_count:]).astype(np.int64)
    return retained, removed
