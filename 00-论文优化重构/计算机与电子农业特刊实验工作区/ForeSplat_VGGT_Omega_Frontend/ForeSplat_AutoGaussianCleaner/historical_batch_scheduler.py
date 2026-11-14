"""Overlap one scene's CPU delivery work with the next serial GPU scene."""
from concurrent.futures import ThreadPoolExecutor


def execute_pipeline(items, clean, postprocess, overlap=True):
    """Drain deliveries and propagate failures; retain at most two scenes.

    Only the calling thread invokes ``clean``. A single delivery worker may
    overlap it. Before submitting another delivery we join its predecessor,
    keeping temporary disk usage bounded even when cleaning is faster.
    """
    if not overlap:
        for item in items:
            clean(item)
            postprocess(item)
        return
    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='gggs-delivery') as worker:
        for item in items:
            if pending is not None and pending.done():
                pending.result()
            clean(item)
            if pending is not None:
                pending.result()
            pending = worker.submit(postprocess, item)
        if pending is not None:
            pending.result()
