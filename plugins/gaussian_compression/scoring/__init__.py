"""Only audited score implementations may be selected by method name."""
from pathlib import Path
import numpy as np
from ..io import sha256


class AdaptationBlocked(RuntimeError):
    pass


METHODS = {
    'random': {'status':'IMPLEMENTED_BASELINE', 'requires_cameras':False},
    'opacity': {'status':'IMPLEMENTED_BASELINE', 'requires_cameras':False},
    'external': {'status':'IMPLEMENTED_WITH_PROVENANCE', 'requires_cameras':'SCORER_DEPENDENT'},
    'PUP_2DGS_ADAPTED': {'status':'PUP_ADAPTATION_BLOCKED', 'requires_cameras':True},
    'LIGHTGAUSSIAN_2DGS_ADAPTED': {'status':'LIGHTGAUSSIAN_ADAPTATION_BLOCKED', 'requires_cameras':True},
}


def score_model(model, method, config, source_sha):
    if method in ['PUP_2DGS_ADAPTED','pup']:
        raise AdaptationBlocked('PUP_ADAPTATION_BLOCKED: native 2DGS has no per-pixel 5-parameter Fisher CUDA accumulator; no heuristic substitute is registered')
    if method in ['LIGHTGAUSSIAN_2DGS_ADAPTED','lightgaussian']:
        raise AdaptationBlocked('LIGHTGAUSSIAN_ADAPTATION_BLOCKED: native 2DGS has no accepted-pixel hit/opacity per-Gaussian count-render output and no agreed surface replacement for 3D volume')
    if method == 'random':
        seed = config.get('seed',0)
        if isinstance(seed,bool) or not isinstance(seed,int) or seed < 0:
            raise ValueError('Random seed must be a nonnegative integer')
        values = np.random.Generator(np.random.PCG64(seed)).random(model.count)
        provenance = {'scorer':'numpy.random.Generator(PCG64).random','seed':seed,
                      'numpy_version':np.__version__,'baseline_only':True}
    elif method == 'opacity':
        # Sigmoid is strictly monotone over the real numbers. Ranking the stored
        # logits avoids numerical saturation ties in activated opacity.
        values = model.vertices['opacity'].astype(np.float64)
        provenance = {'scorer':'stored opacity logit (monotone sigmoid ordering)',
                      'baseline_only':True}
    elif method == 'external':
        provenance = dict(config.get('score_provenance',{}))
        if not provenance.get('scorer') or provenance.get('source_model_sha256') != source_sha:
            raise ValueError('External score provenance must name its scorer and bind to this input PLY SHA256')
        path = Path(config['scores_path']).resolve(strict=True)
        actual = sha256(path)
        if actual != provenance.get('scores_sha256'):
            raise ValueError('External score file SHA256 mismatch')
        values = np.load(path,allow_pickle=False)
        provenance.update(scores_path=str(path))
    else:
        raise ValueError(f'Unknown compression method: {method}')
    values = np.asarray(values)
    if values.shape != (model.count,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError('Importance scores must be finite real values with shape (input_count,)')
    return values, {**provenance,'source_model_sha256':source_sha,'higher_is_more_important':True}
