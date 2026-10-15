"""Stage-only seed control and observation; production training code is untouched."""
from __future__ import annotations
import hashlib
import json
import pickle
import random
from pathlib import Path
from common import E0, GS, GPU_INDEX, GPU_UUID, STAGE, e0_config, read_csv, read_json, sha256


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_hashes():
    import numpy as np
    import torch
    states = {
        'python': hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        'numpy': hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        'torch_cpu': tensor_sha(torch.get_rng_state()),
    }
    states.update({f'torch_cuda_{index}': tensor_sha(state) for index, state in enumerate(torch.cuda.get_rng_state_all())})
    states['combined'] = hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest()
    return states


def set_seed(seed):
    """Native path for seed 0; substitute its three zero seeds for other seeds.

    torch.manual_seed already seeds CPU and all CUDA devices in installed 2.5.1.
    Explicit CUDA reseeding is tested to leave the same states, then used only for
    nonzero seeds. No deterministic algorithm or kernel settings are changed.
    """
    import numpy as np
    import torch
    from utils.general_utils import safe_state
    safe_state(False)
    if seed != 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return rng_hashes()


def config_invariant(config):
    import copy
    value = copy.deepcopy(config)
    value.pop('seed', None)
    value.pop('e2', None)
    for section in ('model', 'all_parameters'):
        value[section].pop('source_path', None)
        value[section].pop('model_path', None)
    return value


def assert_gpu():
    import os
    import subprocess
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_INDEX
    actual = subprocess.check_output(['nvidia-smi', '-i', GPU_INDEX, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert actual == GPU_UUID, actual


def verify_scene(scene, config, sample):
    import numpy as np
    from plyfile import PlyData
    split = [r for r in read_csv(E0 / 'TRAIN_VALIDATION_SPLIT.csv') if r['sample'] == sample]
    ordered = {}
    for label, cameras in [('train', scene.getTrainCameras()), ('validation', scene.getTestCameras())]:
        actual = [c.image_name for c in cameras]
        expected = [Path(r['registered_name']).stem for r in split if r['split'] == label]
        assert len(actual) == len(expected) and set(actual) == set(expected), label
        ordered[label] = actual
    sparse = Path(config['model']['source_path']) / 'sparse/0/points3D.ply'
    assert sha256(Path(config['model']['model_path']) / 'input.ply') == sha256(sparse)
    points = PlyData.read(sparse)['vertex']
    xyz = np.stack([points[k] for k in ('x', 'y', 'z')], axis=1)
    assert np.array_equal(scene.gaussians.get_xyz.detach().cpu().numpy(), xyz)
    fields = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
    return {'ordered_cameras': ordered,
            'ordered_camera_sha256': hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest(),
            'initial_ply_sha256': sha256(sparse), 'initial_gaussians': len(xyz),
            'initial_parameter_sha256': {k: tensor_sha(getattr(scene.gaussians, k)) for k in fields},
            'post_scene_rng_sha256': rng_hashes(), 'camera_extent': float(scene.cameras_extent)}


def assert_pair(left, right):
    for key in ('initial_rng_sha256', 'ordered_camera_sha256', 'initial_ply_sha256',
                'initial_parameter_sha256', 'post_scene_rng_sha256', 'pre_optimizer_rng_sha256',
                'post_optimizer_rng_sha256'):
        assert left[key] == right[key], f'B/C initialization mismatch: {key}'


def logger_compatibility():
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not hasattr(FigureCanvasAgg, 'tostring_rgb'):
        def tostring_rgb(canvas):
            return np.asarray(canvas.buffer_rgba())[..., :3].tobytes()
        FigureCanvasAgg.tostring_rgb = tostring_rgb
