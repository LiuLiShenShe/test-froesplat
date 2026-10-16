"""Read frozen cameras through the original Scene; redirect new models externally."""
from common import *
from argparse import Namespace


def cameras(p):
    import_paths()
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    g=GaussianModel(p['SH_degree'])
    s=Scene(Namespace(**p['model']),g,load_iteration=30000,shuffle=False)
    heldout=sorted(s.getTestCameras(),key=lambda c:c.image_name)
    assert [c.image_name for c in heldout]==p['heldout_camera_names']
    assert all((c.image_width,c.image_height)==(1060,1900) for c in heldout)
    # Keep camera objects but release the source model so memory benchmarking
    # contains only one live model. Scene(load_iteration) does not write files.
    s.gaussians=None
    del g
    return s,heldout


def benchmark_cameras(p):
    """Load only the held-out views plus one train view for the native Scene contract."""
    gate=STAGE/'configs/BENCHMARK_CAMERA_GATE.txt'
    names=sorted(set(p['heldout_camera_names']) | {'0001'})
    gate.write_text('\n'.join(names) + '\n')
    model_cfg={**p['model'],'raw_gate_list':str(gate)}
    import_paths()
    from argparse import Namespace
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    g=GaussianModel(p['SH_degree'])
    s=Scene(Namespace(**model_cfg),g,load_iteration=30000,shuffle=False)
    heldout=sorted(s.getTestCameras(),key=lambda c:c.image_name)
    assert [c.image_name for c in heldout]==p['heldout_camera_names']
    assert all((c.image_width,c.image_height)==(1060,1900) for c in heldout)
    assert len(s.getTrainCameras())==1
    s.gaussians=None
    del g
    return s,heldout,gate


def reference_records(p):
    refs={r['loader_image_name']:r for r in rows(p['reference_manifest']) if r['sample']==p['scene']}
    roi={Path(r['registered_name']).stem:r for r in rows(p['ROI_manifest']) if r['sample']==p['scene'] and r['group']==p['ROI_group'] and r['split']=='validation'}
    assert sorted(refs)==sorted(roi)==p['heldout_camera_names']
    for name,r in refs.items():
        assert sha(r['reference_rgb_path'])==r['common_rgb_sha256']
        assert sha(roi[name]['mask_path'])==roi[name]['mask_sha256']
    return refs,roi
