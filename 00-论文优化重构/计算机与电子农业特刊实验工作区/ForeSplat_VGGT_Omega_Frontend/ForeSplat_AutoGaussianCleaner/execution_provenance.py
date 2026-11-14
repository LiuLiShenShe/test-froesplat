"""Read-only GGGS renderer provenance and lightweight native PLY inventory."""
from pathlib import Path
import sys
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha

ARTIFACT_FILES=['cleaned_gaussians.ply','cleaned_gaussians_supersplat.ply','removed_gaussians.ply',
    'removed_gaussians_supersplat.ply','original_gaussian_ids.npy','removed_original_gaussian_ids.npy','viewer_baking.json']


def resolve_variant(method,configuration):
    variant=configuration.get('production_variant','C5-D') if method=='C5' else method
    if variant not in ['C5-A','C5-B','C5-C','C5-D']:
        raise ValueError(f'Unsupported configured GGGS cleaner variant: {variant}')
    if method=='C5' and variant not in ['C5-C','C5-D']:
        raise ValueError('Production C5 must use a quality-certified grouped C/D variant; request A/B explicitly for ablations')
    return variant


def assert_output_separate(output,spec):
    output=Path(output).resolve()
    for key in ['model','dataset','geometry_cameras','geometry_dir','gggs_repo']:
        if spec.get(key) and Path(spec[key]).resolve().is_relative_to(output):
            raise ValueError(f'Output contains an original input ({key}); choose an independent output directory')


def gaussian_count(path):
    path=Path(path)
    if not path.exists():return float('inf')
    with path.open('rb') as f:
        if f.readline().strip()!=b'ply':raise ValueError(f'Invalid native PLY header: {path}')
        for _ in range(256):
            line=f.readline().strip()
            if line.startswith(b'element vertex '):return int(line.split()[-1])
            if line==b'end_header':break
    raise ValueError(f'Missing Gaussian count: {path}')


def native_renderer_fingerprint(repositories):
    files=set()
    for repository in repositories:
        root=Path(repository)
        for name in ['scene/gaussian_model.py','scene/colmap_loader.py','gaussian_renderer/__init__.py','utils/general_utils.py','utils/sh_utils.py','utils/graphics_utils.py','utils/cuda_preload.py']:
            path=root/name
            if not path.exists():raise FileNotFoundError(path)
            files.add(path.resolve())
        raster=root/'corrected_rasterizer'
        for folder in [raster,raster/'cuda_rasterizer',raster/'diff_gaussian_rasterization']:
            for path in folder.iterdir():
                if path.is_file() and path.suffix in ['.cu','.cpp','.h','.py','.so']:files.add(path.resolve())
    # Include the installed binary used by the uv environment as well as its sources.
    for search in sys.path:
        folder=Path(search)/'diff_gaussian_rasterization'
        if folder.is_dir():files.update(p.resolve() for p in folder.glob('*_C*.so'))
    return {str(path):sha(path) for path in sorted(files)}


def scene_input_fingerprint(spec):
    """Hash source contents; path equality cannot certify an unchanged cache."""
    dataset=Path(spec['dataset']).resolve()
    files=set()
    for name in ['images','masks']:
        folder=dataset/name
        if not folder.is_dir():raise FileNotFoundError(folder)
        files.update(p.resolve() for p in folder.iterdir() if p.is_file())
    sparse=dataset/'sparse/0'
    for name in ['cameras.bin','images.bin','cameras.txt','images.txt']:
        path=sparse/name
        if path.exists():files.add(path.resolve())
    if not any((sparse/n).exists() for n in ['cameras.bin','cameras.txt']):
        raise FileNotFoundError(f'Missing GGGS camera interface: {sparse}')
    if spec.get('geometry_cameras'):
        files.add(Path(spec['geometry_cameras']).resolve())
        import json
        meta=json.loads(Path(spec['geometry_cameras']).read_text())
        for frame in meta['frames']:
            stem=frame.get('stem',Path(frame['image_name']).stem)
            for prefix in ['depth','confidence']:
                files.add((Path(spec['geometry_dir'])/f'{prefix}_{stem}.npy').resolve())
    return {str(path):sha(path) for path in sorted(files)}


def completed_identity(output,config,spec,variant):
    """A completion marker is valid only for this exact requested execution."""
    import json
    output=Path(output)
    old=json.loads((output/'COMPLETE.json').read_text())
    artifacts=old.get('artifact_hashes',{})
    if any(not (output/name).is_file() or artifacts.get(name)!=sha(output/name) for name in ARTIFACT_FILES):
        return False
    actual=old.get('variant')
    if actual is None and (output/'execution.json').exists():
        actual=json.loads((output/'execution.json').read_text()).get('variant')
    return (actual==variant and old.get('config')==config and old.get('scene')==spec
            and old.get('input_sha256')==sha(spec['model'])
            and old.get('native_sha256')==sha(output/'cleaned_gaussians.ply'))


def invalidate_output(output,reason):
    from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json
    output=Path(output)
    write_json(output/'INVALIDATED.json',dict(reason=reason,certified=False))
    marker=output/'COMPLETE.json'
    if marker.exists():marker.replace(output/'INVALIDATED_COMPLETE.json')
