"""A wrong renderer or stale screenshot must never satisfy the batch gate."""
import importlib.util
import json
from pathlib import Path
spec=importlib.util.spec_from_file_location('viewer_review_validity',Path(__file__).parents[1]/'viewer_review_validity.py')
review_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(review_module)


def fixture(tmp_path):
    folder=tmp_path/'comparison/official_supersplat';folder.mkdir(parents=True)
    models=[]
    for label in ['B0_original','B2_stage1','B3_stage2']:
        model=tmp_path/(label+'.ply');model.write_bytes(label.encode())
        captures=[]
        for i in range(8):
            for bg in ['white','black']:
                path=folder/f'{label}_{i}_{bg}.png';path.write_bytes(f'{label}{i}{bg}'.encode())
                captures.append(dict(view=str(i),background=bg,path=str(path),sha256=review_module.digest(path),position_error=1e-8))
        models.append(dict(label=label,ply=str(model),sha256=review_module.digest(model),captures=captures))
    capture=dict(editor_build_sha256='compatible-build',renderer_conformance=dict(passed=True,editor_build_sha256='compatible-build'),models=models)
    direct=dict(direct_image_inspection=True,renderer_build_sha256='compatible-build',reviewed_models=[m['label'] for m in models],eligible_for_batch=True)
    (folder/'REVIEW.json').write_text(json.dumps(capture));(folder/'DIRECT_IMAGE_REVIEW.json').write_text(json.dumps(direct))
    return folder,capture,direct


def test_actual_review_requires_opacity_conformance_for_the_same_build(tmp_path):
    folder,capture,direct=fixture(tmp_path)
    assert review_module.validated_review(tmp_path)
    capture['renderer_conformance']['passed']=False
    (folder/'REVIEW.json').write_text(json.dumps(capture))
    assert review_module.validated_review(tmp_path) is None
    capture['renderer_conformance'].update(passed=True,editor_build_sha256='another-build')
    (folder/'REVIEW.json').write_text(json.dumps(capture))
    assert review_module.validated_review(tmp_path) is None


def test_stale_image_and_unreviewed_model_cannot_pass(tmp_path):
    folder,capture,direct=fixture(tmp_path)
    image=Path(capture['models'][0]['captures'][0]['path']);original=image.read_bytes();image.write_bytes(b'changed screenshot')
    assert review_module.validated_review(tmp_path) is None
    image.write_bytes(original)
    direct['reviewed_models'].remove('B3_stage2');(folder/'DIRECT_IMAGE_REVIEW.json').write_text(json.dumps(direct))
    assert review_module.validated_review(tmp_path) is None
