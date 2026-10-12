import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_datasets import inverse_maps, remap_bilinear, remap_nearest
from phase20e_common import EXPECTED, FINAL, PROD, WORK, binary_mask, frozen_parameters, production, read_csv, save_mask, sha256


def test_r1_production_hashes_unchanged():
    for filename, value in EXPECTED.items():
        assert sha256(PROD / filename) == value


def test_full_frozen_parameters_only_operational_overrides():
    original = json.loads((FINAL / "logs/full_r1_runtime_parameters.json").read_text())
    frozen = frozen_parameters()
    assert original == frozen
    assert {key for key in frozen if frozen[key] != frozen_parameters("base", "cuda")[key]} == {"ranking_policy", "device"}


def test_half_pixel_identity_map():
    camera = pycolmap.Camera(model="PINHOLE", width=12, height=18, params=[15, 16, 6, 9])
    map_x, map_y, valid = inverse_maps(camera, camera)
    columns, rows = np.meshgrid(np.arange(12), np.arange(18))
    assert np.max(np.abs(map_x - columns)) < 1e-13
    assert np.max(np.abs(map_y - rows)) < 1e-13
    rgb = np.arange(18 * 12 * 3, dtype=np.uint8).reshape(18, 12, 3)
    assert np.array_equal(remap_bilinear(rgb, map_x, map_y, valid)[valid], rgb[valid])


def test_nearest_invalid_region_fixed_background():
    mask = np.ones((4, 4), bool)
    map_x = np.array([[1.1, 2.6], [-1, 20]], float)
    map_y = np.array([[2.2, 1.1], [2, 2]], float)
    valid = np.array([[True, True], [False, False]])
    result = remap_nearest(mask, map_x, map_y, valid)
    assert np.array_equal(result, valid)


def test_binary_mask_contract(tmp_path):
    path = tmp_path / "binary.png"
    original = np.array([[True, False], [False, True]])
    save_mask(path, original)
    assert np.array_equal(binary_mask(path), original)
    from PIL import Image
    Image.fromarray(np.array([[0, 127]], np.uint8)).save(path)
    with pytest.raises(ValueError):
        binary_mask(path)


def test_official_opencv_inverse_equation():
    source = pycolmap.Camera(model="OPENCV", width=50, height=70, params=[65, 64, 25, 35, .1, -.2, .002, -.003])
    target = pycolmap.Camera(model="PINHOLE", width=44, height=62, params=[65, 64, 22, 31])
    map_x, map_y, valid = inverse_maps(source, target)
    point = np.array([17.5, 21.5])
    ray = target.cam_from_img(point)
    expected = source.img_from_cam(np.append(ray, 1)) - .5
    assert np.allclose([map_x[21, 17], map_y[21, 17]], expected, atol=1e-12)


def test_production_selector_is_not_reimplemented():
    prod = production()
    assert prod.score_candidate.__module__ == "phase20e0_production"
    import ranking_policy
    assert sha256(Path(ranking_policy.__file__)) == EXPECTED["ranking_policy.py"]


def test_split_registration_identity_when_audit_exists():
    path = WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv"
    if not path.exists():
        pytest.skip("Full audit still executing")
    rows = read_csv(path)
    for sample in ("CaoMei2", "WanNianQing1", "HongZhang"):
        subset = [row for row in rows if row["sample"] == sample]
        assert len({row["registered_image_id"] for row in subset}) == len(subset)
        for index, row in enumerate(subset):
            assert row["split"] == ("validation" if index % 8 == 0 else "train")


def test_caomei_equal_stem_is_not_raw_identity():
    path = WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv"
    if not path.exists():
        pytest.skip("Full audit still executing")
    rows = [row for row in read_csv(path) if row["sample"] == "CaoMei2"]
    registered = {row["registered_name"]: row for row in rows}
    assert registered["crop_0146.png"]["raw_frame"] == "0170"
    assert "crop_0129.png" not in registered
    assert all(row["raw_frame"] != "0146" for row in rows)


def test_raw_recovery_preserves_corrupt_originals():
    path = WORK / "reports/SOURCE_RECOVERY_MANIFEST.csv"
    if not path.exists():
        pytest.skip("Recovery not completed")
    from phase20e_common import ROOT
    rows = read_csv(path)
    recovered = [row for row in rows if row["sample"] == "HongZhang"]
    assert len(recovered) == 12
    for row in recovered:
        assert sha256(ROOT / row["original_corrupt_file_preserved"]) == row["original_corrupt_sha256"]
        assert sha256(row["recovered_path"]) == row["sha256"]


def test_recovery_replay_preserves_original_provenance(tmp_path, monkeypatch):
    import recover_verified_sources
    stage = tmp_path / "stage"
    (stage / "reports").mkdir(parents=True)
    (stage / "reports/SOURCE_RECOVERY_STATUS.json").write_text(json.dumps({"status": "PASS", "registered_views": 1}))
    (stage / "reports/SOURCE_RECOVERY_MANIFEST.csv").touch()
    original = {"original_raw_rgb_path": "corrupt-original.jpg", "raw_rgb_path": "recovered.jpg",
                "raw_rgb_error": "", "common_raw_domain_proven": "True", "source_recovery": "VERIFIED_RECOVERY"}
    snapshot = original.copy()
    monkeypatch.setattr(recover_verified_sources, "WORK", stage)
    monkeypatch.setattr(recover_verified_sources, "read_csv", lambda path: [] if path.name == "SOURCE_RECOVERY_MANIFEST.csv" else [original])
    monkeypatch.setattr(recover_verified_sources, "write_csv", lambda *args: pytest.fail("A valid recovery replay must not rewrite provenance"))
    recover_verified_sources.run()
    assert original == snapshot


def test_no_gt_enters_gpu_inference_or_ranking():
    folder = WORK / "scripts"
    for filename in ("generate_candidates.py", "rank_candidates.py", "build_datasets.py"):
        source = (folder / filename).read_text()
        assert "derived_masks" not in source
        assert "GT_potted" not in source
        assert "rasterize" not in source


def test_preflight_never_trains_optimizer():
    source = (WORK / "scripts/preflight.py").read_text()
    assert ".training_setup(" not in source
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "torch.no_grad()" in source
