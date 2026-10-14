"""Regression tests for publication-safe, read-only acceptance checks."""
import copy
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acceptance import changed_tracked_paths, metric_recomputation_passes


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.views = [{"sample": "HongZhang", "group": "C_R1", "registered_name": "0000.jpg",
                       "status": "PASS", "shared_model_alias": "",
                       "PSNR": "5.25", "SSIM": "0.3", "LPIPS": "0.6"}]
        alias = dict(self.views[0], group="B_R0", shared_model_alias="IDENTICAL TRAINING INPUT")
        self.views.append(alias)
        self.recomputed = [{"sample": "HongZhang", "group": "C_R1", "registered_name": "0000.jpg",
                            "metric": metric, "original": self.views[0][metric],
                            "recomputed": self.views[0][metric], "abs_difference": "0.0", "status": "PASS"}
                           for metric in ("PSNR", "SSIM", "LPIPS")]

    def test_complete_recomputation_excludes_shared_model_alias(self):
        self.assertTrue(metric_recomputation_passes(self.views, self.recomputed))

    def test_missing_or_duplicate_metric_is_rejected(self):
        self.assertFalse(metric_recomputation_passes(self.views, self.recomputed[:-1]))
        self.assertFalse(metric_recomputation_passes(self.views, self.recomputed + self.recomputed[:1]))

    def test_stale_original_metric_is_rejected(self):
        rows = copy.deepcopy(self.recomputed)
        rows[0].update(original="5.5", recomputed="5.5")
        self.assertFalse(metric_recomputation_passes(self.views, rows))

    def test_unreported_recomputation_error_is_rejected(self):
        rows = copy.deepcopy(self.recomputed)
        rows[0]["recomputed"] = "5.2501"
        self.assertFalse(metric_recomputation_passes(self.views, rows))

    def test_nonfinite_metric_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            rows = copy.deepcopy(self.recomputed)
            rows[0]["recomputed"] = str(value)
            self.assertFalse(metric_recomputation_passes(self.views, rows))

    def test_small_recorded_roundoff_is_accepted(self):
        rows = copy.deepcopy(self.recomputed)
        rows[0]["recomputed"] = "5.2500001"
        rows[0]["abs_difference"] = str(abs(float(rows[0]["recomputed"]) - float(rows[0]["original"])))
        self.assertTrue(metric_recomputation_passes(self.views, rows))

    def test_git_paths_keep_chinese_and_linebreaks_after_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "--allow-empty", "-qm", "baseline"], check=True)
            paths = ["阶段二十E1/报告.md", "阶段二十E1/line\nbreak.md"]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test\n")
            subprocess.run(["git", "-C", str(root), "add", "--", *paths], check=True)
            self.assertEqual(set(changed_tracked_paths(root, "HEAD")), set(paths))


if __name__ == "__main__":
    unittest.main()
