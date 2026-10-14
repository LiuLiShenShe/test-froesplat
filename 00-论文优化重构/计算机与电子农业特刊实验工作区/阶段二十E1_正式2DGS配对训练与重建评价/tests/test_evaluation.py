import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import MATRIX, import_paths
from evaluate import proxy_metrics, resize_reference


class EvaluationTests(unittest.TestCase):
    def test_logging_compatibility_rgb_matches_argb(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(1, 1), dpi=50)
        axis.plot([0, 1], [1, 0])
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        argb = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8).reshape(50, 50, 4)
        self.assertEqual(rgba.tobytes(), argb[..., 1:].tobytes())
        plt.close(figure)

    def test_eight_runs_and_no_duplicate(self):
        self.assertEqual(len(MATRIX), 8)
        self.assertNotIn(("HongZhang", "B_R0"), MATRIX)

    def test_channelwise_production_resize(self):
        import numpy as np
        import torch
        from PIL import Image
        array = np.arange(15 * 17 * 3, dtype=np.uint8).reshape(15, 17, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.fromarray(array).save(path)
            actual = resize_reference(path, 8, 7)
            expected = torch.stack([torch.from_numpy(np.array(Image.fromarray(array[..., channel]).resize((8, 7))))
                                    for channel in range(3)]).float() / 255
            self.assertTrue(torch.equal(actual, expected))

    def test_psnr_known_error_and_ssim_identity(self):
        import torch
        import_paths()
        from utils.image_utils import psnr
        from utils.loss_utils import ssim
        reference = torch.ones(3, 32, 32)
        self.assertAlmostEqual(psnr(reference * 0.5, reference).mean().item(), 6.020599913, places=5)
        self.assertAlmostEqual(ssim(reference, reference).item(), 1.0, places=6)

    def test_psnr_global_rgb_mse(self):
        import torch
        import math
        import_paths()
        from utils.image_utils import psnr
        reference = torch.ones(1, 3, 32, 32)
        prediction = torch.zeros_like(reference)
        prediction[:, 0] = 0.5
        expected_mse = (0.25 + 1 + 1) / 3
        self.assertAlmostEqual(psnr(prediction, reference).item(), -10 * math.log10(expected_mse), places=6)

    def test_proxy_is_not_geometry_truth(self):
        import torch
        mask = torch.zeros(32, 32)
        mask[8:24, 8:24] = 1
        rgb = torch.zeros(3, 32, 32)
        values = proxy_metrics(rgb, torch.ones_like(rgb), mask, mask)
        self.assertEqual(values["R0_proxy_silhouette_iou"], 1)
        self.assertEqual(values["R0_proxy_missing_alpha_fraction"], 0)
        self.assertEqual(values["R0_proxy_outside_components_ge16px"], 0)


if __name__ == "__main__":
    unittest.main()
