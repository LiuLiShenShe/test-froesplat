import tempfile
import unittest
from pathlib import Path

from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json
from ForeSplat_AutoGaussianCleaner.summarize_cross_plant_comparison import verified_visual_review


class ReviewEvidenceIntegrity(unittest.TestCase):
    def fixture(self, root):
        inspection = root/'quality_inspection'
        inspection.mkdir()
        worst = inspection/'WORST_VIEW_CONTACT_SHEET.png'
        free = inspection/'FREE_VIEW_CONTACT_SHEET.png'
        source = inspection/'native_free_source.png'
        for path in [worst, free, source]:
            path.write_bytes(path.name.encode())
        result = dict(native_sha256='cleaned-model', input_sha256='original-model')
        write_json(inspection/'DIRECT_VISUAL_REVIEW.json', dict(
            **result, images=[str(worst), str(free)],
            image_sha256={str(path): sha(path) for path in [worst, free]},
            scope='Observed worst views and automatic free views', review_method='Actual direct inspection',
            conspicuous_leaf_loss_in_reviewed_views=False, residual_black_cloud_visible=True))
        write_json(inspection/'QUALITY_INSPECTION.json', dict(**result, contact_sheet=str(worst)))
        write_json(inspection/'FREE_VIEW_CONTACT_SHEET.json', dict(
            **result, contact_sheet=str(free), contact_sheet_sha256=sha(free),
            images=[dict(image=str(source), image_sha256=sha(source))]))
        return result, worst, source

    def test_regenerated_sheet_invalidates_preceding_visual_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, worst, source = self.fixture(root)
            self.assertTrue(verified_visual_review(root, result)['residual_black_cloud_visible'])
            worst.write_bytes(b'Regenerated image of the same model')
            with self.assertRaisesRegex(ValueError, 'reviewed image changed'):
                verified_visual_review(root, result)

    def test_replaced_native_png_invalidates_same_contact_sheet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, worst, source = self.fixture(root)
            source.write_bytes(b'PNG from another native render')
            with self.assertRaisesRegex(ValueError, 'source PNG changed'):
                verified_visual_review(root, result)


if __name__ == '__main__':
    unittest.main()
