import unittest

import torch

from pace_vlm.models.adaptive_pixel_compressor import (
    AdaptivePixelCompressor,
    compressed_resolution,
    smart_resize,
)


class AdaptivePixelCompressorTest(unittest.TestCase):
    def test_compressed_resolution_tracks_requested_area(self) -> None:
        height, width, ratio = compressed_resolution(1400, 980, 28, 0.10)
        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)
        self.assertAlmostEqual(ratio, 0.10, delta=0.01)

    def test_smart_resize_respects_bounds_and_alignment(self) -> None:
        height, width = smart_resize(4000, 3000, min_pixels=200_704, max_pixels=1_605_632)
        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)
        self.assertLessEqual(200_704, height * width)
        self.assertLessEqual(height * width, 1_605_632)

    def test_apc_score_is_bounded(self) -> None:
        compressor = AdaptivePixelCompressor.__new__(AdaptivePixelCompressor)
        compressor.global_weight = 0.6
        compressor.detail_fraction = 0.1
        compressor.detail_scale = 1.5
        compressor.minimum_retention = 0.05
        features = torch.randn(128, 64, generator=torch.Generator().manual_seed(0))
        retention, global_density, local_detail = compressor.score(features)
        self.assertLessEqual(0.05, retention)
        self.assertLessEqual(retention, 1.0)
        self.assertLessEqual(0.0, global_density)
        self.assertLessEqual(global_density, 1.0)
        self.assertLessEqual(0.0, local_detail)
        self.assertLessEqual(local_detail, 1.0)


if __name__ == "__main__":
    unittest.main()
