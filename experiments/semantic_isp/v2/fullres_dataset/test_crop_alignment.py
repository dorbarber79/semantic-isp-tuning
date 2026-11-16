import unittest

import cv2
import numpy as np

from crop_alignment import (CropAlignmentError, align_expert_crop,
                            oriented_default_crop_rect)


def textured_scene(height=760, width=1240):
    rng = np.random.default_rng(42)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:] = np.linspace(.08, .75, height, dtype=np.float32)[:, None, None]
    image += rng.normal(0, .025, image.shape).astype(np.float32)
    for index in range(90):
        x = int(rng.integers(10, width - 10))
        y = int(rng.integers(10, height - 10))
        color = rng.uniform(.05, .95, 3).astype(np.float32)
        radius = int(rng.integers(2, 13))
        cv2.circle(image, (x, y), radius, tuple(float(v) for v in color), -1)
    return np.clip(image, 0, 1)


class CropAlignmentTests(unittest.TestCase):
    def test_default_crop_rect_uses_pre_rotation_sensor_dimensions(self):
        self.assertEqual(oriented_default_crop_rect(3522, 2348, 0, 10, 7, 3504, 2336),
                         (10, 7, 3504, 2336))
        self.assertEqual(oriented_default_crop_rect(3522, 2348, 5, 10, 7, 3504, 2336),
                         (7, 8, 2336, 3504))

    def test_refines_crop_offsets_without_scaling(self):
        baseline = textured_scene()
        target = baseline[53:653, 87:987].copy()
        raw_crop_prior = (60, 40, 900, 600)
        aligned_baseline, aligned_target, record = align_expert_crop(
            baseline, target, raw_crop_prior, rawpy_flip=0)
        self.assertEqual(aligned_baseline.shape, target.shape)
        self.assertTrue(np.array_equal(aligned_target, target))
        self.assertEqual(record["refined_crop_xywh"], [87, 53, 900, 600])
        self.assertFalse(record["resized_or_warped"])

    def test_recovers_rotated_tiff_orientation(self):
        baseline = textured_scene()
        crop = baseline[53:653, 87:987].copy()
        target = np.rot90(crop, 1).copy()
        aligned_baseline, aligned_target, record = align_expert_crop(
            baseline, target, (80, 45, 900, 600), rawpy_flip=0)
        self.assertEqual(aligned_baseline.shape, crop.shape)
        self.assertTrue(np.array_equal(aligned_target, crop))
        self.assertEqual(record["target_rotation_ccw"], 3)
        self.assertEqual(record["refined_crop_xywh"], [87, 53, 900, 600])

    def test_rejects_rescaled_target_instead_of_resizing(self):
        baseline = textured_scene()
        crop = baseline[53:653, 87:987].copy()
        target = cv2.resize(crop, (720, 480), interpolation=cv2.INTER_AREA)
        with self.assertRaises(CropAlignmentError):
            align_expert_crop(baseline, target, (80, 45, 900, 600), rawpy_flip=0)


if __name__ == "__main__":
    unittest.main()
