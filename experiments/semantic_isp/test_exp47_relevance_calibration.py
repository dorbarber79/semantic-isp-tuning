import unittest

import numpy as np
import torch

from exp47_relevance_calibration import constrained_coefficients, constraint_signs, select_threshold
from semantic_isp import compose_parameters, seed_config, neutral_raw_controls


class RelevanceCalibrationTests(unittest.TestCase):
    def test_control_connections_and_signs_are_enforced(self):
        random = np.random.default_rng(42)
        features = random.uniform(0, 1, (40, 10))
        targets = random.normal(0, 1, (40, 6))
        fitted = constrained_coefficients(features, targets, 1.)
        signs = constraint_signs()
        self.assertTrue((fitted[signs == 0] == 0).all())
        self.assertTrue((fitted[signs != 0] * signs[signs != 0] >= 0).all())
        config = seed_config()
        config["learned_calibration"] = {"raw_neutral": neutral_raw_controls().tolist(),
                                         "pole_raw_deltas": fitted.tolist()}
        for axis in range(5):
            for sign in (-1, 1):
                coordinates = torch.zeros(5)
                coordinates[axis] = .4 * sign
                parameters = compose_parameters(coordinates, config)
                if axis == 0:
                    self.assertGreaterEqual((parameters["wb"][0].item() - 1) * sign, -1e-6)
                    self.assertAlmostEqual(parameters["saturation"].item(), 0, places=6)
                    self.assertAlmostEqual(parameters["sharpness"].item(), 0, places=6)
                elif axis == 3:
                    self.assertGreaterEqual(parameters["saturation"].item() * sign, -1e-6)
                    self.assertAlmostEqual(parameters["tone_gain"].item(), 1, places=6)
                elif axis == 4:
                    self.assertGreaterEqual(parameters["palette_coordinate"].item() * sign, -1e-6)
                    torch.testing.assert_close(parameters["ccm"], torch.eye(3))

    def test_threshold_uses_validation_false_positive_budget(self):
        probabilities = np.array([.95, .8, .7, .2, .1])
        labels = np.array([1, 1, 0, 0, 0])
        threshold, recall, false_positive = select_threshold(probabilities, labels)
        self.assertGreaterEqual(threshold, .7)
        self.assertEqual(recall, 1.)
        self.assertEqual(false_positive, 0.)


if __name__ == "__main__":
    unittest.main()