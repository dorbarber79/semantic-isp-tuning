import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from moran_semantic_isp import parameters_from_controls, seed_config
from plot_isp_parameter_activation import FIELDS, bar_figure, control_deltas, number, physical_control_row


class ParameterActivationTests(unittest.TestCase):
    def test_deltas_use_one_for_gain_and_contrast_neutral(self):
        row = dict(zip(FIELDS, [.06, -.2, .8, 1.25, -.3, .04]))
        settings, changes = control_deltas([row])
        np.testing.assert_allclose(settings[0], [.06, -.2, .8, 1.25, -.3, .04])
        np.testing.assert_allclose(changes[0], [.06, -.2, -.2, .25, -.3, .04])

    def test_small_values_not_silently_rounded_to_zero(self):
        self.assertIn("e-", number(1e-5))
        self.assertEqual(number(0), "0")

    def test_extract_rendered_controls_not_semantic_coordinates(self):
        parameters = parameters_from_controls(torch.tensor([.06, -.2, .8, 1.25, -.3, .04]), seed_config())
        row = physical_control_row(parameters)
        _, changes = control_deltas([row])
        np.testing.assert_allclose(changes[0], [.06, -.2, -.2, .25, -.3, .04], atol=1e-7)

    def test_paired_figure_accepts_version_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            changes = np.zeros((2, 6))
            bar_figure(path, ["warm", "cold"], changes, changes,
                       title="Direct vs cinematic", comparison_labels=("Cinematic", "Direct ISP"))
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()