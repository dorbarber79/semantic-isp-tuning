import unittest
from types import SimpleNamespace

import torch
import json

from process_fullres import (align_target_to_native, can_reuse_completed_output,
                             postprocess_dimensions_wh, CONFIG_PATH,
                             calibration_endpoints, fit_controls)


class FullResolutionProcessingTests(unittest.TestCase):
    def test_fit_controls_runs_full_budget_and_records_tint_convergence(self):
        config = json.loads(CONFIG_PATH.read_text())
        endpoints, _ = calibration_endpoints(config, torch.device("cpu"), torch.float32)
        neutral = torch.full((1, 3, 16, 16), .25)
        target = neutral.clone()
        target[:, 0] += .02
        target[:, 1] -= .028
        target[:, 2] += .02

        fit = fit_controls(neutral, target, config, endpoints, iterations=3, learning_rate=.03)

        self.assertIn("tint", fit)
        self.assertFalse(fit["convergence"]["early_stopping"])
        self.assertEqual(fit["convergence"]["total_optimizer_steps"], 9)
        for restart in fit["convergence"]["restart_diagnostics"]:
            self.assertEqual(restart["optimizer_steps"], 3)
            self.assertEqual(restart["evaluations_including_initial"], 4)
            self.assertEqual(restart["loss_trace_every_10_steps"][-1]["iteration"], 3)

    def test_postprocess_dimensions_follow_rawpy_orientation(self):
        sizes = SimpleNamespace(iwidth=3522, iheight=2348, flip=0)
        self.assertEqual(postprocess_dimensions_wh(sizes), [3522, 2348])
        sizes.flip = 5
        self.assertEqual(postprocess_dimensions_wh(sizes), [2348, 3522])
        sizes.flip = 6
        self.assertEqual(postprocess_dimensions_wh(sizes), [2348, 3522])

    def test_target_alignment_resamples_only_matching_aspect(self):
        target = torch.ones(1, 3, 10, 15)
        aligned, record = align_target_to_native(target, (20, 30))
        self.assertEqual(tuple(aligned.shape), (1, 3, 20, 30))
        self.assertEqual(record["source_dimensions_wh"], [15, 10])
        self.assertEqual(record["method"], "area resampling after aspect-ratio validation")
        with self.assertRaises(ValueError):
            align_target_to_native(target, (20, 20))

    def test_completed_output_reuse_is_source_matched_and_rooted(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "processed" / "scene").mkdir(parents=True)
            prior = {"status": "complete", "source_signature": "abc",
                     "output_directory": "processed/scene"}
            self.assertTrue(can_reuse_completed_output(root, prior, "abc"))
            self.assertFalse(can_reuse_completed_output(root, prior, "changed"))
            self.assertFalse(can_reuse_completed_output(root,
                {**prior, "status": "error"}, "abc"))
            self.assertFalse(can_reuse_completed_output(root,
                {**prior, "output_directory": "../outside"}, "abc"))
            self.assertFalse(can_reuse_completed_output(root,
                {**prior, "output_directory": "missing"}, "abc"))