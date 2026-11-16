import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import differentiable_isp
import exp07_emovis_diff as experiment
import exp19_full_evaluation as evaluation


class SmallVLM:
    def tokenize(self, texts):
        return torch.tensor([[len(text), sum(map(ord, text)) % 17, 7, 3] for text in texts],
                            dtype=torch.float32)

    def encode_text(self, tokens):
        return tokens

    def encode_image(self, image):
        means = image.mean(dim=(2, 3))
        return torch.cat((means, image.square().mean(dim=(1, 2, 3))[:, None]), dim=1)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.image = 0.1 + torch.rand(1, 3, 12, 12) * 0.6

    def optimize(self, word, history=None):
        model = SmallVLM()
        isp = differentiable_isp.DifferentiableEMOVIS("cpu")
        with torch.no_grad():
            neutral = isp(self.image)
            feature = evaluation.encode_image(model, neutral, 0.5, 0.5)
        guards = experiment.build_guards(model, feature)
        with patch.object(experiment, "DEVICE", "cpu"):
            return experiment.optimize_for_word(
                word, self.image, guards, model, 0.5, 0.5, num_iters=4, history=history)

    def test_tracing_preserves_legacy_results_and_return_shape(self):
        old = self.optimize("soft")
        history = []
        traced = self.optimize("soft", history)
        self.assertEqual(len(traced), 4)
        self.assertEqual(old, traced)
        self.assertEqual(len(history), 4)
        best = min(history, key=lambda row: row["total"])
        self.assertEqual(traced[0], best["params"])
        self.assertEqual(traced[1], best["semantic_similarity"])
        for index, row in enumerate(history, 1):
            self.assertEqual(row["iteration"], index)
            self.assertAlmostEqual(row["total"], sum(row[key] for key in evaluation.LOSSES[:-1]), places=6)
            restored = differentiable_isp.DifferentiableEMOVIS("cpu")
            with torch.no_grad():
                restored.raw_params.copy_(torch.tensor(row["raw_params"]))
            self.assertEqual(restored.get_params(), row["params"])

    def test_autotune_trace_has_zero_semantic_and_valid_json(self):
        history = []
        self.optimize(None, history)
        for row in history:
            self.assertEqual(row["semantic"], 0)
            self.assertIsNone(row["semantic_similarity"])
        json.dumps(history, allow_nan=False)

    def test_native_loader_preserves_dimensions_and_default_still_crops(self):
        raw = np.full((14, 22, 3), 28000, dtype=np.uint16)
        with patch.object(differentiable_isp.rawpy, "imread") as read:
            read.return_value.__enter__.return_value.postprocess.return_value = raw
            native = differentiable_isp.load_and_preprocess_raw("mock.dng", None)
            crop = differentiable_isp.load_and_preprocess_raw("mock.dng", 8)
        self.assertEqual(native.shape, (1, 3, 14, 22))
        self.assertEqual(crop.shape, (1, 3, 8, 8))
        self.assertEqual(native.dtype, torch.float32)

    def test_native_export_is_16_bit_and_correct_rgb_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.png"
            original = self.image.clone()
            evaluation.save_native(path, self.image)
            restored = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            expected = np.rint(original[0].permute(1, 2, 0).numpy() * 65535).astype(np.uint16)
            self.assertEqual(restored.dtype, np.uint16)
            self.assertEqual(restored.shape, (12, 12, 3))
            np.testing.assert_array_equal(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB), expected)
            torch.testing.assert_close(self.image, torch.round(original * 65535) / 65535, rtol=0, atol=0)

    def test_native_save_failure_is_explicit(self):
        with patch.object(evaluation.cv2, "imwrite", return_value=False):
            with self.assertRaises(OSError):
                evaluation.save_native(Path("not-written.png"), self.image)

    def test_csv_and_graph_keep_every_iteration(self):
        history = []
        self.optimize("soft", history)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "convergence").mkdir()
            evaluation.save_trace(out, "soft", history)
            with (out / "convergence" / "soft.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["iteration"]) for row in rows], [1, 2, 3, 4])
            self.assertEqual(float(rows[0]["total"]), history[0]["total"])
            self.assertTrue((out / "convergence" / "soft.png").stat().st_size > 1000)

    def test_verifier_rejects_missing_image_set(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Image directory mismatch"):
                evaluation.verify(Path(directory))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
