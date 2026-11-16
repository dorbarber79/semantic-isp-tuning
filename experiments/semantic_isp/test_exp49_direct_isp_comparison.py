import unittest
import hashlib
from unittest.mock import patch

import numpy as np
import torch

from exp49_direct_isp_comparison import direct_config, direct_quantile_calibration, directional_mean_endpoint, fit_signed, fit_soft_gate, select_soft_band
from exp42_phrase_pole_calibration import descriptor_rows
from exp38_learn_poles import optimize_controls
from moran_semantic_isp import CONTROL_BOUNDS, CONTROL_NAMES, CONTROL_NEUTRAL, compose_parameters, neutral_raw_controls


class DirectControlComparisonTests(unittest.TestCase):
    def test_directional_mean_moderates_poles_without_reversing_sign(self):
        self.assertAlmostEqual(directional_mean_endpoint([-.4, -.3, -.2], 0., -1), -.3)
        self.assertAlmostEqual(directional_mean_endpoint([.2, .4, .6], 0., 1), .4)
        self.assertAlmostEqual(directional_mean_endpoint([-.2, .2], 0., 1), .1)
        self.assertAlmostEqual(directional_mean_endpoint([-.2, .2], 0., -1), -.1)

    def test_autotune_trains_one_scalar_and_keeps_other_controls_neutral(self):
        config = direct_config()
        config["control_bounds"] = [list(bounds) for bounds in CONTROL_BOUNDS]
        config["control_bounds"][3] = [.2, 3.]
        image = torch.rand((1, 3, 24, 24), generator=torch.Generator().manual_seed(7)) * .7 + .05
        target = torch.nn.functional.normalize(torch.tensor([[.8, .2, .4, .1, .6, .3]]), dim=-1)

        def encode_mock(model, rendered, config=None):
            features = torch.cat((rendered.mean(dim=(-1, -2)), rendered.std(dim=(-1, -2))), dim=-1)
            return torch.nn.functional.normalize(features, dim=-1)

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for index, name in enumerate(CONTROL_NAMES):
                with patch("exp38_learn_poles.encode_image", side_effect=encode_mock), patch(
                        "exp38_learn_poles.torch.optim.Adam", wraps=torch.optim.Adam) as optimizer_factory:
                    result, trace = optimize_controls(image, target, None, config, 3, .03, active_control=name)
                parameters = optimizer_factory.call_args.args[0]
                self.assertEqual(len(parameters), 1)
                self.assertEqual(parameters[0].numel(), 1)
                self.assertEqual(result["active_controls"], [name])
                for row in trace:
                    self.assertEqual(row["active_control"], name)
                    for other_index, other_name in enumerate(CONTROL_NAMES):
                        if other_index != index:
                            self.assertEqual(row[other_name], CONTROL_NEUTRAL[other_index])
                            self.assertEqual(result["controls"][other_index], CONTROL_NEUTRAL[other_index])
        finally:
            torch.set_num_threads(threads)

    def test_palette_strength_prior_selects_requested_pole_direction(self):
        config = direct_config()
        image = torch.full((1, 3, 24, 24), .35)
        target = torch.ones((1, 4))

        def encode_mock(model, rendered, config=None):
            return torch.ones((rendered.shape[0], 4), device=rendered.device) / 2

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for sign in (-1, 1):
                with patch("exp38_learn_poles.encode_image", side_effect=encode_mock):
                    result, trace = optimize_controls(image, target, None, config, 100, .03,
                        active_control="palette", strength_target_sign=sign, strength_weight=.05)
                self.assertGreater(sign * result["controls"][5], .3)
                self.assertEqual(result["active_controls"], ["palette"])
                self.assertTrue(all(row["active_control"] == "palette" for row in trace))
                self.assertTrue(all(row["strength_penalty"] >= 0 for row in trace))
                for index, name in enumerate(CONTROL_NAMES[:-1]):
                    self.assertEqual(result["controls"][index], CONTROL_NEUTRAL[index])
        finally:
            torch.set_num_threads(threads)

    def test_palette_strength_prior_targets_configured_coordinate(self):
        config = direct_config()
        config["palette_tuning_objective"] = {"directional_strength_target_coordinate": .2}
        image = torch.full((1, 3, 24, 24), .35)
        target = torch.ones((1, 4))

        def encode_mock(model, rendered, config=None):
            return torch.ones((rendered.shape[0], 4), device=rendered.device) / 2

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for sign in (-1, 1):
                with patch("exp38_learn_poles.encode_image", side_effect=encode_mock):
                    result, _ = optimize_controls(image, target, None, config, 100, .03,
                        active_control="palette", strength_target_sign=sign, strength_weight=.25)
                self.assertAlmostEqual(sign * result["controls"][5], .2, delta=.02)
        finally:
            torch.set_num_threads(threads)

    def test_table_covers_all_six_physical_controls(self):
        config = direct_config()
        self.assertEqual(len(config["axes"]), 6)
        self.assertEqual(config["color_protection"]["mode"], "gray_white_and_skin")
        self.assertEqual(len(descriptor_rows(config)), 192)
        self.assertEqual(config["axes"][3], "contrast")
        self.assertEqual(config["axes"][4], "sharpness")

    def test_fit_does_not_mix_controls_or_reverse_signs(self):
        signs = np.concatenate((np.eye(6), -np.eye(6)), axis=0)
        random = np.random.default_rng(42)
        design = random.uniform(0, 1, (48, 12))
        target = random.normal(0, 1, (48, 6))
        coefficients = fit_signed(design, target, signs, 1.)
        self.assertTrue((coefficients[signs == 0] == 0).all())
        self.assertTrue((coefficients[signs != 0] * signs[signs != 0] >= 0).all())
        config = direct_config()
        config["learned_calibration"] = {"raw_neutral": neutral_raw_controls().tolist(),
                                         "pole_raw_deltas": coefficients.tolist()}
        parameters = compose_parameters(torch.zeros(6), config)
        torch.testing.assert_close(parameters["wb"], torch.ones(3))
        self.assertAlmostEqual(parameters["tone_gain"].item(), 1., places=6)

    def test_descriptors_follow_active_pole_schema(self):
        config = direct_config()
        self.assertEqual(config["descriptor_table"]["descriptor_revision"], "moran_style_v14_extreme_brightness_4x4")
        rows = descriptor_rows(config)
        self.assertEqual(len({row["text"] for row in rows}), 192)
        for row in rows:
            expected = "perceptual" if row["group"] == "D2" else (
                "cultural_associative" if row["group"] == "D4" else "technical")
            self.assertEqual(row["type"], expected)
        for axis in config["axes"]:
            for side in ("negative", "positive"):
                for group in ("D1", "D2", "D3", "D4"):
                    selected = [row for row in rows if row["axis"] == axis and row["sign"] == side and row["group"] == group]
                    self.assertEqual(len(selected), 4)
        self.assertEqual(config["poles"]["warmth"]["positive"]["texts"][0], "Extreme warm white balance with a strong amber cast")
        self.assertEqual(config["poles"]["sharpness"]["negative"]["texts"][0], "Heavy Gaussian blur across the entire photograph")
        for side in ("negative", "positive"):
            self.assertEqual(len(config["poles"]["brightness"][side]["texts"]), 16)
            marker = ("black", "dark", "underexpos", "midnight", "unlit", "luminance") if side == "negative" else (
                "bright", "white", "overexpos", "luminance", "sunlit", "high-key", "radiant")
            self.assertTrue(all(any(token in text.lower() for token in marker)
                                for text in config["poles"]["brightness"][side]["texts"]))
            brightness_text = " ".join(config["poles"]["brightness"][side]["texts"]).lower()
            for intensity in ("very", "much", "extreme"):
                self.assertIn(intensity, brightness_text)
        negative_saturation = [row for row in rows if row["axis"] == "saturation" and row["sign"] == "negative"]
        self.assertEqual(len(negative_saturation), 16)
        self.assertTrue(all("monochrome" in row["text"].lower() for row in negative_saturation))
        positive_contrast = [row for row in rows if row["axis"] == "contrast" and row["sign"] == "positive"]
        high_contrast_markers = ("contrast", "extreme", "maximum tonal separation",
                     "increased contrast", "crushed blacks", "brilliant highlights",
                                 "deep blacks", "bright white highlights", "near-black shadows",
                                 "black shadow values")
        self.assertTrue(all(any(marker in row["text"].lower() for marker in high_contrast_markers)
                    for row in positive_contrast))

    def test_soft_band_can_fully_accept_moderate_scores(self):
        probabilities = np.array([.5, .6, .7, .01, .02, .03])
        labels = np.array([1, 1, 1, 0, 0, 0])
        lower, upper, positive, negative = select_soft_band(probabilities, labels)
        self.assertLess(upper, 1.)
        self.assertGreater(positive, .9)
        self.assertLess(negative, .05)
        self.assertLess(lower, upper)

    def test_extreme_poles_use_opposite_physical_tails_without_shrinkage(self):
        config = direct_config()
        config["learned_calibration"] = {"raw_neutral": neutral_raw_controls().tolist()}
        records = []
        for column, axis in enumerate(config["axes"]):
            for side in ("negative", "positive"):
                for setting in (0., .1, .2, .3):
                    controls = list(CONTROL_NEUTRAL)
                    controls[column] += setting
                    records.append({"axis": axis, "sign": side,
                        "controls": controls})
        calibrated, statistics = direct_quantile_calibration(config, records)
        positive = compose_parameters(torch.tensor([.4, 0., 0., 0., 0., 0.]), calibrated)
        self.assertAlmostEqual(float(positive["wb"][0] - 1), .27, places=6)
        negative = next(row for row in statistics if row["axis"] == "warmth" and row["sign"] == "negative")
        self.assertFalse(negative["supported_direction"])
        self.assertEqual(negative["selected_setting"], 0)
        for record in records:
            if record["sign"] == "negative":
                record["controls"][0] *= -1
        calibrated, _ = direct_quantile_calibration(config, records)
        negative = compose_parameters(torch.tensor([-.4, 0., 0., 0., 0., 0.]), calibrated)
        self.assertAlmostEqual(float(negative["wb"][0] - 1), -.27, places=6)
        self.assertAlmostEqual(float(negative["saturation"]), 0., places=6)
        records[0]["controls"][3] = 1.1
        with self.assertRaisesRegex(ValueError, "Inactive controls moved"):
            direct_quantile_calibration(config, records)

    def test_contrast_quantiles_respect_widened_per_configuration_bounds(self):
        config = direct_config()
        bounds = [list(pair) for pair in __import__("moran_semantic_isp").CONTROL_BOUNDS]
        bounds[3] = [.2, 3.]
        config["control_bounds"] = bounds
        config["learned_calibration"] = {"raw_neutral": neutral_raw_controls(bounds=bounds).tolist()}
        records = []
        for column, axis in enumerate(config["axes"]):
            for side in ("negative", "positive"):
                settings = (.2, .3, .4, .5) if axis == "contrast" and side == "negative" else (
                    (2.4, 2.6, 2.8, 3.) if axis == "contrast" else (CONTROL_NEUTRAL[column],) * 4)
                for setting in settings:
                    values = list(CONTROL_NEUTRAL)
                    values[column] = setting
                    records.append({"axis": axis, "sign": side, "controls": values,
                                    "active_controls": [CONTROL_NAMES[column]]})
        calibrated, statistics = direct_quantile_calibration(config, records)
        selected = {row["sign"]: row["selected_setting"] for row in statistics if row["axis"] == "contrast"}
        self.assertLess(selected["negative"], .4)
        self.assertGreater(selected["positive"], 2.)
        for side, coordinate in (("negative", -.4), ("positive", .4)):
            coords = torch.zeros(6);coords[3] = coordinate
            params = compose_parameters(coords, calibrated)
            self.assertAlmostEqual(float(params["contrast"]), selected[side], places=5)
            self.assertAlmostEqual(float(params["tone_gain"]), 1., places=6)

    def test_descriptor_tail_selects_endpoint_and_centroid_from_same_phrases(self):
        config = direct_config()
        bounds = [list(pair) for pair in CONTROL_BOUNDS]
        config["control_bounds"] = bounds
        config["learned_calibration"] = {"raw_neutral": neutral_raw_controls(bounds=bounds).tolist()}
        records = []
        for column, axis in enumerate(config["axes"]):
            for side in ("negative", "positive"):
                entries = config["descriptor_table"]["axes"][axis][side]
                for descriptor_index, entry in enumerate(entries):
                    magnitude = .01 * (descriptor_index + 1) if axis == "brightness" else 0.
                    direction = -1 if side == "negative" else 1
                    for image_id, image_offset in ((1, -.001), (2, .001)):
                        controls = list(CONTROL_NEUTRAL)
                        controls[column] += direction * max(0., magnitude + image_offset)
                        records.append({"axis": axis, "sign": side, "id": entry["id"],
                            "text": entry["text"], "image": image_id, "controls": controls,
                            "active_controls": [CONTROL_NAMES[column]]})
        calibrated, statistics = direct_quantile_calibration(config, records)
        learned = calibrated["learned_calibration"]
        self.assertEqual(learned["pole_statistic"]["method"], "descriptor_median_directional_quantile")
        self.assertEqual(learned["pole_centroid_descriptor_ids"]["brightness"]["positive"],
                         ["D4_4", "D4_3"])
        self.assertEqual(learned["pole_centroid_texts"]["brightness"]["positive"],
                         [entry["text"] for entry in config["descriptor_table"]["axes"]["brightness"]["positive"][-2:][::-1]])
        positive = next(row for row in statistics if row["axis"] == "brightness" and row["sign"] == "positive")
        self.assertAlmostEqual(positive["selected_setting"], 1.145, places=6)
        self.assertEqual(positive["descriptor_count"], 16)
        self.assertEqual(positive["sample_count"], 32)

    def test_all_d4_sentences_stay_out_of_gate_training(self):
        config = direct_config()

        def encode(texts):
            return torch.stack([torch.randn(32, generator=torch.Generator().manual_seed(
                int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))) for text in texts])

        _, split = fit_soft_gate(config, encode)
        d4_texts = {row["text"] for row in descriptor_rows(config) if row["group"] == "D4"}
        self.assertEqual(len(d4_texts), 48)
        self.assertFalse(d4_texts.intersection(split["classifier_training"]))
        self.assertTrue(d4_texts.issubset(split["validation_texts"]))


if __name__ == "__main__":
    unittest.main()