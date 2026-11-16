import unittest

import torch
import torch.nn.functional as F

from differentiable_isp import _luminance
from semantic_isp import (AXES, CONTROL_BOUNDS,
                                SemanticProjection, compose_parameters, detail_gaussian,
                                linear_to_srgb, render_linear, seed_config)
from semantic_isp import (
    bounded_controls, neutral_raw_controls, parameters_from_controls, CONTROL_NAMES,
    CONTROL_NEUTRAL,
)
from axis_isp import (apply_nearest_swatch_palette, apply_split_palette_projection,
                      chromaticity_proximity, color_protection_factor)


class SemanticISPTests(unittest.TestCase):
    def test_chromaticity_protection_is_luma_independent_and_operation_scoped(self):
        skin = torch.tensor([.76, .57, .47])
        skin_linear = torch.where(skin <= .04045, skin / 12.92,
                                  ((skin + .055) / 1.055).pow(2.4))
        image = torch.stack((torch.full((3,), .2), skin_linear * .4,
                             torch.tensor([.1, .6, .2])), dim=1).view(1, 3, 1, 3)
        skin_reference = chromaticity_proximity(image[:, :, :, 1:2], skin, .1,
                                                reference_is_srgb=True)
        brighter_skin = chromaticity_proximity(image[:, :, :, 1:2] * 2, skin, .1,
                                               reference_is_srgb=True)
        torch.testing.assert_close(skin_reference, brighter_skin, atol=1e-6, rtol=0)

        protection = {"mode": "gray_white_and_skin", "gray_sigma": .07,
                      "skin_sigma": .10, "skin_reference_srgb": skin.tolist(), "strength": 1.}
        factors = color_protection_factor(image, protection, "warmth")
        self.assertLess(float(factors[0, 0, 0, 0]), 1e-5)
        self.assertLess(float(factors[0, 0, 0, 1]), 1e-5)
        self.assertGreater(float(factors[0, 0, 0, 2]), .9)
        positive_saturation = color_protection_factor(image, protection, "saturation", .5)
        negative_saturation = color_protection_factor(image, protection, "saturation", -.5)
        self.assertLess(float(positive_saturation[0, 0, 0, 1]), 1e-5)
        self.assertTrue(torch.equal(negative_saturation, torch.ones_like(negative_saturation)))
        self.assertTrue(torch.equal(color_protection_factor(image, {"mode": "none"}, "palette"),
                                    torch.ones_like(factors)))
        palette_only = dict(protection, operations=["palette"])
        self.assertTrue(torch.equal(color_protection_factor(image, palette_only, "warmth"),
                        torch.ones_like(factors)))
        self.assertTrue(torch.equal(color_protection_factor(image, palette_only, "saturation", .5),
                        torch.ones_like(factors)))
        self.assertLess(float(color_protection_factor(image, palette_only, "palette")[0, 0, 0, 1]),
                1e-5)

    def test_rgb_blur_removes_color_edges_without_changing_positive_sharpening(self):
        old = seed_config()
        old["detail_filter"] = {"sigma_at_reference": 8., "reference_short_side": 128., "truncate": 3.}
        old["positive_sharpening"] = {"gain": 2.4, "detail_limit": .06}
        new = dict(old, detail_filter=dict(old["detail_filter"], blur_color_mode="rgb"))
        image = torch.full((1, 3, 128, 128), .15)
        image[:, 0, :, :64] = .6
        image[:, 1, :, 64:] = .15 + .299 * (.6 - .15) / .587
        identity_controls = torch.tensor([0., 0., 1., 1., 0., 0.])
        controls = identity_controls.clone()
        controls[4] = -.8
        old_blur = render_linear(image, parameters_from_controls(controls, old))
        current = render_linear(image, parameters_from_controls(controls, new))
        torch.testing.assert_close(old_blur, image, atol=1e-6, rtol=0)
        expected = .2 * image + .8 * detail_gaussian(image, parameters_from_controls(controls, new))
        torch.testing.assert_close(current, expected, atol=1e-6, rtol=0)
        self.assertGreater(float((current - image)[:, :, :, 56:72].abs().mean()), .03)
        separate = torch.cat([detail_gaussian(image[:, channel:channel + 1],
                    parameters_from_controls(controls, new)) for channel in range(3)], dim=1)
        torch.testing.assert_close(detail_gaussian(image, parameters_from_controls(controls, new)), separate)
        for amount in (0., .7237):
            controls = identity_controls.clone()
            controls[4] = amount
            previous = render_linear(image, parameters_from_controls(controls, old))
            current = render_linear(image, parameters_from_controls(controls, new))
            self.assertTrue(torch.equal(current, previous))
        controls = torch.tensor([0., 0., 1., 1., -.4, 0.], requires_grad=True)
        render_linear(image, parameters_from_controls(controls, new)).square().mean().backward()
        self.assertTrue(torch.isfinite(controls.grad).all())
        self.assertGreater(abs(float(controls.grad[4])), 1e-8)

    def test_stronger_positive_sharpening_does_not_change_blur_or_neutral(self):
        old = seed_config()
        old["detail_filter"] = {"sigma_at_reference": 8., "reference_short_side": 128., "truncate": 3.}
        new = dict(old, positive_sharpening={"gain": 2.4, "detail_limit": .06})
        image = torch.full((1, 3, 128, 128), .25)
        image[:, :, :, 64:] = .75
        identity_controls = torch.tensor([0., 0., 1., 1., 0., 0.])
        for amount in (-.8276, 0.):
            controls = identity_controls.clone()
            controls[4] = amount
            previous = render_linear(image, parameters_from_controls(controls, old))
            current = render_linear(image, parameters_from_controls(controls, new))
            self.assertTrue(torch.equal(current, previous))
        controls = identity_controls.clone()
        controls[4] = .8
        previous = render_linear(image, parameters_from_controls(controls, old))
        current = render_linear(image, parameters_from_controls(controls, new))
        self.assertGreater(float((current - image).abs().mean()), 3 * float((previous - image).abs().mean()))
        self.assertTrue(torch.isfinite(current).all())

    def test_large_detail_kernel_has_full_support_and_scales_with_resolution(self):
        config = seed_config()
        config["detail_filter"] = {"sigma_at_reference": 8., "reference_short_side": 128., "truncate": 3.}
        parameters = parameters_from_controls(bounded_controls(neutral_raw_controls()), config)
        impulse = torch.zeros(1, 1, 128, 128)
        impulse[:, :, 64, 64] = 1
        blurred = detail_gaussian(impulse, parameters)
        self.assertEqual(int((blurred[0, 0, 64] > 0).sum()), 49)
        self.assertAlmostEqual(float(blurred.sum()), 1., places=6)
        coordinates = torch.arange(128) - 64
        sigma = (blurred.sum(dim=-2)[0, 0] * coordinates.square()).sum().sqrt()
        self.assertGreater(float(sigma), 7.8)
        step = torch.full((1, 1, 64, 64), .2)
        step[:, :, :, 32:] = .6
        low = detail_gaussian(step, parameters)
        high = detail_gaussian(step.repeat_interleave(2, -2).repeat_interleave(2, -1), parameters)
        torch.testing.assert_close(low, F.avg_pool2d(high, 2), atol=.001, rtol=0)
        high_legacy = detail_gaussian(impulse, {"radius": torch.tensor(2.)})
        self.assertEqual(int((high_legacy[0, 0, 64] > 0).sum()), 7)

    def test_large_detail_operator_keeps_neutral_and_gradients(self):
        config = seed_config()
        config["detail_filter"] = {"sigma_at_reference": 8., "reference_short_side": 512., "truncate": 3.}
        identity_controls = torch.tensor([0., 0., 1., 1., 0., 0.])
        image = torch.rand(1, 3, 32, 32) * .6 + .1
        output = render_linear(image, parameters_from_controls(identity_controls, config))
        torch.testing.assert_close(output, image, atol=1e-6, rtol=0)
        raw = neutral_raw_controls().requires_grad_()
        output2 = render_linear(image, parameters_from_controls(bounded_controls(raw), config))
        output2.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(abs(float(raw.grad[4])), 1e-8)

    def test_sharpening_radius_is_smaller_than_preserved_blur_radius(self):
        config = seed_config()
        config["detail_filter"] = {"sigma_at_reference": 8., "reference_short_side": 128.,
                       "truncate": 3., "sharpen_sigma_at_reference": 3.}
        controls = bounded_controls(neutral_raw_controls())
        parameters = parameters_from_controls(controls, config)
        self.assertEqual(float(parameters["radius"]), 8.)
        self.assertEqual(float(parameters["sharpen_radius"]), 3.)
        x = torch.arange(128, dtype=torch.float32)
        low = (.3 + .1 * torch.sin(2 * torch.pi * x / 16)).view(1, 1, 1, 128).expand(1, 3, 128, 128)
        high = (.3 + .1 * torch.sin(2 * torch.pi * x / 8)).view(1, 1, 1, 128).expand(1, 3, 128, 128)
        with_small = render_linear(low, parameters_from_controls(controls + torch.tensor([0., 0., 0., 0., .7, 0.]), config))
        controls_large = controls.clone();controls_large[4] = .7
        config_large = dict(config, detail_filter=dict(config["detail_filter"], sharpen_sigma_at_reference=8.))
        with_large = render_linear(low, parameters_from_controls(controls_large, config_large))
        self.assertLess(float((with_small - low).abs().mean()), float((with_large - low).abs().mean()))
        high_small = render_linear(high, parameters_from_controls(controls + torch.tensor([0., 0., 0., 0., .7, 0.]), config))
        high_large = render_linear(high, parameters_from_controls(controls_large, config_large))
        small_frequency_ratio = float((high_small - high).abs().mean() / (with_small - low).abs().mean())
        large_frequency_ratio = float((high_large - high).abs().mean() / (with_large - low).abs().mean())
        self.assertGreater(small_frequency_ratio, large_frequency_ratio)

    def test_per_configuration_contrast_bounds_preserve_neutral_and_endpoint(self):
        from math import atanh
        config = seed_config()
        config["axes"] = ["warmth", "saturation", "brightness", "contrast", "sharpness", "palette"]
        config["control_bounds"] = [list(bounds) for bounds in __import__("semantic_isp").CONTROL_BOUNDS]
        config["control_bounds"][3] = [.2, 3.]
        neutral = neutral_raw_controls(bounds=config["control_bounds"])
        config["learned_calibration"] = {"raw_neutral": neutral.tolist(),
            "pole_raw_deltas": torch.zeros(12, 6).tolist(), "control_bounds": config["control_bounds"]}
        torch.testing.assert_close(bounded_controls(neutral, config["control_bounds"]),
                                   torch.tensor(list(CONTROL_NEUTRAL)))
        high_setting = 2.8
        raw_value = atanh((high_setting - 1.6) / 1.4)
        config["learned_calibration"]["pole_raw_deltas"][3][3] = raw_value - float(neutral[3])
        params = compose_parameters(torch.tensor([0., 0., 0., .4, 0., 0.]), config)
        self.assertAlmostEqual(float(params["contrast"]), high_setting, places=5)

    def test_stronger_asymmetric_palette_projection_increases_both_poles(self):
        import copy
        old = seed_config()
        strong = copy.deepcopy(old)
        strong["palette_projection"].update(max_attraction_positive=1.5,
            shift_fraction_positive=1., max_attraction_negative=1.2,
            shift_fraction_negative=.65)
        ramp = torch.linspace(.05, .9, 64)
        red = ramp.view(1, 1, 1, 64).expand(1, 1, 64, 64)
        green = ramp.flip(0).view(1, 1, 64, 1).expand(1, 1, 64, 64)
        blue = (.1 + .6 * (torch.arange(64) % 2).float()).view(1, 1, 1, 64).expand(1, 1, 64, 64)
        image = torch.cat((red, green, blue), dim=1)
        outputs = {}
        for label, amount, config in (("old_negative", -.4, old), ("old_positive", .4, old),
                                      ("new_negative", -.4, strong), ("new_positive", .4, strong)):
            controls = torch.tensor([0., 0., 1., 1., 0., amount])
            outputs[label] = render_linear(image, parameters_from_controls(controls, config))
            self.assertTrue(torch.isfinite(outputs[label]).all())
        self.assertGreater(float((outputs["new_positive"] - image).abs().mean()),
                           4 * float((outputs["old_positive"] - image).abs().mean()))
        self.assertGreater(float((outputs["new_negative"] - image).abs().mean()),
                           3 * float((outputs["old_negative"] - image).abs().mean()))
        neutral = torch.tensor([0., 0., 1., 1., 0., 0.])
        torch.testing.assert_close(render_linear(image, parameters_from_controls(neutral, strong)),
                                   image, atol=1e-6, rtol=0)

    def test_palette_split_tone_targets_named_shadow_and_highlight_swatches(self):
        config = seed_config()
        rgb_to_yiq = torch.tensor(config["palette_projection"]["rgb_to_yiq"])
        yiq_to_rgb = torch.tensor(config["palette_projection"]["yiq_to_rgb"])
        swatches = torch.tensor([config["palette_projection"]["swatches"]["retro"],
                                 config["palette_projection"]["swatches"]["blockbuster"]])
        luminance = torch.linspace(.2, .8, 64).view(1, 1, 1, 64).expand(1, 1, 32, 64)
        image = luminance.expand(1, 3, 32, 64).clone()
        params = parameters_from_controls(torch.tensor([0., 0., 1., 1., 0., 0.]), config)
        neutral = apply_split_palette_projection(image, image.new_tensor(0.), rgb_to_yiq,
            yiq_to_rgb, swatches, image.new_tensor(.4), image.new_tensor(.65), image.new_tensor(.5))
        self.assertTrue(torch.equal(neutral, image))
        positive = apply_split_palette_projection(image, image.new_tensor(.4), rgb_to_yiq,
            yiq_to_rgb, swatches, image.new_tensor(.4), image.new_tensor(.65), image.new_tensor(.5))
        negative = apply_split_palette_projection(image, image.new_tensor(-.4), rgb_to_yiq,
            yiq_to_rgb, swatches, image.new_tensor(.4), image.new_tensor(.65), image.new_tensor(.5))
        positive_yiq = torch.einsum("ij,bjhw->bihw", rgb_to_yiq, positive)
        negative_yiq = torch.einsum("ij,bjhw->bihw", rgb_to_yiq, negative)
        target_yiq = torch.einsum("ij,pkj->pki", rgb_to_yiq, swatches)
        positive_targets = target_yiq[1, :, 1:3].T.view(1, 2, 1, 2)
        negative_targets = target_yiq[0, :, 1:3].T.view(1, 2, 1, 2)
        thresholds = torch.quantile(luminance.flatten(2), torch.tensor([.15, .85]), dim=-1)
        low = thresholds[0].view(-1, 1, 1, 1)
        high = thresholds[1].view(-1, 1, 1, 1)
        progress = ((luminance - low) / (high - low).clamp_min(1e-4)).clamp(0, 1)
        weights = (progress.square() * (3 - 2 * progress))[:, :, :, ::63]
        expected_positive = .65 * (positive_targets[:, :, :, :1] * (1 - weights)
                                   + positive_targets[:, :, :, 1:] * weights)
        expected_negative = .5 * (negative_targets[:, :, :, :1] * (1 - weights)
                                  + negative_targets[:, :, :, 1:] * weights)
        torch.testing.assert_close(positive_yiq[:, 1:3, :, ::63], expected_positive.expand(1, 2, 32, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close(negative_yiq[:, 1:3, :, ::63], expected_negative.expand(1, 2, 32, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close(positive_yiq[:, :1], negative_yiq[:, :1], atol=1e-6, rtol=0)

    def test_palette_nearest_swatch_pulls_each_pixel_toward_closest_endpoint(self):
        config = seed_config()
        matrix = torch.tensor(config["palette_projection"]["rgb_to_yiq"])
        inverse = torch.tensor(config["palette_projection"]["yiq_to_rgb"])
        for key, amount in (("blockbuster", .4), ("retro", -.4)):
            endpoints = torch.tensor(config["palette_projection"]["swatches"][key])
            pixels = torch.stack((.8 * endpoints[0] + .2 * endpoints[1],
                                  .2 * endpoints[0] + .8 * endpoints[1]), dim=0)
            image = pixels.T.view(1, 3, 1, 2)
            output = apply_nearest_swatch_palette(image, torch.tensor(amount), matrix, inverse,
                torch.tensor([config["palette_projection"]["swatches"]["retro"],
                             config["palette_projection"]["swatches"]["blockbuster"]]),
                torch.tensor(.4), torch.tensor(.85), torch.tensor(.75), .18)
            before = torch.einsum("ij,bjhw->bihw", matrix, image)[:, 1:3, 0].transpose(1, 2)
            after = torch.einsum("ij,bjhw->bihw", matrix, output)[:, 1:3, 0].transpose(1, 2)
            swatch_chroma = torch.einsum("ij,kj->ki", matrix, endpoints)[:, 1:]
            chosen = torch.tensor([0, 1])
            before_distance = (before[0] - swatch_chroma[chosen]).norm(dim=-1)
            after_distance = (after[0] - swatch_chroma[chosen]).norm(dim=-1)
            self.assertTrue((after_distance < before_distance).all())
            torch.testing.assert_close(_luminance(output), _luminance(image), atol=1e-6, rtol=0)

    def test_six_axis_projection_and_soft_band_full_activation(self):
        config = seed_config()
        config["axes"] = ["warmth", "saturation", "brightness", "contrast", "sharpness", "palette"]
        config["poles"] = {axis: {"negative": {"texts": [f"negative {axis}"]},
                                  "positive": {"texts": [f"positive {axis}"]}}
                            for axis in config["axes"]}
        vectors = {}
        for index, axis in enumerate(config["axes"]):
            positive = torch.eye(6)[index]
            vectors[f"positive {axis}"] = positive
            vectors[f"negative {axis}"] = -positive
        config["relevance_gate"] = {
            "feature_mean": [[0.] * 5] * 6, "feature_scale": [[1.] * 5] * 6,
            "coefficients": [[0.] * 5] * 6,
            "intercepts": [2.] * 6, "thresholds": [.1] * 6, "full_weight_at": [.8] * 6,
        }
        projection = SemanticProjection(lambda texts: torch.stack([vectors[t] for t in texts]), config)
        _, coordinates = projection.project_features(torch.eye(6))
        self.assertEqual(tuple(coordinates.shape), (6, 6))
        _, weights = projection.relevance_weights(torch.eye(6))
        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_gate_rejects_far_word_despite_equal_pole_preference(self):
        config = seed_config()
        negative = torch.tensor([-.6, .8, 0.])
        positive = torch.tensor([.6, .8, 0.])
        embeddings = {}
        for axis in AXES:
            for sign, embedding in (("negative", negative), ("positive", positive)):
                for text in config["poles"][axis][sign]["texts"]:
                    embeddings[text] = embedding
        config["relevance_gate"] = {
            "feature_mean": [[0.] * 5] * 5, "feature_scale": [[1.] * 5] * 5,
            "coefficients": [[10., 0., 0., 0., 0.]] * 5,
            "intercepts": [-4.] * 5, "thresholds": [.8] * 5,
        }
        projection = SemanticProjection(lambda texts: torch.stack([embeddings[t] for t in texts]), config)
        features = torch.tensor([[.1, .99 ** .5, 0.], [.1, 0., .99 ** .5]])
        scores, coordinates = projection.project_features(features)
        torch.testing.assert_close(scores[0], scores[1])
        self.assertTrue((coordinates[0].abs() > 0).all())
        self.assertTrue((coordinates[1] == 0).all())

    def test_physical_control_gradients_and_neutral(self):
        torch.manual_seed(91)
        raw = neutral_raw_controls().requires_grad_()
        controls = bounded_controls(raw)
        torch.testing.assert_close(controls, torch.tensor(list(CONTROL_NEUTRAL)))
        image = .05 + .7 * torch.rand(1, 3, 16, 16)
        output = render_linear(image, parameters_from_controls(controls, seed_config()))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad.abs() > 1e-8).all(), raw.grad)

    def test_statistical_calibration_fits_known_controls(self):
        from exp38_learn_poles import fit_calibration
        import numpy as np
        random = np.random.default_rng(13)
        scores = random.uniform(-.1, .1, (40, 5))
        targets = np.tile(neutral_raw_controls().numpy(), (40, 1))
        targets[:, 0] += scores[:, 0] * 3
        config = fit_calibration(scores, np.zeros(5), targets, seed_config())
        self.assertEqual(len(config["learned_calibration"]["pole_raw_deltas"]), 10)
        coordinates = torch.tensor([.2, 0., 0., 0., 0.])
        parameters = compose_parameters(coordinates, config)
        self.assertGreater(parameters["wb"][0].item(), 1.0)
        neutral = compose_parameters(torch.zeros(5), config)
        expected_wb = torch.stack((1 + torch.tensor(CONTROL_NEUTRAL[0]),
                                   torch.ones(()), 1 - torch.tensor(CONTROL_NEUTRAL[0])))
        torch.testing.assert_close(neutral["wb"], expected_wb, atol=1e-4, rtol=1e-4)
        self.assertNotIn("black_point", parameters)

    def test_clip_backend_and_hue_uses_projection_not_ccm(self):
        config = seed_config()
        self.assertEqual((config["model"], config["pretrained"]), ("ViT-B-32", "openai"))
        self.assertTrue(config["force_quick_gelu"])
        self.assertEqual(sum(len(config["poles"][axis][sign]["texts"])
                             for axis in AXES for sign in ("negative", "positive")), 40)
        controls = torch.tensor([0., 0., 1., 1., 0., .4])
        parameters = parameters_from_controls(controls, config)
        torch.testing.assert_close(parameters["ccm"], torch.eye(3))
        self.assertAlmostEqual(parameters["palette_coordinate"].item(), .4, places=6)
        self.assertAlmostEqual(parameters["saturation"].item(), 0.0, places=6)
        self.assertNotIn("black_point", parameters)

    def test_projection_formula_and_cached_centroids(self):
        config = seed_config()
        vectors = {}
        for index, axis in enumerate(AXES):
            for sign, direction in (("negative", -1), ("positive", 1)):
                for text in config["poles"][axis][sign]["texts"]:
                    vector = torch.zeros(5)
                    vector[index] = direction
                    vectors[text] = vector
        vectors["warm"] = torch.tensor([1., 0., 0., 0., 0.])
        calls = []

        def encode(texts):
            calls.append(list(texts))
            return torch.stack([vectors[text] for text in texts])

        projection = SemanticProjection(encode, config)
        self.assertEqual(len(calls), 10)
        raw, coordinates = projection.project(["warm"])
        torch.testing.assert_close(raw, torch.tensor([[2., 0., 0., 0., 0.]]))
        torch.testing.assert_close(coordinates, torch.tensor([[0.4, 0., 0., 0., 0.]]))
        self.assertEqual(len(calls), 11)

    def test_project_centroid_embeddings_directly(self):
        config = seed_config()
        vectors = {}
        for index, axis in enumerate(AXES):
            for sign, direction in (("negative", -1), ("positive", 1)):
                for text in config["poles"][axis][sign]["texts"]:
                    value = torch.zeros(5)
                    value[index] = direction
                    vectors[text] = value
        projection = SemanticProjection(
            lambda texts: torch.stack([vectors[text] for text in texts]), config)
        scores, coordinates = projection.project_features(projection.centroids.reshape(10, 5))
        self.assertEqual(tuple(scores.shape), (10, 5))
        self.assertEqual(tuple(coordinates.shape), (10, 5))
        self.assertTrue(torch.isfinite(coordinates).all())

    def test_selected_pole_centroids_average_unit_embeddings_and_mapping_normalizes_queries(self):
        config = seed_config()
        axis = AXES[0]
        config["axes"] = [axis]
        selected = {axis: {"negative": ["long", "short"],
                           "positive": ["positive_a", "positive_b"]}}
        config["learned_calibration"] = {
            "pole_centroid_texts": selected,
            "score_origin": [0.], "positive_scale": [1.], "negative_scale": [1.],
        }
        encoded = {
            "long": torch.tensor([10., 0., 0.]),
            "short": torch.tensor([0., 1., 0.]),
            "positive_a": torch.tensor([1., 0., 0.]),
            "positive_b": torch.tensor([1., 1., 0.]),
        }
        seen = []

        def encode(texts):
            seen.extend(texts)
            return torch.stack([encoded[text] for text in texts])

        projection = SemanticProjection(encode, config)
        expected = F.normalize(torch.tensor([.5, .5, 0.]), dim=0)
        torch.testing.assert_close(projection.centroids[0, 0], expected)
        self.assertEqual(seen, selected[axis]["negative"] + selected[axis]["positive"])
        feature = torch.tensor([[2., 1., 0.]])
        _, coordinates = projection.project_features(feature)
        _, scaled_coordinates = projection.project_features(feature * 17.)
        torch.testing.assert_close(coordinates, scaled_coordinates)

    def test_neutral_composition_and_linear_render_identity(self):
        parameters = compose_parameters(torch.zeros(5), seed_config())
        torch.testing.assert_close(parameters["ccm"], torch.eye(3))
        torch.testing.assert_close(parameters["wb"], torch.ones(3))
        image = 0.05 + 0.8 * torch.rand(1, 3, 16, 16)
        torch.testing.assert_close(render_linear(image, parameters), image, rtol=0, atol=2e-6)

    def test_composition_rules(self):
        config = seed_config()
        config["poles"]["fidelity"]["positive"]["tone_gain"] = 1.2
        coordinates = torch.tensor([0.2, 0.3, 0.4, 0.1, -0.2])
        result = compose_parameters(coordinates, config)
        self.assertAlmostEqual(result["tone_gain"].item(), (1 + .3 * .2) * (1 + .4 * .3), places=6)
        self.assertAlmostEqual(result["sharpness"].item(), .3 * .8, places=6)
        self.assertNotIn("black_point", result)

    def test_all_axis_gradients_and_render_bounds(self):
        torch.manual_seed(9)
        coordinates = torch.tensor([0.1, -0.2, -0.1, 0.2, -0.1], requires_grad=True)
        image = 0.05 + 0.7 * torch.rand(1, 3, 24, 24)
        output = render_linear(image, compose_parameters(coordinates, seed_config()))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(output.min().item(), 0)
        self.assertLessEqual(output.max().item(), 1)
        self.assertTrue(torch.isfinite(coordinates.grad).all())
        self.assertTrue((coordinates.grad.abs() > 1e-8).all(), coordinates.grad)

    def test_tone_gain_bounds_produce_strong_monotone_brightness_endpoints(self):
        self.assertEqual(CONTROL_BOUNDS[2], (.05, 5.0))
        ramp = torch.linspace(.01, .35, 128).view(1, 1, 1, 128).expand(1, 3, 32, 128)
        dark_controls = torch.tensor([0., 0., .05, 1., 0., 0.])
        bright_controls = torch.tensor([0., 0., 5., 1., 0., 0.])
        dark = render_linear(ramp, parameters_from_controls(dark_controls, seed_config()))
        bright = render_linear(ramp, parameters_from_controls(bright_controls, seed_config()))
        dark_median = _luminance(dark).flatten().median()
        bright_median = _luminance(bright).flatten().median()
        self.assertLess(float(dark_median), .02)
        self.assertGreater(float(bright_median), .5)
        self.assertTrue(torch.all(_luminance(dark)[..., 1:] >= _luminance(dark)[..., :-1]))
        self.assertTrue(torch.all(_luminance(bright)[..., 1:] >= _luminance(bright)[..., :-1]))

    def test_negative_contrast_is_monotone_without_amplifying_srgb_chroma(self):
        controls = torch.tensor([0., 0., 1., .2, 0., 0.])
        parameters = parameters_from_controls(controls, seed_config())
        ramp = torch.linspace(.0005, .04, 256).view(1, 1, 1, 256).expand(1, 3, 32, 256).clone()
        noise = (.0002 * (torch.arange(256) % 2).float().mul(2).sub(1)).view(1, 1, 1, 256)
        ramp[:, 0] += noise[:, 0]
        ramp[:, 2] -= noise[:, 0]
        reduced = render_linear(ramp, parameters)
        neutral_parameters = dict(parameters, contrast=torch.ones_like(parameters["contrast"]))
        neutral = render_linear(ramp, neutral_parameters)
        encoded_reduced = linear_to_srgb(reduced)
        encoded_neutral = linear_to_srgb(neutral)
        reduced_luma = _luminance(encoded_reduced)
        neutral_luma = _luminance(encoded_neutral)
        reduced_chroma = (encoded_reduced - reduced_luma).norm(dim=1)
        neutral_chroma = (encoded_neutral - neutral_luma).norm(dim=1)
        self.assertTrue(torch.all(reduced_chroma <= neutral_chroma + 1e-5))
        self.assertTrue(torch.all(reduced_luma[..., 1:] >= reduced_luma[..., :-1] - 1e-6))
        self.assertTrue(torch.isfinite(reduced).all())

    def test_negative_fidelity_really_blurs(self):
        grid = torch.arange(24)
        pattern = ((grid[:, None] + grid[None, :]) % 2).float() * .4 + .3
        image = pattern[None, None].expand(1, 3, 24, 24)
        config = seed_config()
        config["poles"]["fidelity"]["negative"]["contrast"] = 1.0
        output = render_linear(image, compose_parameters(torch.tensor([0., -.4, 0., 0., 0.]), config))
        self.assertLess(output.var().item(), image.var().item())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()