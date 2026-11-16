import unittest

import torch

from axis_isp import MoranAxisISP, linear_to_srgb8
from differentiable_isp import DifferentiableEMOVIS


class MoranAxisISPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.image = 0.05 + 0.8 * torch.rand(1, 3, 24, 24)
        self.isp = MoranAxisISP("cpu")

    def test_sweep_range_is_plus_minus_0_4(self):
        self.assertAlmostEqual(MoranAxisISP.AXIS_STEPS[0], -0.4)
        self.assertAlmostEqual(MoranAxisISP.AXIS_STEPS[-1], 0.4)
        self.assertEqual(MoranAxisISP.AXIS_STEPS[3], 0.0)

    def test_explicit_defaults_are_used_as_neutral_axis_anchor(self):
        defaults = {name: 0.0 for name in DifferentiableEMOVIS.PARAM_NAMES}
        defaults["contrast"] = 0.4
        isp = MoranAxisISP("cpu", defaults=defaults)
        controls = isp.mapped_controls()
        for name, value in defaults.items():
            self.assertAlmostEqual(controls[name].item(), value, places=6)

    def test_linear_float_output_and_zero_palette_axis_is_identity(self):
        output = self.isp(self.image)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, self.image.shape)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(
            self.isp._attract_to_palette_axis(self.image, torch.tensor(0.0)),
            self.image,
            rtol=0,
            atol=2e-6,
        )

    def test_output_transfer_and_8bit_quantization(self):
        linear = torch.tensor([0.0, 0.0031308, 0.18, 1.0]).reshape(1, 1, 1, 4).expand(1, 3, 1, 4)
        encoded = linear_to_srgb8(linear)
        self.assertEqual(encoded.dtype, torch.uint8)
        self.assertEqual(encoded.min().item(), 0)
        self.assertEqual(encoded.max().item(), 255)
        self.assertGreater(encoded[0, 0, 0, 2].item(), round(0.18 * 255))
        self.assertEqual(encoded[0, 0, 0, 1].item(), 10)

    def test_all_axes_have_finite_nonzero_gradients(self):
        self.isp.set_axes({"thermal": 0.4, "fidelity": -0.5, "moral": 0.3,
                           "vitality": 0.2, "hue_shift": -0.4})
        self.isp(self.image).square().mean().backward()
        gradients = self.isp.raw_axes.grad
        self.assertTrue(torch.isfinite(gradients).all())
        self.assertTrue(torch.all(gradients.abs() > 1e-8), gradients)

    def test_fidelity_negative_pole_performs_real_blur(self):
        coordinates = torch.arange(32)
        checker = ((coordinates[:, None] + coordinates[None, :]) % 2).float()
        checker = (0.25 + 0.5 * checker).reshape(1, 1, 32, 32).expand(1, 3, 32, 32)
        unchanged = self.isp._apply_sharpness(checker, torch.tensor(0.0))
        blurred = self.isp._apply_sharpness(checker, torch.tensor(-0.5))
        unchanged_detail = (unchanged[..., 1:] - unchanged[..., :-1]).abs().mean()
        blurred_detail = (blurred[..., 1:] - blurred[..., :-1]).abs().mean()
        self.assertLess(blurred_detail.item(), 0.6 * unchanged_detail.item())

    def test_mapped_fidelity_contrast_and_monochrome_endpoints(self):
        self.isp.set_axes({"fidelity": -0.4})
        soft = self.isp.mapped_controls()
        self.assertAlmostEqual(soft["sharpness"].item(), -0.515, places=5)
        self.assertAlmostEqual(soft["contrast"].item(), -0.027, places=5)

        self.isp.set_axes({"fidelity": 0.4})
        documentary = self.isp.mapped_controls()
        self.assertAlmostEqual(documentary["sharpness"].item(), 0.405, places=5)
        self.assertAlmostEqual(documentary["contrast"].item(), 0.573, places=5)

        self.isp.set_axes({"vitality": -0.4})
        self.assertEqual(self.isp.mapped_controls()["saturation"].item(), -1.0)
        monochrome = self.isp(self.image)
        self.assertLess((monochrome[:, 0] - monochrome[:, 1]).abs().max().item(), 2e-6)
        self.assertLess((monochrome[:, 1] - monochrome[:, 2]).abs().max().item(), 2e-6)

    def test_moral_high_key_is_unchanged_and_dark_pole_is_darker(self):
        self.isp.set_axes({"moral": 0.4})
        high_key = self.isp.mapped_controls()["brightness"].item()
        self.assertAlmostEqual(high_key, 0.236, places=5)
        self.isp.set_axes({"moral": -0.4})
        dark = self.isp.mapped_controls()["brightness"].item()
        self.assertAlmostEqual(dark, MoranAxisISP.MORAL_DARKNESS_ENDPOINT, places=5)

    def test_single_axis_sweep_is_bounded_and_other_axes_stay_zero(self):
        for axis in MoranAxisISP.AXES:
            for coordinate in MoranAxisISP.AXIS_STEPS:
                self.isp.set_axes({axis: coordinate})
                values = self.isp.axis_values().detach()
                for index, name in enumerate(MoranAxisISP.AXES):
                    if name != axis:
                        self.assertAlmostEqual(values[index].item(), 0.0, places=6)
                output = self.isp(self.image)
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreaterEqual(output.min().item(), 0.0)
                self.assertLessEqual(output.max().item(), 1.0)
                controls = self.isp.mapped_controls()
                for name, value in controls.items():
                    lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
                    self.assertGreaterEqual(value.item(), lo)
                    self.assertLessEqual(value.item(), hi)

    def test_palette_axis_attracts_partially_and_moves_in_signed_direction(self):
        image_yiq = torch.einsum("ij,bjhw->bihw", self.isp.rgb_to_yiq, self.image)
        chroma = image_yiq[:, 1:3]
        negative, positive = self.isp.palette_axis_points.unbind(dim=0)
        direction = positive - negative
        axis = direction / direction.norm()

        def distance_to_line(candidate):
            yiq = torch.einsum("ij,bjhw->bihw", self.isp.rgb_to_yiq, candidate)
            delta = yiq[:, 1:3] - negative.view(1, 2, 1, 1)
            perpendicular = delta - (delta * axis.view(1, 2, 1, 1)).sum(
                dim=1, keepdim=True
            ) * axis.view(1, 2, 1, 1)
            return perpendicular.square().sum(dim=1).sqrt().mean()

        before_distance = distance_to_line(self.image)
        retro = self.isp._attract_to_palette_axis(self.image, torch.tensor(-1.0))
        blockbuster = self.isp._attract_to_palette_axis(self.image, torch.tensor(1.0))
        self.assertLess(distance_to_line(retro).item(), before_distance.item())
        self.assertLess(distance_to_line(blockbuster).item(), before_distance.item())
        self.assertGreater(distance_to_line(retro).item(), 0.4 * before_distance.item())

        def along_axis(candidate):
            yiq = torch.einsum("ij,bjhw->bihw", self.isp.rgb_to_yiq, candidate)
            return ((yiq[:, 1:3] - negative.view(1, 2, 1, 1))
                    * axis.view(1, 2, 1, 1)).sum(dim=1).mean()

        self.assertGreater(along_axis(blockbuster).item(), along_axis(self.image).item())
        self.assertLess(along_axis(retro).item(), along_axis(self.image).item())
        original_luma = image_yiq[:, :1]
        for candidate in (retro, blockbuster):
            transformed_luma = torch.einsum("ij,bjhw->bihw", self.isp.rgb_to_yiq,
                                            candidate)[:, :1]
            torch.testing.assert_close(transformed_luma, original_luma, rtol=0, atol=2e-7)

    def test_invalid_axis_values_are_rejected(self):
        with self.assertRaises(ValueError):
            self.isp.set_axes({"thermal": 1.1})
        with self.assertRaises(ValueError):
            self.isp.set_axes({"unknown": 0.0})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
