import itertools
import unittest
from unittest.mock import patch

import torch

import differentiable_isp as isp_module
from differentiable_isp import DifferentiableEMOVIS


def luma(rgb):
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


class GamutCompressionTests(unittest.TestCase):
    def test_safe_colors_and_identity_are_unchanged(self):
        torch.manual_seed(17)
        reference = 0.2 + 0.4 * torch.rand(2, 3, 8, 8)
        self.assertTrue(torch.equal(isp_module._compress_gamut(reference, reference), reference))
        target = 0.25 + 0.2 * torch.rand_like(reference)
        self.assertTrue(torch.equal(isp_module._compress_gamut(target, reference), target))
        saturated = torch.tensor([0.98, 0.2, 0.02]).reshape(1, 3, 1, 1)
        self.assertTrue(torch.equal(isp_module._compress_gamut(saturated, saturated), saturated))

    def test_bounds_luma_and_chroma_direction(self):
        torch.manual_seed(19)
        reference = 0.2 + 0.4 * torch.rand(2, 3, 16, 16, dtype=torch.float64)
        target_luma = 0.05 + 0.9 * torch.rand(2, 1, 16, 16, dtype=torch.float64)
        chroma = 3 * torch.randn_like(reference)
        chroma = chroma - luma(chroma)
        target = target_luma + chroma
        out = isp_module._compress_gamut(target, reference)
        self.assertGreater(out.min().item(), 0)
        self.assertLess(out.max().item(), 1)
        torch.testing.assert_close(luma(out), target_luma, rtol=0, atol=2e-14)
        output_chroma = out - luma(out)
        scale = (output_chroma * chroma).sum(dim=1, keepdim=True) / chroma.square().sum(dim=1, keepdim=True)
        torch.testing.assert_close(output_chroma, scale * chroma, rtol=1e-11, atol=1e-13)
        self.assertTrue(torch.all(scale >= 0))
        self.assertTrue(torch.all(scale <= 1 + 1e-12))

    def test_gray_black_white_and_unattainable_luma(self):
        gray = torch.linspace(0, 1, 41).reshape(1, 1, 1, -1).expand(1, 3, 1, -1)
        self.assertTrue(torch.equal(isp_module._compress_gamut(gray, gray), gray))
        target = torch.tensor([-0.5, -0.5, -0.5, 1.5, 1.5, 1.5]).reshape(2, 3, 1, 1)
        reference = torch.full_like(target, 0.5)
        out = isp_module._compress_gamut(target, reference)
        self.assertTrue(torch.equal(out[0], torch.zeros_like(out[0])))
        self.assertTrue(torch.equal(out[1], torch.ones_like(out[1])))

    def test_compression_does_not_flatten_red_to_one(self):
        red = torch.linspace(0.9, 1.5, 64, dtype=torch.float64)
        target = torch.stack((red, torch.full_like(red, 0.2), torch.full_like(red, 0.1)))[None, :, None, :]
        reference = torch.full_like(target, 0.4)
        out = isp_module._compress_gamut(target, reference)
        self.assertLess(out.max().item(), 1)
        self.assertTrue(torch.all(out[0, 0, 0].diff() > 0))
        torch.testing.assert_close(luma(out), luma(target), rtol=0, atol=1e-14)

    def test_successive_gains_do_not_consume_all_headroom(self):
        image = torch.tensor([0.99, 0.1, 0.05]).reshape(1, 3, 1, 1)
        for _ in range(60):
            image = isp_module._compress_gamut(image * 1.005, image)
            self.assertLess(image.max().item(), 1 - 1e-4)

    def test_gradcheck(self):
        target = torch.tensor([[1.2, 0.2, -0.1], [0.2, 0.4, 0.7]], dtype=torch.float64)
        target = target.reshape(2, 3, 1, 1).requires_grad_()
        reference = torch.tensor([[0.25, 0.4, 0.55], [0.3, 0.4, 0.5]], dtype=torch.float64)
        reference = reference.reshape(2, 3, 1, 1).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(isp_module._compress_gamut, (target, reference)))

    def test_native_gamut_boundary_has_finite_gradients(self):
        reference = torch.tensor([1.0, 0.2, 0.0], dtype=torch.float64).reshape(1, 3, 1, 1).requires_grad_()
        target = torch.tensor([1.3, 0.15, -0.1], dtype=torch.float64).reshape(1, 3, 1, 1).requires_grad_()
        out = isp_module._compress_gamut(target, reference)
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(target.grad).all())
        self.assertTrue(torch.isfinite(reference.grad).all())
        self.assertGreaterEqual(out.min().item(), 0)
        self.assertLessEqual(out.max().item(), 1)


class PipelineGamutTests(unittest.TestCase):
    def test_all_stages_preserve_feasible_luma(self):
        torch.manual_seed(37)
        image = torch.rand(1, 3, 24, 24)
        isp = DifferentiableEMOVIS("cpu")
        isp.set_params({"wb_warmth": 0.2, "saturation": 0.8, "brightness": 0.4,
                        "sharpness": 0.5, "contrast": -0.5})
        compress = isp_module._compress_gamut
        stages = []

        def recording_compress(rgb, reference):
            out = compress(rgb, reference)
            stages.append((rgb.detach(), out.detach()))
            return out

        with patch.object(isp_module, "_compress_gamut", side_effect=recording_compress):
            isp(image)
        self.assertEqual(len(stages), 5)
        for requested, actual in stages:
            target_luma = luma(requested)
            feasible = (target_luma >= 0) & (target_luma <= 1)
            torch.testing.assert_close(luma(actual)[feasible], target_luma[feasible], rtol=0, atol=2e-7)
            self.assertGreaterEqual(actual.min().item(), 0)
            self.assertLessEqual(actual.max().item(), 1)

    def test_no_rgb_overflow_before_clamp(self):
        original_clamp = torch.clamp
        for rgb in ([0.9, 0.45, 0.2], [0.9, 0.2, 0.1]):
            with self.subTest(rgb=rgb):
                image = torch.tensor(rgb).reshape(1, 3, 1, 1).expand(1, 3, 16, 16).clone()
                isp = DifferentiableEMOVIS("cpu")
                isp.set_params({"saturation": 0.4, "contrast": 0.0})
                inputs = []

                def recording_clamp(value, *args, **kwargs):
                    if value.ndim == 4 and value.shape[1] == 3 and args == (0, 1):
                        inputs.append(value.detach().clone())
                    return original_clamp(value, *args, **kwargs)

                with patch.object(isp_module.torch, "clamp", side_effect=recording_clamp):
                    isp(image)
                self.assertTrue(inputs)
                for value in inputs:
                    self.assertGreaterEqual(value.min().item(), -1e-6)
                    self.assertLessEqual(value.max().item(), 1 + 1e-6)

    def test_parameter_extremes_and_backward(self):
        torch.manual_seed(23)
        image = torch.rand(3, 3, 12, 12)
        image[0] = 0
        image[1] = 1
        image[2, :, 3, 3] = torch.tensor([1.0, 0.0, 0.0])
        isp = DifferentiableEMOVIS("cpu")
        original_clamp = torch.clamp

        def checked_clamp(value, *args, **kwargs):
            if value.ndim == 4 and value.shape[1] == 3 and args == (0, 1):
                self.assertGreaterEqual(value.min().item(), -1e-6)
                self.assertLessEqual(value.max().item(), 1 + 1e-6)
            return original_clamp(value, *args, **kwargs)

        for signs in itertools.product((-1, 1), repeat=5):
            params = {name: sign * (0.3 if name == "wb_warmth" else 0.5 if name == "brightness" else 1.0)
                      for name, sign in zip(isp.PARAM_NAMES, signs)}
            with self.subTest(signs=signs):
                isp.set_params(params)
                isp.zero_grad(set_to_none=True)
                with patch.object(isp_module.torch, "clamp", side_effect=checked_clamp):
                    out = isp(image)
                self.assertTrue(torch.isfinite(out).all())
                self.assertGreaterEqual(out.min().item(), 0)
                self.assertLessEqual(out.max().item(), 1)
                out.square().mean().backward()
                self.assertTrue(torch.isfinite(isp.raw_params.grad).all())

    def test_all_five_parameters_receive_gradients(self):
        torch.manual_seed(29)
        image = 0.05 + 0.7 * torch.rand(1, 3, 24, 24)
        isp = DifferentiableEMOVIS("cpu")
        isp.set_params({"wb_warmth": 0.1, "saturation": 0.4, "brightness": 0.3,
                        "sharpness": 0.2, "contrast": 0.4})
        isp(image).square().mean().backward()
        grad = isp.raw_params.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.all(grad.abs() > 1e-8), grad)

    def test_zero_contrast_matches_with_and_without_grad(self):
        torch.manual_seed(31)
        image = torch.rand(1, 3, 24, 24)
        isp = DifferentiableEMOVIS("cpu")
        isp.set_params({"saturation": 0.6, "brightness": 0.3, "contrast": 0.0})
        with torch.no_grad():
            no_grad = isp(image)
        with_grad = isp(image)
        torch.testing.assert_close(with_grad, no_grad, rtol=0, atol=1e-6)

    def test_defaults(self):
        isp = DifferentiableEMOVIS("cpu")
        expected = {"wb_warmth": 0.022, "saturation": 0.205, "brightness": 0.116,
                    "sharpness": -0.055, "contrast": 0.273}
        for name, value in expected.items():
            self.assertAlmostEqual(isp.get_params()[name], value, places=6)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
