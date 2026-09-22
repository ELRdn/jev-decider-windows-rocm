import unittest

import torch
from service.jev_weight_only import WeightOnlyLinear, convert_weight_only


class WeightOnlyTests(unittest.TestCase):
    def test_quantized_layer_retains_shape_bias_and_close_outputs(self):
        torch.manual_seed(42)
        original = torch.nn.Linear(64, 48, bias=True).to(torch.bfloat16).eval()
        quantized = WeightOnlyLinear(original)
        x = torch.randn(2, 7, 64, dtype=torch.bfloat16)
        with torch.no_grad():
            expected, actual = original(x), quantized(x)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(quantized.quantized.dtype, torch.int8)
        relative_error = (actual.float() - expected.float()).abs().mean() / expected.float().abs().mean()
        self.assertLess(float(relative_error), .02)
        torch.testing.assert_close(quantized.bias, original.bias)

    def test_zero_weights_are_finite(self):
        original = torch.nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            original.weight.zero_()
        quantized = WeightOnlyLinear(original)
        actual = quantized(torch.ones(3, 8))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(int(torch.count_nonzero(actual)), 0)

    def test_conversion_preserves_small_layers_and_embedding(self):
        model = torch.nn.ModuleDict({'embedding': torch.nn.Embedding(16, 8),
            'large': torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16),
            'small': torch.nn.Linear(4, 8, bias=False).to(torch.bfloat16)})
        original_embedding, original_small = model['embedding'], model['small']
        result = convert_weight_only(model, min_dim=8)
        self.assertEqual(result['layers'], 1)
        self.assertIs(model['embedding'], original_embedding)
        self.assertIs(model['small'], original_small)
        self.assertIsInstance(model['large'], WeightOnlyLinear)
