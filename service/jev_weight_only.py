"""Portable per-output-channel INT8/FP8 storage with BF16 linear computation.

No native FP8 GEMM or external quantization library is required. This is
weight-only quantization, not an accelerated W8A8 inference backend.
"""
import torch
from torch import nn
from torch.nn import functional as F


class WeightOnlyLinear(nn.Module):
    def __init__(self, source, kind='int8'):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.out_dtype = source.weight.dtype
        weight = source.weight.detach().float()
        maximum = 127.0 if kind == 'int8' else 448.0
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / maximum
        normalized = (weight / scale).clamp(-maximum, maximum)
        quantized = normalized.round().to(torch.int8) if kind == 'int8' else normalized.to(torch.float8_e4m3fn)
        self.register_buffer('quantized', quantized.contiguous())
        self.register_buffer('scale', scale.to(source.weight.dtype))
        self.register_buffer('bias', None if source.bias is None else source.bias.detach().clone())

    def forward(self, x):
        weight = self.quantized.to(self.out_dtype) * self.scale
        return F.linear(x, weight, self.bias)


def convert_weight_only(model, kind='int8', min_dim=1024):
    count = 0
    saved = 0
    # Do not retain a list of all original modules throughout conversion.
    def visit(parent):
        nonlocal count, saved
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= min_dim:
                replacement = WeightOnlyLinear(child, kind)
                saved += child.weight.numel() - replacement.scale.numel() * replacement.scale.element_size()
                setattr(parent, name, replacement)
                count += 1
            else:
                visit(child)
    visit(model)
    return {'format': kind + '-weight-only', 'layers': count, 'saved_mib': round(saved / 2**20, 2)}
