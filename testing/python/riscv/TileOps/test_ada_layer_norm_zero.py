"""Original upstream AdaLN-Zero kernel against a two-pass CPU reference."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [19, 256])
def test_ada_layer_norm_zero_runtime_compare(dtype, width):
    get_kernel_class("norm.ada_layer_norm", "AdaLayerNormZeroKernel")
    from tileops.kernels.norm.ada_layer_norm import _ada_layer_norm_kernel

    torch.manual_seed(127)
    rows, eps = 3, 1e-5
    x = (torch.randn(rows, width) * 0.3).to(dtype)
    scale = (torch.randn(rows, width) * 0.1 + 1).to(dtype)
    shift = (torch.randn(rows, width) * 0.1).to(dtype)
    gate = (torch.randn(rows, width) * 0.2).to(dtype)
    dummy = torch.empty(1, dtype=dtype)
    originals = [v.clone() for v in (x, scale, shift, gate)]
    factory = _ada_layer_norm_kernel(rows, width, eps, str(dtype).split(".")[-1], has_gate=True, use_cp_async=False)
    kernel = compile_tileops_jit(factory, {"block_m": 1, "threads": 128})
    try:
        actual = kernel(x, scale, shift, gate, dummy)
    finally:
        kernel.close()
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    variance = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
    expected = ((scale.float() * (xf - mean) * torch.rsqrt(variance + eps) + shift.float()) * gate.float()).to(dtype)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
    for value, original in zip((x, scale, shift, gate), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
