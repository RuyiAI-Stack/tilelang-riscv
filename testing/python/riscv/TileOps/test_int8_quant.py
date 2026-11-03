"""Original INT8 quantizers with independent float32 and nearest-even references."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


def _check_quantized(x, codes, scales, *, block_size=None):
    expected_scales = torch.empty_like(scales)
    expected_codes = torch.empty_like(codes)
    block_size = block_size or x.shape[1]
    for row in range(x.shape[0]):
        for col in range((x.shape[1] + block_size - 1) // block_size):
            block = x[row, col * block_size : (col + 1) * block_size].float()
            amax = block.abs().amax()
            scale = torch.where(amax > 0, amax * torch.tensor(1.0 / 127.0), torch.ones_like(amax))
            expected_scales.reshape(x.shape[0], -1)[row, col] = scale
            expected_codes[row, col * block_size : (col + 1) * block_size] = (block / scale).round().clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(codes, expected_codes, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", [33, 128])
@pytest.mark.parametrize("smoothed", [False, True])
def test_int8_row_quantizers_runtime_compare(dtype, cols, smoothed):
    class_name = "SmoothQuantFwdKernel" if smoothed else "INT8QuantPerChannelFwdKernel"
    get_kernel_class("quantization.int8_quant_per_channel", class_name)
    from tileops.kernels.quantization.int8_quant_per_channel import _int8_quant_per_channel_kernel, _smooth_quant_kernel

    torch.manual_seed(177)
    x = torch.randn(3, cols).to(dtype)
    x[0].zero_()
    x[2] *= 2.0**-10
    before = x.view(torch.uint8).clone()
    smooth = torch.linspace(0.5, 2.0, cols)
    smooth_before = smooth.clone()
    codes = torch.full(x.shape, -128, dtype=torch.int8)
    scales = torch.full((3,), float("nan"))
    builder = _smooth_quant_kernel if smoothed else _int8_quant_per_channel_kernel
    factory = builder(3, cols, str(dtype).split(".")[-1])
    cfg = dict(threads=32, vpt=1, pair=False, evict_first=True, min_blocks=1)
    if smoothed:
        cfg["ctas"] = 3
    kernel = compile_tileops_jit(factory, cfg)
    try:
        if smoothed:
            kernel(x.reshape(-1), smooth, codes.reshape(-1), scales)
        else:
            kernel(x.reshape(-1), codes.reshape(-1), scales)
    finally:
        kernel.close()
    _check_quantized(x.float() / smooth if smoothed else x, codes, scales)
    torch.testing.assert_close(x.view(torch.uint8), before, atol=0, rtol=0)
    torch.testing.assert_close(smooth, smooth_before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", [128, 131])
def test_int8_block_quantizers_runtime_compare(dtype, cols):
    shifted = cols % (16 // dtype.itemsize) != 0
    class_name = "INT8QuantPerBlockShiftedFwdKernel" if shifted else "INT8QuantPerBlockFwdKernel"
    get_kernel_class("quantization.int8_quant_per_block", class_name)
    from tileops.kernels.quantization.int8_quant_per_block import _int8_quant_per_block_kernel, _int8_quant_per_block_shifted_kernel

    torch.manual_seed(178)
    x = torch.randn(3, cols).to(dtype)
    x[0].zero_()
    before = x.view(torch.uint8).clone()
    codes = torch.full(x.shape, -128, dtype=torch.int8)
    scales = torch.full((3, (cols + 127) // 128), float("nan"))
    builder = _int8_quant_per_block_shifted_kernel if shifted else _int8_quant_per_block_kernel
    cfg = dict(threads=32, lanes=4)
    if not shifted:
        cfg["pack"] = 1
    kernel = compile_tileops_jit(builder(3, cols, str(dtype).split(".")[-1]), cfg)
    try:
        kernel(x.reshape(-1), codes.reshape(-1), scales.reshape(-1))
    finally:
        kernel.close()
    _check_quantized(x, codes, scales, block_size=128)
    torch.testing.assert_close(x.view(torch.uint8), before, atol=0, rtol=0)
