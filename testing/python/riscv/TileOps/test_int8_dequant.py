"""Original upstream INT8 dequantization kernels with independent CPU references."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [5, 19])
def test_int8_dequant_per_channel_runtime_compare(dtype, k):
    get_kernel_class("quantization.int8_dequant", "INT8DequantPerChannelFwdKernel")
    from tileops.kernels.quantization.int8_dequant import _int8_dequant_per_channel_kernel

    m = 5
    torch.manual_seed(125)
    codes = torch.randint(-128, 128, (m, k), dtype=torch.int8)
    scales = torch.rand(m, dtype=torch.float32) * 0.04 + 0.01
    factory = _int8_dequant_per_channel_kernel(m, k, str(dtype).split(".")[-1], 16 // dtype.itemsize)
    kernel = compile_tileops_jit(factory, {"threads": 32, "steps": 1})
    try:
        actual = kernel(codes.reshape(-1), scales).reshape(m, k)
    finally:
        kernel.close()
    expected = (codes.float() * scales[:, None]).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [5, 131])
def test_int8_dequant_per_block_runtime_compare(dtype, k):
    get_kernel_class("quantization.int8_dequant", "INT8DequantPerBlockFwdKernel")
    from tileops.kernels.quantization.int8_dequant import _int8_dequant_per_block_kernel

    m = 5
    torch.manual_seed(126)
    codes = torch.randint(-128, 128, (m, k), dtype=torch.int8)
    scales = torch.rand(m, (k + 127) // 128, dtype=torch.float32) * 0.04 + 0.01
    factory = _int8_dequant_per_block_kernel(
        m,
        k,
        str(dtype).split(".")[-1],
        16 // dtype.itemsize,
        False,
        2048,
        4,
    )
    kernel = compile_tileops_jit(factory, {"threads": 32, "steps": 1})
    try:
        actual = kernel(codes.reshape(-1), scales.reshape(-1)).reshape(m, k)
    finally:
        kernel.close()
    expected = (codes.float() * scales[:, torch.arange(k) // 128]).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
