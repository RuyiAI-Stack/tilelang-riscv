"""Numerical coverage for the original unaligned FP8 block quantizer."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


@pytest.mark.parametrize("rows,cols", [(3, 17), (129, 131)])
@pytest.mark.parametrize("input_kind", ["random", "zeros", "tiny"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fp8_quant_per_block_unaligned_runtime_compare(rows, cols, input_kind, dtype):
    get_kernel_class("quantization.fp8_quant_per_block", "FP8QuantPerBlockUnalignedFwdKernel")
    from tileops.kernels.quantization.fp8_quant_per_block import _fp8_quant_per_block_kernel

    torch.manual_seed(133)
    x = torch.randn(rows, cols, dtype=torch.float32) * 0.3
    if input_kind == "zeros":
        x.zero_()
    elif input_kind == "tiny":
        x *= 1e-5 if dtype == torch.float16 else 1e-35
    x = x.to(dtype)
    before = x.clone()
    codes = torch.empty(rows, cols, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows + 127) // 128, (cols + 127) // 128, dtype=torch.float32)
    factory = _fp8_quant_per_block_kernel(rows, cols, str(dtype).split(".")[-1], False)
    kernel = compile_tileops_jit(factory, {"threads": 128, "min_blocks": None, "evict_first": False})
    try:
        kernel(x.reshape(-1), codes.reshape(-1), scales.reshape(-1))
    finally:
        kernel.close()

    expected = torch.empty_like(codes)
    expected_scales = torch.empty_like(scales)
    for row in range(expected_scales.shape[0]):
        for col in range(expected_scales.shape[1]):
            block = x[row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128].float()
            # The upstream contract uses no epsilon floor and scale=1 for a
            # zero tile. Its CUDA/PyTorch scalar division multiplies by the
            # FP32 reciprocal; host tensor division can differ by one ULP.
            amax = block.abs().amax()
            inv_max = torch.tensor(1.0 / 448.0, dtype=torch.float32)
            scale = torch.where(amax > 0, amax * inv_max, torch.ones_like(amax))
            expected_scales[row, col] = scale
            expected[row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128] = (block / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(codes.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
