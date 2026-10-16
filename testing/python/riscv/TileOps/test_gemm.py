"""Numerical GEMM coverage, including upstream W4A16 MMA and CUDA-only rejection."""

import importlib

import pytest
import torch
from tilelang import tvm

from ._harness import compile_tileops_jit, get_kernel_class


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_basic_runtime_compare(trans_a, trans_b, dtype):
    get_kernel_class("gemm.dense", "GemmCpAsyncKernel")
    module = importlib.import_module("tileops.kernels.gemm.dense")
    m, n, k = 5, 7, 16
    torch.manual_seed(7)
    a = (torch.randn(m, k) * 0.2).to(dtype)
    b = (torch.randn(k, n) * 0.2).to(dtype)
    factory = module._gemm_basic_kernel(m, n, k, trans_a, trans_b, str(dtype).split(".")[-1])
    kernel = compile_tileops_jit(factory, dict(block_m=8, block_n=8, block_k=8, num_stages=1, threads=128))
    try:
        actual = kernel(a.T.contiguous() if trans_a else a, b.T.contiguous() if trans_b else b)
    finally:
        kernel.close()
    torch.testing.assert_close(actual, (a.float() @ b.float()).to(dtype), atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_w4a16_mma_runtime_compare(dtype):
    """Use the original non-WGMMA kernel and check its prepacked INT4 contract."""
    get_kernel_class("gemm.w4a16", "GemmW4A16MMAKernel")
    module = importlib.import_module("tileops.kernels.gemm.w4a16")
    m, n, k = 3, 8, 256
    torch.manual_seed(124)
    activation = (torch.randn(m, k) * 0.1).to(dtype)
    unsigned = torch.randint(0, 16, (n, k), dtype=torch.int32)
    scale = (torch.rand(n, k // 128) * 0.1 + 0.05).to(dtype)
    zero = torch.randint(0, 16, (n, k // 128), dtype=torch.uint8)
    # Encode the upstream W4A16RepackKernel ABI directly from logical nibbles.
    # One uint32 word holds four low and four high nibbles from a 128-K step.
    packed = torch.zeros(n, k // 8, dtype=torch.int64)
    for offset in range(k):
        byte = (offset % 128) // 2
        word = byte % 4 * 4 + byte // 16
        shift = 4 * (byte % 16 // 4) + 16 * (offset % 2)
        packed[:, offset // 128 * 16 + word] |= unsigned[:, offset].to(torch.int64) << shift
    packed = packed.to(torch.uint32)
    originals = [x.clone() for x in (activation, packed, scale, zero)]
    factory = module._gemm_w4a16_mma_kernel(m, n, k, str(dtype).split(".")[-1])
    kernel = compile_tileops_jit(factory, dict(block_m=8, block_n=8, num_stages=1, threads=32))
    try:
        actual = kernel(activation, packed, scale, zero)
    finally:
        kernel.close()
    dequantized = (
        ((unsigned.reshape(n, k // 128, 128).float() - zero.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)).reshape(n, k).to(dtype)
    )
    expected = (activation.float() @ dequantized.float().T).to(dtype)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    for value, original in zip((activation, packed, scale, zero), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_w4a16_rejects_cuda_dequant_extern(dtype):
    get_kernel_class("gemm.w4a16", "GemmW4A16Kernel")
    module = importlib.import_module("tileops.kernels.gemm.w4a16")
    m, n, k = 3, 8, 256
    with pytest.raises(tvm.error.InternalError, match="Unsupported call_extern.*tileops::w4a16_dequant_word"):
        compile_tileops_jit(
            module._gemm_w4a16_kernel(m, n, k, str(dtype).split(".")[-1]), dict(block_m=8, block_n=8, block_k=128, num_stages=1, threads=32)
        )
