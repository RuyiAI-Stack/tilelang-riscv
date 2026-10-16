from __future__ import annotations

import torch

from ._harness import compile_tileops_jit, get_kernel_class


def test_bmm_fp8_float32_runtime_compare() -> None:
    module_cls = get_kernel_class("bmm", "BmmFP8Kernel")
    # The public BmmFP8Kernel constructor checks for CUDA SM90 before selecting
    # a kernel. Compile the underlying TileLang factory directly so the RISC-V
    # backend tests the implementation rather than the CUDA dispatch guard.
    del module_cls
    module = __import__("tileops.kernels.gemm.bmm", fromlist=["_bmm_fp8_kernel"])
    batch, m, n, k = 1, 4, 4, 32
    jit_kernel = module._bmm_fp8_kernel(batch, m, n, k, "float8_e4m3fn", "float32")
    kernel = compile_tileops_jit(
        jit_kernel,
        {"block_m": 4, "block_n": 4, "block_k": 32, "num_stages": 1, "threads": 128},
        out_idx=[-1],
    )

    a_fp32 = torch.linspace(-0.5, 0.5, batch * m * k, dtype=torch.float32).reshape(batch, m, k)
    b_fp32 = torch.linspace(-0.25, 0.25, batch * n * k, dtype=torch.float32).reshape(batch, n, k)
    a = a_fp32.to(torch.float8_e4m3fn)
    b = b_fp32.to(torch.float8_e4m3fn)
    scale_a = torch.tensor([1.0], dtype=torch.float32)
    scale_b = torch.tensor([1.0], dtype=torch.float32)

    actual = kernel(a.contiguous(), b.contiguous(), scale_a, scale_b)
    expected = torch.bmm(a.float(), b.float().transpose(1, 2))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_gemm_fp8_per_tensor_float32_runtime_compare() -> None:
    module_cls = get_kernel_class("gemm", "GemmFP8TensorScaleKernel")
    del module_cls
    module = __import__("tileops.kernels.gemm.dense", fromlist=["_gemm_fp8_kernel"])
    m, n, k = 4, 4, 32
    jit_kernel = module._gemm_fp8_kernel(m, n, k, "float8_e4m3fn", "float32", block_scaled=False)
    kernel = compile_tileops_jit(
        jit_kernel,
        {"block_m": 4, "block_n": 4, "block_k": 32, "num_stages": 1, "threads": 128},
        out_idx=[-1],
    )

    a_fp32 = torch.linspace(-0.5, 0.5, m * k, dtype=torch.float32).reshape(m, k)
    b_fp32 = torch.linspace(-0.25, 0.25, n * k, dtype=torch.float32).reshape(n, k)
    a = a_fp32.to(torch.float8_e4m3fn)
    b = b_fp32.to(torch.float8_e4m3fn)
    scale_a = torch.tensor([[1.0]], dtype=torch.float32)
    scale_b = torch.tensor([[1.0]], dtype=torch.float32)

    actual = kernel(a.contiguous(), b.contiguous(), scale_a, scale_b)
    expected = a.float() @ b.float().T
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_gemm_fp8_block_scaled_float32_runtime_compare() -> None:
    module_cls = get_kernel_class("gemm", "GemmFP8BlockScaleKernel")
    del module_cls
    module = __import__("tileops.kernels.gemm.dense", fromlist=["_gemm_fp8_kernel"])
    m, n, k = 8, 8, 128
    jit_kernel = module._gemm_fp8_kernel(m, n, k, "float8_e4m3fn", "float32", block_scaled=True)
    kernel = compile_tileops_jit(
        jit_kernel,
        {"block_m": 8, "block_n": 8, "block_k": 128, "num_stages": 1, "threads": 128},
        out_idx=[-1],
    )

    a_fp32 = torch.linspace(-0.5, 0.5, m * k, dtype=torch.float32).reshape(m, k)
    b_fp32 = torch.linspace(-0.25, 0.25, n * k, dtype=torch.float32).reshape(n, k)
    a = a_fp32.to(torch.float8_e4m3fn)
    b = b_fp32.to(torch.float8_e4m3fn)
    scale_a = torch.ones((m, 1), dtype=torch.float32)
    scale_b = torch.ones((n, 1), dtype=torch.float32)

    actual = kernel(a.contiguous(), b.contiguous(), scale_a, scale_b)
    expected = a.float() @ b.float().T
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_gqa_paged_fp8_kv_cache_runtime_compare() -> None:
    get_kernel_class("attention.gqa_fwd", "GQAPrefillPagedWithFP8KVCacheFwdKernel")
    module = __import__("tileops.kernels.attention.gqa.prefill_paged_kv_append", fromlist=["_unused"])
    jit_kernel = module._gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel(1, 2, 1, 2, 4, 2, 2, 16, False, 16**-0.5, 0.0, "float16")
    kernel = compile_tileops_jit(
        jit_kernel,
        {"block_m": 2, "block_n": 2, "num_stages": 1, "threads": 128},
    )
    q = torch.ones((2, 2, 16), dtype=torch.float16)
    k_new = torch.ones((2, 1, 16), dtype=torch.float16)
    v_new = torch.full((2, 1, 16), 2.0, dtype=torch.float16)
    k_pages = torch.ones((4, 1, 16), dtype=torch.float32).to(torch.float8_e4m3fn)
    v_pages = torch.ones((4, 1, 16), dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.ones((1,), dtype=torch.float32)
    cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
    cache_seqlens = torch.tensor([2], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    actual = kernel(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        scales,
        scales,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
        2,
    )
    torch.testing.assert_close(actual, torch.full_like(actual, 1.5), rtol=2e-2, atol=2e-2)
