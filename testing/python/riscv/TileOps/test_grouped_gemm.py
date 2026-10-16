from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor, batch_offsets: torch.Tensor):
    batch_sum, _ = A.shape
    _, N, _ = B.shape
    out = torch.empty(batch_sum, N, dtype=torch.float32)
    for i in range(batch_offsets.numel()):
        start = int(batch_offsets[i])
        end = start + int(batch_sizes[i])
        out[start:end] = A[start:end].float() @ B[i].float().transpose(0, 1)
    return out


def test_grouped_gemm_nt_float32_runtime_compare():
    batch_sum, batch_count, N, K = 2, 2, 4, 4
    kernel_cls = get_kernel_class("grouped_gemm.grouped_gemm", "GroupedGemmKernel")
    tileops_kernel = kernel_cls(
        batch_sum=batch_sum,
        batch_count=batch_count,
        N=N,
        K=K,
        dtype=torch.float32,
        transpose_a=False,
        transpose_b=True,
        config={"block_m": 1, "block_n": 4, "block_k": 4, "num_stages": 1, "threads": 32},
    )

    A = torch.linspace(-0.5, 0.5, batch_sum * K, dtype=torch.float32).reshape(batch_sum, K)
    B = torch.linspace(-0.4, 0.4, batch_count * N * K, dtype=torch.float32).reshape(batch_count, N, K)
    batch_sizes = torch.tensor([1, 1], dtype=torch.int32)
    batch_offsets = torch.tensor([0, 1], dtype=torch.int32)

    actual = tileops_kernel(A, B, batch_sizes, batch_offsets)
    expected = _reference(A, B, batch_sizes, batch_offsets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
