from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_da_cumsum_float32_runtime_compare():
    batch, num_chunks, chunk_len, n_heads = 1, 2, 4, 3
    seq_len = num_chunks * chunk_len
    kernel_cls = get_kernel_class("mamba.da_cumsum", "SSDChunkCumsumFwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        seq_len,
        torch.float32,
        dt_softplus=True,
        has_dt_bias=True,
        dt_min=0.05,
        dt_max=1.5,
        config={
            "block_h": 2,
            "threads": 8,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    dt = torch.linspace(
        -1.0,
        1.0,
        batch * seq_len * n_heads,
        dtype=torch.float32,
    ).reshape(batch, seq_len, n_heads)
    a = torch.linspace(-0.5, -0.25, n_heads, dtype=torch.float32)
    dt_bias = torch.linspace(0.1, 0.3, n_heads, dtype=torch.float32)

    actual_dt, actual_da = kernel(dt.contiguous(), a.contiguous(), dt_bias.contiguous())

    transformed = torch.nn.functional.softplus(dt + dt_bias.view(1, 1, n_heads))
    transformed = transformed.clamp(0.05, 1.5)
    expected_dt = transformed.reshape(batch, num_chunks, chunk_len, n_heads).permute(0, 3, 1, 2)
    expected_da = torch.cumsum(expected_dt * a.view(1, n_heads, 1, 1), dim=3)

    torch.testing.assert_close(actual_dt, expected_dt.contiguous(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_da, expected_da.contiguous(), rtol=1e-5, atol=1e-5)
