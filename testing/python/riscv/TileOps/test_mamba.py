from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def _da_cumsum_reference(
    dt: torch.Tensor,
    a: torch.Tensor,
    *,
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt_out = dt.reshape(batch, num_chunks, chunk_len, n_heads).permute(0, 3, 1, 2).contiguous()
    da = dt_out * a.reshape(1, n_heads, 1, 1)
    return dt_out, da.cumsum(dim=-1)


def test_da_cumsum_float32_runtime_compare():
    batch, num_chunks, chunk_len, n_heads = 1, 2, 4, 2
    seq_len = num_chunks * chunk_len
    kernel_cls = get_kernel_class("mamba.da_cumsum", "SSDChunkCumsumFwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        seq_len,
        dtype=torch.float32,
        dt_softplus=False,
        has_dt_bias=False,
        config={"block_h": 2, "threads": 8},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    dt = torch.arange(batch * seq_len * n_heads, dtype=torch.float32).reshape(batch, seq_len, n_heads)
    dt = dt / 10.0 + 1.0
    a = -(torch.arange(n_heads, dtype=torch.float32) + 1.0) / 10.0
    dt_bias = torch.zeros(n_heads, dtype=torch.float32)

    actual_dt, actual_da = kernel(dt, a, dt_bias)
    expected_dt, expected_da = _da_cumsum_reference(
        dt,
        a,
        batch=batch,
        num_chunks=num_chunks,
        chunk_len=chunk_len,
        n_heads=n_heads,
    )

    torch.testing.assert_close(actual_dt, expected_dt)
    torch.testing.assert_close(actual_da, expected_da)
