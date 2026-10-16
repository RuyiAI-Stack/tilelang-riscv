from __future__ import annotations

import torch

from ._harness import compile_tileops_jit, get_kernel_class


def test_fp8_lightning_indexer_float32_runtime_compare() -> None:
    kernel_cls = get_kernel_class("fp8_lightning_indexer", "FP8LightningIndexerKernel")
    del kernel_cls
    module = __import__(
        "tileops.kernels.attention.fp8_lightning_indexer",
        fromlist=["_fp8_lightning_indexer_kernel"],
    )
    batch, seq_len, heads, index_dim, seq_len_kv, kv_group = 1, 2, 1, 16, 2, 1
    jit_kernel = module._fp8_lightning_indexer_kernel(batch, seq_len, heads, index_dim, seq_len_kv, kv_group, 4)
    kernel = compile_tileops_jit(
        jit_kernel,
        {"block_N": 2, "num_stages": 0, "threads": 128, "block_Q": 2},
        out_idx=[],
    )

    q_f = torch.linspace(-0.25, 0.25, batch * seq_len * heads * index_dim, dtype=torch.float32)
    k_f = torch.linspace(-0.2, 0.2, batch * seq_len_kv * kv_group * index_dim, dtype=torch.float32)
    q = q_f.reshape(batch, seq_len * heads, index_dim).to(torch.float8_e4m3fn)
    k = k_f.reshape(batch, seq_len_kv, kv_group, index_dim).to(torch.float8_e4m3fn)
    k_scale = torch.ones((batch, seq_len_kv, kv_group), dtype=torch.float32)
    logits = torch.zeros((batch, seq_len, seq_len_kv, kv_group), dtype=torch.float32)
    weights = torch.ones((seq_len, heads), dtype=torch.float32)
    cu_start = torch.zeros(seq_len, dtype=torch.int32)
    cu_end = torch.full((seq_len,), seq_len_kv, dtype=torch.int32)

    kernel(
        q.contiguous(),
        k.contiguous(),
        k_scale.contiguous(),
        logits,
        weights.contiguous(),
        cu_start.contiguous(),
        cu_end.contiguous(),
    )
    expected = torch.einsum(
        "t d,s d->t s",
        q.float().reshape(seq_len, index_dim),
        k.float().reshape(seq_len_kv, index_dim),
    ).clamp_min(0)
    torch.testing.assert_close(logits.reshape(seq_len, seq_len_kv), expected, rtol=2e-2, atol=2e-2)
