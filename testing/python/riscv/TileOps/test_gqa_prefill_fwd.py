from __future__ import annotations

import torch

from ._harness import compile_tileops_jit


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads_kv: int):
    batch, seq_q, heads, dim = q.shape
    heads_per_group = heads // heads_kv
    out = torch.empty(batch, seq_q, heads, dim, dtype=torch.float32)
    scale = dim**-0.5
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum("bqd,bkd->bqk", q[:, :, head, :].float(), k[:, :, group, :].float()) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, :, head, :] = torch.einsum("bqk,bkd->bqd", probs, v[:, :, group, :].float())
    return out


def test_gqa_prefill_fwd_float32_runtime_compare():
    batch, heads, heads_kv, seq_q, seq_kv, dim = 1, 4, 2, 6, 8, 16
    from tileops.kernels.attention.gqa.prefill_varlen import _gqa_prefill_varlen_fwd_kernel

    factory = _gqa_prefill_varlen_fwd_kernel(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        is_causal=False,
        dtype="float32",
    )
    tileops_kernel = compile_tileops_jit(factory, {"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128})

    q = torch.linspace(-0.5, 0.5, batch * seq_q * heads * dim, dtype=torch.float32).reshape(batch, seq_q, heads, dim)
    k = torch.linspace(-0.4, 0.4, batch * seq_kv * heads_kv * dim, dtype=torch.float32).reshape(batch, seq_kv, heads_kv, dim)
    v = torch.linspace(-0.3, 0.3, batch * seq_kv * heads_kv * dim, dtype=torch.float32).reshape(batch, seq_kv, heads_kv, dim)

    cu_seqlens_q = torch.tensor([0, seq_q], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, seq_kv], dtype=torch.int32)
    try:
        actual = tileops_kernel(
            q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), cu_seqlens_q, cu_seqlens_kv, torch.empty(heads)
        ).reshape_as(q)
        expected = _reference(q, k, v, heads_kv)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    finally:
        tileops_kernel.close()
