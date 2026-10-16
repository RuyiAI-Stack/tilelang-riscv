from __future__ import annotations

import torch

from ._harness import compile_tileops_jit


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    batch: int,
    heads_kv: int,
):
    total_q, heads, dim = q.shape
    heads_per_group = heads // heads_kv
    out = torch.empty(total_q, heads, dim, dtype=torch.float32)
    scale = dim**-0.5
    for b in range(batch):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        kv_start, kv_end = cu_seqlens_kv[b].item(), cu_seqlens_kv[b + 1].item()
        q_b = q[q_start:q_end].unsqueeze(0)
        k_b = k[kv_start:kv_end].unsqueeze(0)
        v_b = v[kv_start:kv_end].unsqueeze(0)
        for head in range(heads):
            group = head // heads_per_group
            scores = torch.einsum("bqd,bkd->bqk", q_b[:, :, head, :].float(), k_b[:, :, group, :].float()) * scale
            probs = torch.softmax(scores, dim=-1)
            out[q_start:q_end, head, :] = torch.einsum("bqk,bkd->bqd", probs, v_b[:, :, group, :].float())[0]
    return out


def test_gqa_prefill_varlen_fwd_float32_runtime_compare():
    batch, heads, heads_kv, dim = 2, 4, 2, 16
    cu_seqlens_q = torch.tensor([0, 3, 5], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 4, 7], dtype=torch.int32)
    total_q = cu_seqlens_q[-1].item()
    total_kv = cu_seqlens_kv[-1].item()

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

    q = torch.linspace(-0.5, 0.5, total_q * heads * dim, dtype=torch.float32).reshape(total_q, heads, dim)
    k = torch.linspace(-0.4, 0.4, total_kv * heads_kv * dim, dtype=torch.float32).reshape(total_kv, heads_kv, dim)
    v = torch.linspace(-0.3, 0.3, total_kv * heads_kv * dim, dtype=torch.float32).reshape(total_kv, heads_kv, dim)

    try:
        actual = tileops_kernel(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            torch.empty(heads),
        )
        expected = _reference(q, k, v, cu_seqlens_q, cu_seqlens_kv, batch, heads_kv)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    finally:
        tileops_kernel.close()
