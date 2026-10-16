from __future__ import annotations

import torch

from ._harness import compile_tileops_jit


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, real_seqlen_kv: int):
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,khd->bhqk", q.float(), k[:real_seqlen_kv].float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,khd->bqhd", probs, v[:real_seqlen_kv].float())


def test_mha_decode_paged_no_split_float32_runtime_compare():
    batch, heads, seq_q, seq_kv, dim, page_size = 1, 2, 1, 8, 16, 4
    from tileops.kernels.attention.gqa.paged_decode import gqa_decode_no_split_paged_kernel

    factory = gqa_decode_no_split_paged_kernel(
        batch, heads, heads, seq_q, seq_kv, dim, page_size, seq_kv // page_size, False, None, 0.0, "float32"
    )
    kernel = compile_tileops_jit(factory, dict(block_M=2, block_N=4, num_stages=1, threads=128))

    q = torch.linspace(-0.5, 0.5, batch * seq_q * heads * dim, dtype=torch.float32).reshape(batch, seq_q, heads, dim)
    k = torch.linspace(-0.4, 0.4, seq_kv * heads * dim, dtype=torch.float32).reshape(seq_kv, heads, dim)
    v = torch.linspace(-0.3, 0.3, seq_kv * heads * dim, dtype=torch.float32).reshape(seq_kv, heads, dim)
    real_seqlen_kv = torch.full((batch,), seq_kv, dtype=torch.int32)
    block_table = torch.arange(seq_kv // page_size, dtype=torch.int32).unsqueeze(0)

    actual = kernel(q.reshape(batch, 1, heads, dim), k, v, real_seqlen_kv, block_table).reshape_as(q)
    expected = _reference(q, k, v, seq_kv)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
