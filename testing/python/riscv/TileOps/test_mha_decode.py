from __future__ import annotations

import torch

from ._harness import compile_tileops_jit


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, real_seqlen_kv: int):
    q_f = q.float()
    k_f = k[:, :real_seqlen_kv].float()
    v_f = v[:, :real_seqlen_kv].float()
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q_f, k_f) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v_f)


def test_mha_decode_no_split_float32_runtime_compare():
    batch, heads, seqlen_q, seqlen_kv, dim = 1, 2, 1, 8, 16
    from tileops.kernels.attention.gqa.decode import gqa_decode_no_split_kernel

    factory = gqa_decode_no_split_kernel(batch, heads, heads, dim, None, 0.0, "float32")
    kernel = compile_tileops_jit(factory, dict(block_H=1, block_N=8, num_stages=1, threads=128))

    q = torch.linspace(
        -0.5,
        0.5,
        batch * seqlen_q * heads * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_q, heads, dim)
    k = torch.linspace(
        -0.4,
        0.4,
        batch * seqlen_kv * heads * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, heads, dim)
    v = torch.linspace(
        -0.3,
        0.3,
        batch * seqlen_kv * heads * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, heads, dim)

    actual = kernel(q.reshape(batch, heads, dim), k, v).reshape_as(q)
    expected = _reference(q, k, v, seqlen_kv)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
