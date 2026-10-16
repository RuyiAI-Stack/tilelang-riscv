from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    block_size: int,
    groups: int,
    scale: float,
):
    c_seq_len, heads, dim_k = q.shape
    head_kv = heads // groups
    out = torch.zeros_like(q)
    lse = torch.zeros(c_seq_len, heads, dtype=torch.float32)
    for t in range(c_seq_len):
        nc = (t + 1) // block_size
        if nc == 0:
            continue
        for kvh in range(head_kv):
            for g in range(groups):
                h = kvh * groups + g
                scores = (q[t, h].float() * k_cmp[:nc, kvh].float()).sum(dim=-1) * scale
                probs = torch.softmax(scores, dim=0)
                out[t, h] = (probs[:, None] * v_cmp[:nc, kvh].float()).sum(dim=0)
                lse[t, h] = torch.logsumexp(scores, dim=0)
    return out, lse


def test_nsa_cmp_fwd_varlen_float32_runtime_compare():
    seq_num, c_seq_len, heads = 1, 4, 2
    dim_k = dim_v = 16
    chunk_num, groups = 2, 2
    block_size = bc = bs = 2
    scale = dim_k**-0.5
    kv_group = heads // groups

    kernel_cls = get_kernel_class(
        "attention.deepseek_nsa_cmp_fwd",
        "NSACompressedFwdVarlenKernel",
    )
    tileops_kernel = kernel_cls(
        seq_num,
        c_seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_num,
        groups,
        scale,
        bc,
        bs,
        torch.float32,
        torch.float32,
        config={"threads": 32},
    )

    q = torch.linspace(-0.5, 0.5, c_seq_len * heads * dim_k, dtype=torch.float32).reshape(c_seq_len, heads, dim_k)
    k_cmp = torch.linspace(
        -0.4,
        0.4,
        chunk_num * kv_group * dim_k,
        dtype=torch.float32,
    ).reshape(chunk_num, kv_group, dim_k)
    v_cmp = torch.linspace(
        -0.3,
        0.3,
        chunk_num * kv_group * dim_v,
        dtype=torch.float32,
    ).reshape(chunk_num, kv_group, dim_v)
    offsets = torch.tensor([0, c_seq_len], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, chunk_num], dtype=torch.int32)
    token_indices = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.int32)

    actual_out, actual_lse = tileops_kernel(
        q,
        k_cmp,
        v_cmp,
        offsets,
        chunk_offsets,
        token_indices,
    )
    expected_out, expected_lse = _reference(q, k_cmp, v_cmp, block_size, groups, scale)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
