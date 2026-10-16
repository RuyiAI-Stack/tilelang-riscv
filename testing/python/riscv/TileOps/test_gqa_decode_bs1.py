from __future__ import annotations

import torch

from ._harness import compile_tileops_jit, get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_seqlen_kv: int,
    groups: int,
):
    q_f = q.float()
    k_f = k[:, :real_seqlen_kv].float()
    v_f = v[:, :real_seqlen_kv].float()
    scale = q.shape[-1] ** -0.5
    batch, heads, dim = q.shape
    heads_per_group = heads // groups
    out = torch.empty(batch, heads, dim, dtype=torch.float32)
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum("bd,bkd->bk", q_f[:, head, :], k_f[:, :, group, :]) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, head, :] = torch.einsum("bk,bkd->bd", probs, v_f[:, :, group, :])
    return out


def test_gqa_decode_bs1_short_context_float32_runtime_compare():
    batch, heads, groups, seqlen_kv, dim = 1, 64, 1, 128, 16
    kernel_cls = get_kernel_class("attention.gqa_decode_bs1", "GQADecodeBs1Kernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=groups,
        seq_len_kv=seqlen_kv,
        dim=dim,
        dtype=torch.float32,
    )
    del tileops_kernel
    module = __import__("tileops.kernels.attention.gqa.decode", fromlist=["_unused"])
    jit_kernel = module.gqa_decode_no_split_kernel(batch, heads, groups, dim, dim**-0.5, 0.0, "float32")
    kernel = compile_tileops_jit(
        jit_kernel,
        {"block_H": 64, "block_N": 128, "num_stages": 2, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, batch * heads * dim, dtype=torch.float32).reshape(batch, heads, dim)
    k = torch.linspace(
        -0.4,
        0.4,
        batch * seqlen_kv * groups * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, groups, dim)
    v = torch.linspace(
        -0.3,
        0.3,
        batch * seqlen_kv * groups * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, groups, dim)

    actual = kernel(q, k, v)
    expected = _reference(q, k, v, seqlen_kv, groups)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
