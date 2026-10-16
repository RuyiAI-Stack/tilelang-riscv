from __future__ import annotations

import torch
import pytest

from ._harness import compile_tileops_jit


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


def test_gqa_decode_no_split_float32_runtime_compare():
    batch, heads, groups, seqlen_kv, dim = 1, 2, 1, 8, 16
    from tileops.kernels.attention.gqa.decode import gqa_decode_no_split_kernel

    factory = gqa_decode_no_split_kernel(batch, heads, groups, dim, None, 0.0, "float32")
    kernel = compile_tileops_jit(factory, dict(block_H=2, block_N=8, num_stages=1, threads=128))

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen_kv", [63, 65])
def test_gqa_decode_d256_runtime_compare(dtype, seqlen_kv):
    """Exercise the original D=256 dense-decode kernel across a KV tile tail."""
    from tileops.kernels.attention.gqa.decode import (
        GQADecodeKernel,
        gqa_decode_no_split_kernel,
    )

    assert GQADecodeKernel._limit_refusal(256, seqlen_kv, 80) is None
    batch, heads, groups, dim = 1, 2, 1, 256
    torch.manual_seed(2026)
    q = (torch.randn(batch, heads, dim) * 0.2).to(dtype)
    k = (torch.randn(batch, seqlen_kv, groups, dim) * 0.2).to(dtype)
    v = (torch.randn_like(k.float()) * 0.2).to(dtype)
    originals = [tensor.clone() for tensor in (q, k, v)]

    factory = gqa_decode_no_split_kernel(batch, heads, groups, dim, None, 0.0, str(dtype).split(".")[-1])
    kernel = compile_tileops_jit(factory, dict(block_H=2, block_N=64, num_stages=1, threads=128))
    try:
        actual = kernel(q, k, v)
    finally:
        kernel.close()

    expected = _reference(q, k, v, seqlen_kv, groups).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    for tensor, original in zip((q, k, v), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
