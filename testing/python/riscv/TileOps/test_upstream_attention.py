"""Compile the current upstream programs, including formerly missing implementations."""

import pytest
import torch

from ._harness import compile_tileops_jit


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 2])
def test_mla_mma_runtime(dtype, splits):
    from tileops.kernels.attention.mla.decode import _mla_decode_mma_kernel

    torch.manual_seed(91)
    batch, heads, length, dim, pe = 1, 4, 19, 32, 16
    q = (torch.randn(batch, heads, dim) * 0.2).to(dtype)
    qp = (torch.randn(batch, heads, pe) * 0.2).to(dtype)
    k = (torch.randn(batch, length, 1, dim) * 0.2).to(dtype)
    kp = (torch.randn(batch, length, 1, pe) * 0.2).to(dtype)
    factory = _mla_decode_mma_kernel(batch, heads, 1, length, dim, pe, str(dtype).split(".")[-1])
    kernel = compile_tileops_jit(factory, dict(block_H=16, block_N=16, num_split=splits, num_stages=1, threads=128))
    try:
        actual = kernel(q, qp, k, kp, torch.empty(batch, heads, splits), torch.empty(batch, heads, splits, dim, dtype=dtype))
    finally:
        kernel.close()
    scores = (q.float() @ k[:, :, 0].float().transpose(-1, -2) + qp.float() @ kp[:, :, 0].float().transpose(-1, -2)) / (dim + pe) ** 0.5
    expected = (scores.softmax(-1) @ k[:, :, 0].float()).to(dtype)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


def attention_reference(q, k, v, cq, ck, causal, sinks=None):
    result = torch.empty_like(q)
    for i in range(len(cq) - 1):
        qi = q[cq[i] : cq[i + 1]].float().transpose(0, 1)
        ki = k[ck[i] : ck[i + 1]].float().repeat_interleave(q.shape[1] // k.shape[1], 1).transpose(0, 1)
        vi = v[ck[i] : ck[i + 1]].float().repeat_interleave(q.shape[1] // k.shape[1], 1).transpose(0, 1)
        scores = qi @ ki.transpose(-1, -2) / q.shape[-1] ** 0.5
        if causal:
            mask = torch.arange(ki.shape[1])[None, :] > torch.arange(qi.shape[1])[:, None] + ki.shape[1] - qi.shape[1]
            scores.masked_fill_(mask, -float("inf"))
        if sinks is None:
            probabilities = scores.softmax(-1)
        else:
            # One extra softmax column contributes only to the denominator.
            logits = torch.cat((scores.double(), sinks.double()[:, None, None].expand(-1, qi.shape[1], 1)), dim=-1)
            probabilities = logits.softmax(-1)[..., :-1].float()
            probabilities[torch.isposinf(sinks)] = 0
        result[cq[i] : cq[i + 1]] = (probabilities @ vi).transpose(0, 1).to(q.dtype)
    return result


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_upstream_runtime(causal, dtype):
    from tileops.kernels.attention.gqa.prefill_varlen import _gqa_prefill_varlen_fwd_kernel

    torch.manual_seed(92)
    q = (torch.randn(5, 4, 32) * 0.2).to(dtype)
    k = (torch.randn(9, 2, 32) * 0.2).to(dtype)
    v = torch.randn_like(k)
    cq, ck = torch.tensor([0, 3, 5], dtype=torch.int32), torch.tensor([0, 5, 9], dtype=torch.int32)
    factory = _gqa_prefill_varlen_fwd_kernel(2, 4, 2, 32, causal, dtype=str(dtype).split(".")[-1])
    kernel = compile_tileops_jit(factory, dict(block_m=16, block_n=16, num_stages=1, threads=128))
    try:
        actual = kernel(q, k, v, cq, ck, torch.empty(4, dtype=torch.float32))
    finally:
        kernel.close()
    torch.testing.assert_close(actual, attention_reference(q, k, v, cq, ck, causal), atol=4e-3, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_upstream_runtime(causal, dtype):
    from tileops.kernels.attention.gqa.paged import _gqa_paged_varlen_kernel

    torch.manual_seed(93)
    q = (torch.randn(5, 4, 32) * 0.2).to(dtype)
    kp = (torch.randn(5, 3, 2, 32) * 0.2).to(dtype)
    vp = torch.randn_like(kp)
    pages = torch.tensor([[3, 1], [4, 0]], dtype=torch.int32)
    lengths, cq = torch.tensor([5, 4], dtype=torch.int32), torch.tensor([0, 3, 5], dtype=torch.int32)
    factory = _gqa_paged_varlen_kernel(
        2,
        4,
        2,
        32,
        3,
        2,
        causal,
        -1,
        -1,
        32**-0.5,
        0.0,
        str(dtype).split(".")[-1],
        False,
        1,
        0,
        "neox",
        has_sinks=False,
        sm90=False,
        producer_warpgroup=False,
    )
    kernel = compile_tileops_jit(factory, dict(block_M=16, block_N=16, num_stages=1, threads=128))
    try:
        unused = torch.empty(1, 1, dtype=dtype)
        actual = kernel(q, kp.flatten(0, 1), vp.flatten(0, 1), lengths, pages, cq, unused, unused, torch.empty(4, dtype=torch.float32))
    finally:
        kernel.close()
    k = torch.cat([kp[pages[i].long()].flatten(0, 1)[: lengths[i]] for i in range(2)])
    v = torch.cat([vp[pages[i].long()].flatten(0, 1)[: lengths[i]] for i in range(2)])
    ck = torch.tensor([0, 5, 9], dtype=torch.int32)
    torch.testing.assert_close(actual, attention_reference(q, k, v, cq, ck, causal), atol=4e-3, rtol=2e-2)
