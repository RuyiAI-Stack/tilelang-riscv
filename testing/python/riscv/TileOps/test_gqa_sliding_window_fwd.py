from __future__ import annotations

import torch
import pytest
import math

from ._harness import compile_tileops_kernel, get_kernel_class


@pytest.mark.parametrize("is_causal", [False, True])
def test_gqa_dense_sliding_window_runtime_compare(is_causal):
    batch, heads, heads_kv, seq_len, dim = 1, 4, 2, 9, 16
    left, right = 2, 1
    kernel_cls = get_kernel_class("attention.gqa_dense", "GQADenseSlidingWindowKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq_len,
        dim=dim,
        is_causal=is_causal,
        window_size_left=left,
        window_size_right=right,
        dtype=torch.float32,
        config={"block_m": 8, "block_n": 8, "num_stages": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    torch.manual_seed(33)
    q = torch.randn(batch, seq_len, heads, dim) * 0.2
    k = torch.randn(batch, seq_len, heads_kv, dim) * 0.2
    v = torch.randn_like(k)
    try:
        actual, lse = kernel(q, k, v)
    finally:
        kernel.close()
    scores = q.transpose(1, 2) @ k.repeat_interleave(heads // heads_kv, 2).permute(0, 2, 3, 1)
    scores *= dim**-0.5
    positions = torch.arange(seq_len)
    distance = positions[None, :] - positions[:, None]
    allowed = (distance >= -left) & (distance <= (0 if is_causal else right))
    scores.masked_fill_(~allowed, -float("inf"))
    expected = (scores.softmax(-1) @ v.repeat_interleave(heads // heads_kv, 2).transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(lse, scores.logsumexp(-1) / math.log(2), atol=1e-5, rtol=1e-4)
