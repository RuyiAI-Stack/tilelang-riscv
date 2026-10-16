from __future__ import annotations

import torch

from ._harness import get_kernel_class


def test_gqa_bwd_preprocess_float32_runtime_compare():
    batch, heads, seq_len, dim = 1, 1, 256, 256
    kernel_cls = get_kernel_class("attention.gqa_bwd", "GQABwdPreprocessKernel")
    tileops_kernel = kernel_cls(batch, heads, seq_len, dim, torch.float32)

    o = torch.linspace(-0.5, 0.5, batch * seq_len * heads * dim, dtype=torch.float32).reshape(batch, seq_len, heads, dim)
    do = torch.linspace(0.25, -0.25, batch * seq_len * heads * dim, dtype=torch.float32).reshape(batch, seq_len, heads, dim)

    actual, dq = tileops_kernel(o, do)
    torch.testing.assert_close(dq, torch.zeros_like(dq), atol=0, rtol=0)
    expected = (o * do).sum(dim=-1).permute(0, 2, 1).contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
