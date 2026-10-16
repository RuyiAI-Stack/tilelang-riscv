from __future__ import annotations

import torch

from ._harness import compile_tileops_jit


def test_gqa_sliding_window_varlen_runtime_compare():
    batch, heads, heads_kv, dim = 2, 4, 2, 16
    left, right = 2, 1
    is_causal = False

    module = __import__("tileops.kernels.attention.gqa.sliding_window_varlen", fromlist=["_unused"])
    jit_kernel = module._gqa_sw_fwd_varlen_wgmma_pipelined_kernel(
        batch,
        heads,
        heads_kv,
        dim,
        is_causal,
        left,
        right,
        "float32",
        "float",
    )
    kernel = compile_tileops_jit(jit_kernel, {"block_m": 8, "block_n": 8, "num_stages": 1, "threads": 128})
    torch.manual_seed(123)
    q_lens, kv_lens = (3, 5), (5, 4)
    cu_q = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_kv = torch.tensor([0, 5, 9], dtype=torch.int32)
    q = torch.randn(sum(q_lens), heads, dim) * 0.2
    k = torch.randn(sum(kv_lens), heads_kv, dim) * 0.2
    v = torch.randn_like(k)
    originals = [x.clone() for x in (q, k, v, cu_q, cu_kv)]
    try:
        actual = kernel(q, k, v, cu_q, cu_kv)
    finally:
        kernel.close()

    expected = torch.empty_like(q)
    for batch_idx, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        q_start, q_end = cu_q[batch_idx : batch_idx + 2].tolist()
        kv_start, kv_end = cu_kv[batch_idx : batch_idx + 2].tolist()
        queries = q[q_start:q_end].transpose(0, 1)
        keys = k[kv_start:kv_end].repeat_interleave(heads // heads_kv, dim=1).transpose(0, 1)
        values = v[kv_start:kv_end].repeat_interleave(heads // heads_kv, dim=1).transpose(0, 1)
        scores = queries @ keys.transpose(-1, -2) / dim**0.5
        q_pos = torch.arange(q_len)[:, None]
        k_pos = torch.arange(kv_len)[None, :]
        aligned_q_pos = q_pos + kv_len - q_len
        masked = (k_pos < aligned_q_pos - left) | (k_pos > aligned_q_pos + right)
        scores.masked_fill_(masked.unsqueeze(0), -float("inf"))
        expected[q_start:q_end] = (scores.softmax(-1) @ values).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    for value, original in zip((q, k, v, cu_q, cu_kv), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
