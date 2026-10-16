from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_topk_selector_float32_runtime_compare():
    batch, seq_len, seq_len_kv, kv_group, topk = 1, 2, 8, 1, 2
    kernel_cls = get_kernel_class("topk_selector", "TopKSelectKernel")
    tileops_kernel = kernel_cls(
        batch,
        seq_len,
        seq_len_kv,
        kv_group,
        topk,
        torch.float32,
        torch.int32,
        config={
            "RADIX": 256,
            "BLOCK_SIZE": 256,
            "SMEM_INPUT_SIZE": 16,
            "block_m": 32,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    index_score = torch.tensor(
        [
            [
                [[0.1], [0.9], [0.2], [0.8], [0.3], [0.7], [0.4], [0.6]],
                [[0.5], [0.0], [0.6], [0.1], [0.7], [0.2], [0.8], [0.3]],
            ]
        ],
        dtype=torch.float32,
    )
    starts = torch.zeros(batch, seq_len, dtype=torch.int32)
    ends = torch.full((batch, seq_len), seq_len_kv, dtype=torch.int32)

    actual = kernel(index_score, starts, ends)
    expected = torch.topk(index_score, topk, dim=2)[1].permute(0, 1, 3, 2).to(torch.int32)
    torch.testing.assert_close(actual, expected)
