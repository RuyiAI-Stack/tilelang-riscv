from __future__ import annotations

import torch

from ._harness import get_kernel_class


def test_moe_unpermute_float32_runtime_compare():
    num_tokens, top_k, hidden_size = 2, 2, 8
    padded_batch_sum = num_tokens * top_k
    kernel_cls = get_kernel_class("moe.unpermute", "MoEUnpermuteKernel")
    tileops_kernel = kernel_cls(
        num_tokens=num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        materialized_rows=padded_batch_sum,
        dtype=torch.float32,
    )

    # The public TileOps wrapper asserts CUDA tensors.  Validate the underlying
    # TileLang kernel directly on SG2044's RISC-V host adapter.
    kernel = tileops_kernel._unpermute_fn()
    adapter = getattr(kernel, "adapter", None)
    assert type(adapter).__name__ == "RiscvKernelAdapter"

    mm2_pad = torch.linspace(
        -0.5,
        0.5,
        padded_batch_sum * hidden_size,
        dtype=torch.float32,
    ).reshape(padded_batch_sum, hidden_size)
    fwd_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
    actual = torch.empty(num_tokens, hidden_size, dtype=torch.float32)

    kernel(mm2_pad, fwd_idx, topk_weights, actual)

    expected = torch.stack(
        [
            mm2_pad[0] * topk_weights[0, 0] + mm2_pad[1] * topk_weights[0, 1],
            mm2_pad[2] * topk_weights[1, 0] + mm2_pad[3] * topk_weights[1, 1],
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
