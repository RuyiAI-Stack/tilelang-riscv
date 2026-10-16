"""Original upstream sampling masks checked against independent row references."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


@pytest.mark.parametrize("vocab,threads", [(33, 32), (129, 64)])
def test_top_p_mask_ragged_runtime_compare(vocab, threads):
    get_kernel_class("sampling.top_p_mask", "TopPMaskFwdKernel")
    from tileops.kernels.sampling.top_p_mask import _top_p_mask_kernel

    batch = 2
    torch.manual_seed(129)
    logits = torch.randn(batch, vocab, dtype=torch.float32)
    p = torch.tensor([0.55, 0.9], dtype=torch.float32)
    before = logits.clone()
    part_top = torch.empty(batch, dtype=torch.float32)
    part_bins = torch.empty(batch * 32, dtype=torch.float32)
    masked = torch.empty_like(logits)
    factory = _top_p_mask_kernel(batch, vocab, "float32", 1, threads, 1, 7, 32)
    kernel = compile_tileops_jit(factory, {"reg_tiles": (vocab + threads - 1) // threads, "smem_tiles": 0, "pace": 1})
    try:
        kernel(logits.reshape(-1), p, part_top, part_bins, masked.reshape(-1))
    finally:
        kernel.close()

    expected = torch.empty_like(logits)
    for row in range(batch):
        probs = logits[row].softmax(dim=0)
        for col in range(vocab):
            mass_above = probs[logits[row] > logits[row, col]].sum()
            expected[row, col] = logits[row, col] if mass_above < p[row] else -float("inf")
    torch.testing.assert_close(masked, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)


@pytest.mark.parametrize("vocab,threads", [(33, 32), (129, 64)])
def test_top_k_mask_ragged_runtime_compare(vocab, threads):
    get_kernel_class("sampling.top_k_mask", "TopKMaskFwdKernel")
    from tileops.kernels.sampling.top_k_mask import _top_k_mask_kernel

    batch = 2
    torch.manual_seed(130)
    logits = torch.randn(batch, vocab, dtype=torch.float32)
    logits[0, 25] = logits[0, 26]
    k = torch.tensor([7, vocab], dtype=torch.int32)
    before = logits.clone()
    masked = torch.empty_like(logits)
    factory = _top_k_mask_kernel(batch, vocab, "float32")
    kernel = compile_tileops_jit(factory, {"threads": threads, "cluster": 1, "slots": (vocab + threads - 1) // threads})
    try:
        kernel(logits.reshape(-1), k, masked.reshape(-1))
    finally:
        kernel.close()

    threshold = logits.sort(dim=-1, descending=True).values.gather(1, (k - 1).long()[:, None])
    expected = torch.where(logits >= threshold, logits, -float("inf"))
    torch.testing.assert_close(masked, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)


@pytest.mark.parametrize("vocab,threads", [(33, 32), (129, 64)])
def test_top_k_top_p_mask_ragged_runtime_compare(vocab, threads):
    get_kernel_class("sampling.top_k_top_p_mask", "TopKTopPMaskFwdKernel")
    from tileops.kernels.sampling.top_k_top_p_mask import _top_k_top_p_mask_kernel

    batch = 2
    torch.manual_seed(131)
    logits = torch.randn(batch, vocab, dtype=torch.float32)
    k = torch.tensor([9, 17], dtype=torch.int32)
    p = torch.tensor([0.7, 0.85], dtype=torch.float32)
    before = logits.clone()
    masked = torch.empty_like(logits)
    factory = _top_k_top_p_mask_kernel(batch, vocab, "float32")
    kernel = compile_tileops_jit(factory, {"threads": threads, "cluster": 1, "slots": (vocab + threads - 1) // threads})
    try:
        kernel(logits.reshape(-1), k, p, masked.reshape(-1))
    finally:
        kernel.close()

    top_k = torch.where(
        logits >= logits.sort(dim=-1, descending=True).values.gather(1, (k - 1).long()[:, None]),
        logits,
        -float("inf"),
    )
    probs = top_k.float().softmax(dim=-1)
    expected = torch.empty_like(logits)
    for row in range(batch):
        for col in range(vocab):
            mass_above = probs[row][top_k[row] > top_k[row, col]].sum()
            expected[row, col] = logits[row, col] if mass_above < p[row] and top_k[row, col].isfinite() else -float("inf")
    torch.testing.assert_close(masked, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
