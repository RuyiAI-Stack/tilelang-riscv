"""Original upstream min-p masking kernel with an independent row reference."""

import torch

from ._harness import compile_tileops_jit, get_kernel_class


def test_min_p_mask_ragged_runtime_compare():
    get_kernel_class("sampling.min_p_mask", "MinPMaskFwdKernel")
    from tileops.kernels.sampling.min_p_mask import _min_p_mask_kernel

    batch, vocab = 2, 33
    torch.manual_seed(128)
    logits = torch.randn(batch, vocab, dtype=torch.float32)
    min_p = torch.tensor([0.2, 0.8], dtype=torch.float32)
    before = logits.clone()
    partial = torch.empty(batch, dtype=torch.float32)
    masked = torch.empty_like(logits)
    factory = _min_p_mask_kernel(batch, vocab, "float32", 32, 1)
    kernel = compile_tileops_jit(factory, {"reg_tiles": 2, "smem_tiles": 0, "pace": 1})
    try:
        kernel(logits.reshape(-1), min_p, partial, masked.reshape(-1))
    finally:
        kernel.close()
    threshold = logits.amax(dim=-1, keepdim=True) + min_p.log()[:, None]
    expected = torch.where(logits < threshold, -float("inf"), logits)
    torch.testing.assert_close(masked, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
