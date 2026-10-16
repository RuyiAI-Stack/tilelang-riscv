from __future__ import annotations

import math

import torch

from ._harness import get_kernel_class


def _reference(flat: torch.Tensor, num_experts: int, block_size: int):
    numel = flat.numel()
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    counts = [(flat == expert).sum().item() for expert in range(num_experts)]
    padded = [math.ceil(count / block_size) * block_size for count in counts]
    post_pad = sum(padded)
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32)
    expert_ids = torch.empty(math.ceil(max_padded / block_size), dtype=torch.int32)

    offset = 0
    for expert, padded_count in enumerate(padded):
        ids = [idx for idx, value in enumerate(flat.tolist()) if value == expert]
        if ids:
            sorted_ids[offset : offset + len(ids)] = torch.tensor(ids, dtype=torch.int32)
        for block in range(offset // block_size, (offset + padded_count) // block_size):
            expert_ids[block] = expert
        offset += padded_count
    return sorted_ids, expert_ids, torch.tensor([post_pad], dtype=torch.int32)


def test_moe_permute_align_small_batch_int32_runtime_compare():
    numel, num_experts, block_size = 8, 4, 4
    kernel_cls = get_kernel_class("moe.permute_align", "MoEPermuteAlignKernel")
    tileops_kernel = kernel_cls(numel=numel, num_experts=num_experts, block_size=block_size)

    # The public TileOps wrapper asserts CUDA tensors.  Validate the underlying
    # small-batch TileLang kernel directly on SG2044's RISC-V host adapter.
    kernel = tileops_kernel._small_batch_fn()
    adapter = getattr(kernel, "adapter", None)
    assert type(adapter).__name__ == "RiscvKernelAdapter"

    flat = torch.tensor([2, 0, 1, 2, 3, 1, 0, 2], dtype=torch.int32)
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_blocks = math.ceil(max_padded / block_size)
    sorted_ids = torch.empty(max_padded, dtype=torch.int32)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32)
    post_pad = torch.empty(1, dtype=torch.int32)

    kernel(flat, sorted_ids, expert_ids, post_pad)

    expected_sorted, expected_expert, expected_post = _reference(flat, num_experts, block_size)
    valid_blocks = expected_post.item() // block_size
    torch.testing.assert_close(post_pad, expected_post)
    torch.testing.assert_close(sorted_ids[: expected_post.item()], expected_sorted[: expected_post.item()])
    torch.testing.assert_close(expert_ids[:valid_blocks], expected_expert[:valid_blocks])
