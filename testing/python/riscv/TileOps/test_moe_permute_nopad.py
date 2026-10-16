from __future__ import annotations

from types import SimpleNamespace

import torch

from ._harness import get_kernel_class


def test_moe_pre_permute_contiguous_float32_runtime_compare():
    num_tokens, top_k, num_experts, hidden_size = 2, 2, 4, 8
    kernel_cls = get_kernel_class("moe.permute_contiguous", "MoEPrePermuteContiguousKernel")
    call_spec = __import__("tileops.kernels.moe.call_spec", fromlist=["PrePermuteCall"])
    call = call_spec.PrePermuteCall(
        arch=90,
        sm_count=1,
        layout=SimpleNamespace(selection_key="tight_physical_psum", alignment=1),
        input_dtype=torch.float32,
        num_experts=num_experts,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
    )
    tileops_kernel = kernel_cls(
        call,
        config={"threads": 4, "gather_rows_per_block": 2},
    )

    # The public TileOps wrapper asserts CUDA tensors.  Validate the two
    # underlying TileLang kernels directly on SG2044's RISC-V host adapter.
    scan = tileops_kernel._scan_fn(tileops_kernel.config["threads"])
    assert type(getattr(scan, "adapter", None)).__name__ == "RiscvKernelAdapter"
    gather = tileops_kernel._gather_fn()
    assert type(getattr(gather, "adapter", None)).__name__ == "RiscvKernelAdapter"

    flat_ids = torch.tensor([2, 0, 1, 2], dtype=torch.int32)
    hidden = torch.linspace(
        -0.5,
        0.5,
        steps=num_tokens * hidden_size,
        dtype=torch.float32,
    ).reshape(num_tokens, hidden_size)

    physical_ends = torch.empty(num_experts, dtype=torch.int32)
    permuted_idx = torch.empty(num_tokens * top_k, dtype=torch.int32)
    inverse_indices = torch.empty(num_tokens * top_k, dtype=torch.int32)
    perm_h = torch.empty(num_tokens * top_k, hidden_size, dtype=torch.float32)

    scan(
        flat_ids,
        physical_ends,
        permuted_idx,
        inverse_indices,
    )
    gather(hidden, permuted_idx, perm_h)

    torch.testing.assert_close(
        physical_ends,
        torch.tensor([1, 2, 4, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(permuted_idx, torch.tensor([0, 1, 0, 1], dtype=torch.int32))
    torch.testing.assert_close(inverse_indices, torch.tensor([2, 0, 1, 3], dtype=torch.int32))
    torch.testing.assert_close(
        perm_h,
        torch.vstack([hidden[0], hidden[1], hidden[0], hidden[1]]),
    )
