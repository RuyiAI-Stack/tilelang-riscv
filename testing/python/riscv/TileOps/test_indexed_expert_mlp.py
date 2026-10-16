"""Route-indexed MoE expert MLP: routing, both matmuls, activation, weighted reduction."""

import importlib

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


@pytest.mark.parametrize("dispatch_mode", ["direct", "grouped"])
def test_indexed_expert_mlp_runtime_compare(dispatch_mode):
    get_kernel_class("moe.indexed_expert_gemm", "IndexedExpertGemmTemplate")
    module = importlib.import_module("tileops.kernels.moe.indexed_expert_gemm")
    tokens, topk, experts, hidden, intermediate = 3, 2, 3, 8, 8
    torch.manual_seed(12)
    x = (torch.randn(tokens, hidden) * 0.2).half()
    w1 = (torch.randn(experts, 2 * intermediate, hidden) * 0.2).half()
    w2 = (torch.randn(experts, hidden, intermediate) * 0.2).half()
    ids = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.8, 0.2]])
    capacity = tokens
    meta = torch.zeros((module._route_metadata_size(experts, tokens * topk, capacity),), dtype=torch.int32)
    route = compile_tileops_jit(module._route_stats(tokens, topk, experts), {"threads": 32})
    config = dict(block_n=8, block_k=8, threads=128, num_stages=1)
    up = compile_tileops_jit(
        module._indexed_expert_gemm(tokens, topk, experts, intermediate, hidden, "float16", "silu", False, dispatch_mode), config
    )
    down = compile_tileops_jit(
        module._indexed_expert_gemm(tokens, topk, experts, hidden, intermediate, "float16", "none", True, dispatch_mode), config
    )
    scaling = 0.75
    reduce = compile_tileops_jit(module._weighted_reduce(tokens, topk, hidden, "float16", scaling), {"threads": 32})
    middle = torch.empty((tokens, topk, intermediate), dtype=torch.float16)
    result = torch.empty((tokens, topk, hidden), dtype=torch.float16)
    output = torch.empty((tokens, hidden), dtype=torch.float16)
    try:
        route(ids, meta)
        torch.testing.assert_close(meta[:experts], torch.tensor([2, 2, 2], dtype=torch.int32))
        metadata = meta if dispatch_mode == "grouped" else meta[:1]
        up(x, w1, ids, metadata, middle)
        down(middle, w2, ids, metadata, result)
        reduce(result, weights, output)
    finally:
        route.close()
        up.close()
        down.close()
        reduce.close()
    expected = torch.empty_like(result)
    for token in range(tokens):
        for slot in range(topk):
            expert = ids[token, slot].item()
            projected = x[token].float() @ w1[expert].float().T
            gated = (torch.nn.functional.silu(projected[:intermediate]) * projected[intermediate:]).half()
            expected[token, slot] = (gated.float() @ w2[expert].float().T).half()
    torch.testing.assert_close(result, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(output, (expected.float() * weights.unsqueeze(-1)).sum(1).mul(scaling).half(), atol=2e-3, rtol=2e-2)
