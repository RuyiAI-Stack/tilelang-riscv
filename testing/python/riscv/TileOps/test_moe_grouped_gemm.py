"""Numerical coverage of the upstream MMA implementation and its five layouts."""

import pytest
import torch
import torch.nn.functional as F

from ._harness import compile_tileops_jit


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["tight_psum", "tight_per_row", "aligned_psum", "aligned_per_row", "masked"])
@pytest.mark.parametrize("activation", ["none", "silu_and_mul", "gelu_and_mul"])
def test_moe_grouped_mma_runtime(dtype, layout, activation):
    from tileops.kernels.gemm.persistent.heuristics import GemmType
    from tileops.kernels.moe.moe_grouped_gemm import _moe_grouped_gemm_mma_kernel

    torch.manual_seed(96)
    n, k = 8 if activation == "none" else 16, 8
    if layout == "masked":
        gtype = GemmType.M_GROUPED_MASKED
        a = torch.randn(3, 4, k).to(dtype) * 0.2
        metadata = torch.tensor([3, 0, 2], dtype=torch.int32)
        ids = [0, 0, 0, -1, -1, -1, -1, -1, 2, 2, -1, -1]
    else:
        aligned = layout.startswith("aligned")
        ids = [0] * (4 if aligned else 3) + [2] * (4 if aligned else 2)
        per_row = layout.endswith("per_row")
        metadata = torch.tensor(ids if per_row else ([4, 4, 8] if aligned else [3, 3, 5]), dtype=torch.int32)
        gtype = getattr(GemmType, "M_GROUPED_" + layout.upper())
        a = torch.randn(len(ids), k).to(dtype) * 0.2
    b = torch.randn(3, n, k).to(dtype) * 0.2
    expected = torch.zeros(len(ids), 8)
    for row, expert in enumerate(ids):
        if expert >= 0:
            value = a.reshape(-1, k)[row].float() @ b[expert].float().T
            if activation != "none":
                gate, up = value.chunk(2)
                value = (F.silu(gate) if activation == "silu_and_mul" else F.gelu(gate)) * up
            expected[row] = value
    factory = _moe_grouped_gemm_mma_kernel(gtype.value, 3, n, k, 4, 4, activation, str(dtype).split(".")[-1], "float32")
    kernel = compile_tileops_jit(factory, dict(block_m=4, block_n=n, block_k=8, num_stages=1, threads=128), out_idx=[])
    out = torch.zeros(*a.shape[:-1], 8)
    try:
        kernel(a, b, metadata, out)
        torch.testing.assert_close(out.reshape(-1, 8), expected, atol=3e-5, rtol=3e-5)
        out.zero_()
        kernel(a, b * 0, metadata, out)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    finally:
        kernel.close()
