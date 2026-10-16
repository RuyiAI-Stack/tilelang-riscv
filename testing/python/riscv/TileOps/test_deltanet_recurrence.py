from __future__ import annotations

import importlib
import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
):
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    beta_f = beta.float()
    state_f = state.float()
    sk = torch.einsum("bhk,bhkv->bhv", k_f, state_f)
    sq = torch.einsum("bhk,bhkv->bhv", q_f, state_f)
    qk = (q_f * k_f).sum(dim=-1, keepdim=True)
    v_new = beta_f.unsqueeze(-1) * (v_f - sk)
    out = sq + qk * v_new
    new_state = state_f + k_f.unsqueeze(-1) * v_new.unsqueeze(-2)
    return out, new_state


@pytest.mark.parametrize("class_name", ["DeltaNetDecodeKernel", "DeltaNetDecodeFP32Kernel"])
def test_deltanet_decode_float32_runtime_compare(class_name: str):
    batch, heads, dim_k, dim_v = 1, 2, 16, 16
    kernel_cls = get_kernel_class("deltanet_recurrence", class_name)
    tileops_kernel = kernel_cls(
        batch=batch,
        head=heads,
        dim_k=dim_k,
        dim_v=dim_v,
        dtype="float32",
        config={"k_tile": 16, "num_stages": 1, "threads": 128},
    )

    adapter = getattr(tileops_kernel._kernel_fn, "adapter", None)
    assert type(adapter).__name__ == "RiscvKernelAdapter"

    q = torch.linspace(-0.5, 0.5, batch * heads * dim_k, dtype=torch.float32).reshape(batch, heads, dim_k)
    k = torch.linspace(-0.4, 0.4, batch * heads * dim_k, dtype=torch.float32).reshape(batch, heads, dim_k)
    v = torch.linspace(-0.3, 0.3, batch * heads * dim_v, dtype=torch.float32).reshape(batch, heads, dim_v)
    beta = torch.linspace(0.2, 0.8, batch * heads, dtype=torch.float32).reshape(batch, heads)
    state = torch.linspace(
        -0.2,
        0.2,
        batch * heads * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, dim_k, dim_v)

    actual = tileops_kernel(q, k, v, beta, state)
    expected = _reference(q, k, v, beta, state)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)


def test_deltanet_raw_decode_bfloat16_runtime_compare():
    get_kernel_class("deltanet_recurrence", "DeltaNetDecodeRawCudaFlaStyleKernel")
    module = importlib.import_module("tileops.kernels.linear_attention.deltanet.recurrent")
    jit_kernel = module._deltanet_decode_raw_cuda_flastyle_tl(1, 1, 128, 128)
    kernel = compile_tileops_jit(jit_kernel, {"threads": 32})
    q = torch.linspace(-0.1, 0.1, 128, dtype=torch.bfloat16).reshape(1, 1, 128)
    k = torch.linspace(-0.2, 0.2, 128, dtype=torch.bfloat16).reshape(1, 1, 128)
    v = torch.linspace(-0.3, 0.3, 128, dtype=torch.bfloat16).reshape(1, 1, 128)
    beta = torch.full((1, 1), 0.5, dtype=torch.bfloat16)
    state = torch.zeros((1, 1, 128, 128), dtype=torch.bfloat16)

    actual = kernel(q, k, v, beta, state)
    expected = _reference(q, k, v, beta, state)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor.float(), expected_tensor, rtol=2e-2, atol=2e-2)
