from __future__ import annotations

import pytest
import torch

from ._harness import compile_tileops_kernel, get_elementwise_kernel_class, run_unary_runtime_compare, tileops_kernel_tir


@pytest.mark.parametrize(
    ("kernel_name", "input_factory", "reference", "rtol", "atol"),
    [
        ("ReluFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.relu, 1e-6, 1e-6),
        (
            "GeluTanhFwdKernel",
            lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32),
            lambda value: 0.5 * value * (1.0 + torch.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value))),
            1e-5,
            1e-5,
        ),
        (
            "GeluFwdKernel",
            lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32),
            lambda value: torch.nn.functional.gelu(value, approximate="none"),
            1e-5,
            1e-5,
        ),
        (
            "ErfFwdKernel",
            lambda: torch.linspace(-3.0, 3.0, 256, dtype=torch.float32),
            torch.erf,
            1e-5,
            1e-5,
        ),
        # The newest upstream sigmoid uses a CUDA-only vector load for aligned rows;
        # this odd length exercises its generic TIR load and ragged tail on RISC-V.
        ("SigmoidFwdKernel", lambda: torch.linspace(-4.0, 4.0, 257, dtype=torch.float32), torch.sigmoid, 1e-5, 1e-5),
        ("TanhFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.tanh, 1e-5, 1e-5),
        (
            "SiluFwdKernel",
            lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32),
            lambda value: value * torch.sigmoid(value),
            1e-5,
            1e-5,
        ),
        ("ExpFwdKernel", lambda: torch.linspace(-2.0, 2.0, 256, dtype=torch.float32), torch.exp, 1e-5, 1e-5),
        ("LogFwdKernel", lambda: torch.linspace(0.1, 4.0, 256, dtype=torch.float32), torch.log, 1e-5, 1e-5),
        ("Log1pFwdKernel", lambda: torch.linspace(0.0, 4.0, 256, dtype=torch.float32), torch.log1p, 1e-5, 1e-5),
        ("Expm1FwdKernel", lambda: torch.linspace(-2.0, 2.0, 256, dtype=torch.float32), torch.expm1, 1e-5, 1e-5),
        ("SqrtFwdKernel", lambda: torch.linspace(0.0, 4.0, 256, dtype=torch.float32), torch.sqrt, 1e-5, 1e-5),
        (
            "RsqrtFwdKernel",
            lambda: torch.linspace(0.1, 4.0, 256, dtype=torch.float32),
            torch.rsqrt,
            1e-5,
            1e-5,
        ),
        (
            "ReciprocalFwdKernel",
            lambda: torch.linspace(0.1, 4.0, 256, dtype=torch.float32),
            torch.reciprocal,
            1e-5,
            1e-5,
        ),
        ("FloorFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.floor, 1e-6, 1e-6),
        ("CeilFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.ceil, 1e-6, 1e-6),
        ("SinFwdKernel", lambda: torch.linspace(-2.0, 2.0, 256, dtype=torch.float32), torch.sin, 1e-5, 1e-5),
        ("CosFwdKernel", lambda: torch.linspace(-2.0, 2.0, 256, dtype=torch.float32), torch.cos, 1e-5, 1e-5),
        ("RoundFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.round, 1e-6, 1e-6),
        ("TruncFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.trunc, 1e-6, 1e-6),
        ("NegFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.neg, 1e-6, 1e-6),
        ("AbsFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.abs, 1e-6, 1e-6),
        ("SignFwdKernel", lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32), torch.sign, 1e-6, 1e-6),
        (
            "HardsigmoidFwdKernel",
            lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32),
            lambda value: torch.clamp(value + 3.0, min=0.0, max=6.0) / 6.0,
            1e-6,
            1e-6,
        ),
        (
            "HardswishFwdKernel",
            lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32),
            lambda value: value * torch.clamp(value + 3.0, min=0.0, max=6.0) / 6.0,
            1e-6,
            1e-6,
        ),
        (
            "MishFwdKernel",
            lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32),
            lambda value: value * torch.tanh(torch.nn.functional.softplus(value)),
            1e-5,
            1e-5,
        ),
        (
            "SeluFwdKernel",
            lambda: torch.linspace(-4.0, 4.0, 256, dtype=torch.float32),
            torch.nn.functional.selu,
            1e-5,
            1e-5,
        ),
    ],
)
def test_unary_float32_runtime_compare(kernel_name, input_factory, reference, rtol, atol):
    x = input_factory()
    run_unary_runtime_compare(
        get_elementwise_kernel_class(kernel_name),
        x,
        reference,
        kernel_kwargs={"strategy": "direct"},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 1e-5), (torch.float16, 1e-3), (torch.bfloat16, 8e-3)])
@pytest.mark.parametrize("size", [256, 257])
def test_sigmoid_precision_and_load_paths_runtime_compare(dtype, rtol, size):
    # Aligned lengths select upstream's streaming vector load; odd lengths
    # select its generic load and exercise the ragged tail.
    # Large negative values cover __fdividef's large-divisor/overflow behavior.
    x = torch.linspace(-12.0, 12.0, size).to(dtype)
    x[:12] = torch.tensor([-float("inf"), -120.0, -90.0, -88.0, -87.0, -20.0, -0.0, 0.0, 20.0, 88.0, 120.0, float("inf")], dtype=dtype)
    instance = get_elementwise_kernel_class("SigmoidFwdKernel")(N_total=size, dtype=dtype, config={"threads": 32, "num_per_thread": 8})
    tir = tileops_kernel_tir(instance).script()
    assert ("tileops::load16_evict_last" in tir) == (size == 256)
    before = x.view(torch.uint8).clone()
    kernel = compile_tileops_kernel(instance)
    try:
        expected = torch.sigmoid(x.float()).to(dtype)
        torch.testing.assert_close(kernel(x), expected, rtol=rtol, atol=0)
        torch.testing.assert_close(x.view(torch.uint8), before, rtol=0, atol=0)
    finally:
        kernel.close()
