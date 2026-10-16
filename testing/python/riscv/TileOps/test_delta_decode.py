"""Original DeltaNet and KDA decode programs checked against one recurrence step."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


def _reference(q, k, v, beta, state, l2norm):
    q, k, v, beta = [x.float() for x in (q, k, v, beta)]
    if l2norm:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    repeat = v.shape[1] // q.shape[1]
    q, k = [x.repeat_interleave(repeat, dim=1) for x in (q, k)]
    residual = (v - torch.einsum("bhk,bhkv->bhv", k, state)) * beta.unsqueeze(-1)
    final = state + k.unsqueeze(-1) * residual.unsqueeze(-2)
    output = torch.einsum("bhk,bhkv->bhv", q, final) * q.shape[-1] ** -0.5
    return output, final


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("l2norm", [False, True])
@pytest.mark.parametrize("has_initial", [False, True])
def test_deltanet_decode_runtime_compare(dtype, l2norm, has_initial):
    get_kernel_class("linear_attention.deltanet.dense_decode", "DeltaNetDenseDecodeFwdKernel")
    from tileops.kernels.linear_attention.delta_decode import delta_decode_tl

    batch, heads, dim = 2, 2, 64
    torch.manual_seed(136)
    q, k, v = [(torch.randn(batch, 1, heads, dim) * 0.1).to(dtype) for _ in range(3)]
    beta = torch.rand(batch, 1, heads).to(dtype)
    state = torch.randn(batch, heads, dim, dim) * 0.03 if has_initial else torch.zeros(batch, heads, dim, dim)
    inputs = (q, k, v, beta, state)
    originals = [x.clone() for x in inputs]
    factory = delta_decode_tl(
        batch,
        heads,
        heads,
        dim,
        dim**-0.5,
        str(dtype).split(".")[-1],
        gated=False,
        gate_in_kernel=False,
        beta_sigmoid=False,
        allow_neg_eigval=False,
        l2norm=l2norm,
        has_initial_state=has_initial,
        state_v_first=False,
        threads=32,
        lane_group=1,
    )
    compiled = compile_tileops_jit(factory, {})
    try:
        unread = torch.empty(1)
        actual, final = compiled(q, k, v, torch.empty(1, dtype=dtype), beta, unread, unread, state if has_initial else unread)
    finally:
        compiled.close()
    expected, expected_state = _reference(q[:, 0], k[:, 0], v[:, 0], beta[:, 0], state, l2norm)
    torch.testing.assert_close(actual, expected.unsqueeze(1).to(dtype), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(final, expected_state, rtol=2e-4, atol=2e-5)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("l2norm", [False, True])
def test_kda_decode_runtime_compare(dtype, l2norm):
    get_kernel_class("linear_attention.kda.decode", "KDARecurrentDecodeFwdKernel")
    from tileops.kernels.linear_attention.kda.decode_program import decode_program

    batch, heads, value_heads, dim = 2, 1, 2, 64
    torch.manual_seed(137)
    q, k = [(torch.randn(1, batch, heads, dim) * 0.1).to(dtype) for _ in range(2)]
    v = (torch.randn(1, batch, value_heads, dim) * 0.1).to(dtype)
    gate = (-torch.rand(1, batch, value_heads, dim) * 0.3).to(dtype)
    beta = torch.rand(1, batch, value_heads).to(dtype)
    state = torch.randn(batch, value_heads, dim, dim) * 0.03
    inputs = (q, k, v, gate, beta, state)
    originals = [x.clone() for x in inputs]
    # This upstream builder returns the compiled JIT program directly.
    compiled = decode_program(heads, value_heads, dim, dim, str(dtype).split(".")[-1], dim**-0.5, l2norm, batch, value_tile=32, threads=128)
    try:
        actual, final = compiled(*inputs)
    finally:
        compiled.close()
    decayed = state * gate[0].float().exp().unsqueeze(-1)
    expected, expected_state = _reference(q[0], k[0], v[0], beta[0], decayed, l2norm)
    torch.testing.assert_close(actual, expected.unsqueeze(0).to(dtype), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(final, expected_state, rtol=2e-4, atol=2e-5)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
