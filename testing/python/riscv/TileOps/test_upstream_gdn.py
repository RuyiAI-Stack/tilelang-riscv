"""Current upstream GDN decode program, with independent state/output references."""

import pytest
import torch

from ._harness import compile_tileops_jit


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gdn_decode_upstream_runtime(state_v_first, dtype):
    from tileops.kernels.linear_attention.delta_decode import delta_decode_tl

    torch.manual_seed(94)
    q, k = [(torch.randn(1, 1, 1, 64) * 0.1).to(dtype) for _ in range(2)]
    v = (torch.randn(1, 1, 2, 64) * 0.2).to(dtype)
    g = torch.full((1, 1, 2), -0.1, dtype=dtype)
    beta = torch.full((1, 1, 2), 0.3, dtype=dtype)
    state = torch.randn(1, 2, 64, 64) * 0.05
    before = state.clone()
    factory = delta_decode_tl(
        1,
        1,
        2,
        64,
        64**-0.5,
        str(dtype).split(".")[-1],
        gated=True,
        gate_in_kernel=False,
        beta_sigmoid=False,
        allow_neg_eigval=False,
        l2norm=False,
        has_initial_state=True,
        state_v_first=state_v_first,
        threads=32,
        lane_group=1,
    )
    kernel = compile_tileops_jit(factory, {})
    try:
        unused = torch.empty(1)
        output, final = kernel(q, k, v, g, beta, unused, unused, state)
    finally:
        kernel.close()
    canonical = state.transpose(-1, -2) if state_v_first else state
    decayed = canonical * g.float().exp().reshape(1, 2, 1, 1)
    kk = k[:, 0].float().expand(1, 2, 64)
    residual = (v[:, 0].float() - torch.einsum("bhk,bhkv->bhv", kk, decayed)) * beta[:, 0].float().unsqueeze(-1)
    expected_state = decayed + kk.unsqueeze(-1) * residual.unsqueeze(-2)
    expected = torch.einsum("bhk,bhkv->bhv", q[:, 0].float().expand_as(kk), expected_state) * 64**-0.5
    torch.testing.assert_close(output, expected.unsqueeze(1).to(dtype), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(final, expected_state.transpose(-1, -2) if state_v_first else expected_state, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(state, before, atol=0, rtol=0)


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [5, 67])
def test_gdn_prefill_upstream_pipeline(state_v_first, dtype, seq_len):
    from tileops.kernels.linear_attention.call_spec import GDNCall
    from tileops.kernels.linear_attention.chunk_prefill import run_chunk_prefill

    torch.manual_seed(95)
    q, k = [(torch.randn(1, seq_len, 1, 64) * 0.1).to(dtype) for _ in range(2)]
    v = (torch.randn(1, seq_len, 2, 64) * 0.2).to(dtype)
    g = torch.full((1, seq_len, 2), -0.1, dtype=dtype)
    beta = torch.full((1, seq_len, 2), 0.3, dtype=dtype)
    initial = torch.randn(1, 2, 64, 64) * 0.01
    before = [x.clone() for x in (q, k, v, g, beta, initial)]
    call = GDNCall(
        batch=1,
        seq_len=seq_len,
        heads=1,
        value_heads=2,
        dim_k=64,
        dim_v=64,
        dtype=q.dtype,
        scale=64**-0.5,
        num_sequences=1,
        has_initial_state=True,
        state_v_first=state_v_first,
        device=q.device,
        arch=89,
        sm_count=1,
        smem_budget=98304,
    )
    output, final = run_chunk_prefill(call, (q, k, v, g, beta, initial, None, None, None), gate_mode="head")
    state = initial.transpose(-1, -2).clone() if state_v_first else initial.clone()
    expected = []
    for t in range(seq_len):
        kk = k[:, t].float().expand(1, 2, 64)
        state = state * g[:, t].float().exp().reshape(1, 2, 1, 1)
        delta = (v[:, t].float() - torch.einsum("bhk,bhkv->bhv", kk, state)) * beta[:, t].float().unsqueeze(-1)
        state = state + kk.unsqueeze(-1) * delta.unsqueeze(-2)
        expected.append(torch.einsum("bhk,bhkv->bhv", q[:, t].float().expand_as(kk), state) * 64**-0.5)
    torch.testing.assert_close(output, torch.stack(expected, 1).to(dtype), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(final, state.transpose(-1, -2) if state_v_first else state, atol=2e-3, rtol=2e-2)
    for actual_input, original in zip((q, k, v, g, beta, initial), before):
        torch.testing.assert_close(actual_input, original, rtol=0, atol=0)
