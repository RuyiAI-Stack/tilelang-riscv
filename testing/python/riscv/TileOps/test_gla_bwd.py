from __future__ import annotations

import sys
import types

import torch

from ._harness import TILEOPS_ROOT, _load_module, compile_tileops_jit, get_kernel_class
from .test_gla_fwd import _gla_reference, _load_gla_fwd_module, _run_full_gla_forward


def _load_gla_bwd_module():
    _load_gla_fwd_module()
    pkg = sys.modules.setdefault("tileops.kernels.gla", types.ModuleType("tileops.kernels.gla"))
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "gla")]
    return _load_module(
        "tileops.kernels.gla.gla_bwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "gla" / "gla_bwd.py",
    )


def _gla_state_grad_reference(
    q: torch.Tensor,
    g_cumsum: torch.Tensor,
    grad_output: torch.Tensor,
    chunk_size: int,
    scale: float,
) -> torch.Tensor:
    batch, seq_len, heads, dim_k = q.shape
    dim_v = grad_output.shape[-1]
    num_chunks = seq_len // chunk_size
    state_grad = torch.zeros((batch, heads, dim_k, dim_v), dtype=torch.float32)
    result = torch.empty((batch, num_chunks, heads, dim_k, dim_v), dtype=torch.float32)
    for chunk in range(num_chunks - 1, -1, -1):
        result[:, chunk] = state_grad
        start = chunk * chunk_size
        stop = start + chunk_size
        g_chunk = g_cumsum[:, start:stop].float()
        q_gated = q[:, start:stop].float() * torch.exp(g_chunk)
        state_grad = state_grad * torch.exp(g_chunk[:, -1]).unsqueeze(-1)
        state_grad = state_grad + scale * torch.einsum("bthk,bthv->bhkv", q_gated, grad_output[:, start:stop].float())
    return result


def test_gla_bwd_fused_float32_runtime_compare() -> None:
    kernel_cls = get_kernel_class("gla.gla_bwd", "GLAChunkBwdKernel")
    del kernel_cls
    batch, seq_len, heads, dim_k, dim_v, chunk_size = 1, 32, 1, 16, 16, 16
    scale = dim_k**-0.5
    values_k = batch * seq_len * heads * dim_k
    values_v = batch * seq_len * heads * dim_v
    q = torch.linspace(-0.2, 0.2, values_k, dtype=torch.float32).reshape(batch, seq_len, heads, dim_k)
    k = torch.linspace(0.2, -0.2, values_k, dtype=torch.float32).reshape_as(q)
    v = torch.linspace(-0.3, 0.3, values_v, dtype=torch.float32).reshape(batch, seq_len, heads, dim_v)
    g = -torch.linspace(0.01, 0.2, values_k, dtype=torch.float32).reshape_as(q)
    grad_output = torch.linspace(-0.1, 0.1, values_v, dtype=torch.float32).reshape_as(v)

    _output, states, g_cumsum = _run_full_gla_forward(q, k, v, g, chunk_size, scale)
    module = _load_gla_bwd_module()
    fused_jit = module._gla_bwd_fused_kernel(batch, seq_len, heads, dim_k, dim_v, chunk_size, scale, "float32")
    fused_kernel = compile_tileops_jit(fused_jit, {"num_stages": 1, "threads": 64})
    state_grads = _gla_state_grad_reference(q, g_cumsum, grad_output, chunk_size, scale)
    actual = fused_kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g_cumsum,
        grad_output.contiguous(),
        states,
        state_grads,
    )

    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    g_ref = g.detach().requires_grad_(True)
    expected_output = _gla_reference(q_ref, k_ref, v_ref, g_ref, chunk_size, scale)
    expected = torch.autograd.grad(
        (expected_output * grad_output).sum(),
        (q_ref, k_ref, v_ref, g_ref),
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=2e-3, atol=2e-3)
