from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    compile_tileops_jit,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
)


def _load_gla_fwd_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.gla",
        types.ModuleType("tileops.kernels.gla"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "gla")]
    return _load_module(
        "tileops.kernels.gla.gla_fwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "gla" / "gla_fwd.py",
    )


def test_gla_fwd_precompute_g_float32_runtime_compare():
    batch, seq_len, heads, dim_k, chunk_size = 1, 4, 2, 4, 2
    module = _load_gla_fwd_module()
    kernel = module.gla_precompute_g_kernel(
        batch,
        seq_len,
        heads,
        dim_k,
        chunk_size,
        "float32",
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    g = torch.linspace(
        -0.4,
        0.4,
        batch * seq_len * heads * dim_k,
        dtype=torch.float32,
    ).reshape(batch, seq_len, heads, dim_k)

    actual = kernel(g)
    expected = torch.empty_like(g)
    for start in range(0, seq_len, chunk_size):
        expected[:, start : start + chunk_size, :, :] = torch.cumsum(
            g[:, start : start + chunk_size, :, :],
            dim=1,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _gla_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    scale: float,
) -> torch.Tensor:
    batch, seq_len, heads, dim_k = q.shape
    dim_v = v.shape[-1]
    num_chunks = seq_len // chunk_size
    g_cum = g.reshape(batch, num_chunks, chunk_size, heads, dim_k).cumsum(dim=2)
    g_cum = g_cum.reshape_as(g)
    state = torch.zeros((batch, heads, dim_k, dim_v), dtype=torch.float32)
    outputs = []
    causal = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.float32))
    for chunk in range(num_chunks):
        start = chunk * chunk_size
        stop = start + chunk_size
        q_chunk = q[:, start:stop].float()
        k_chunk = k[:, start:stop].float()
        v_chunk = v[:, start:stop].float()
        g_chunk = g_cum[:, start:stop].float()
        g_last = g_chunk[:, -1:]

        q_gated = q_chunk * torch.exp(g_chunk)
        inter = scale * torch.einsum("bthk,bhkv->bthv", q_gated, state)
        k_ungated = k_chunk * torch.exp(-g_chunk)
        scores = scale * torch.einsum("bihk,bjhk->bhij", q_gated, k_ungated)
        intra = torch.einsum("bhij,bjhv->bihv", scores * causal, v_chunk)
        outputs.append(inter + intra)

        k_adjusted = k_chunk * torch.exp(g_last - g_chunk)
        state = state * torch.exp(g_last[:, 0]).unsqueeze(-1)
        state = state + torch.einsum("bthk,bthv->bhkv", k_adjusted, v_chunk)
    return torch.cat(outputs, dim=1)


def _run_full_gla_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim_k = q.shape
    dim_v = v.shape[-1]
    module = _load_gla_fwd_module()

    g_jit = module.gla_precompute_g_kernel(batch, seq_len, heads, dim_k, chunk_size, "float32")
    h_jit = module.gla_fwd_h_kernel(
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_size,
        "float32",
        num_v_partitions=1,
        num_k_partitions=1,
    )
    o_jit = module._gla_fwd_o_kernel(batch, seq_len, heads, dim_k, dim_v, chunk_size, scale, "float32")
    g_kernel = compile_tileops_jit(g_jit, {"num_stages": 1, "threads": 64})
    h_kernel = compile_tileops_jit(h_jit, {"num_stages": 1, "threads": 64})
    o_kernel = compile_tileops_jit(o_jit, {"num_stages": 1, "threads": 64})

    g_cumsum = g_kernel(g.contiguous())
    initial_state = torch.zeros((batch, heads, dim_k, dim_v), dtype=torch.float32)
    states = h_kernel(k.contiguous(), v.contiguous(), g_cumsum, initial_state)
    output = o_kernel(q.contiguous(), k.contiguous(), v.contiguous(), g_cumsum, states)
    return output, states, g_cumsum


def test_gla_fwd_full_float32_runtime_compare() -> None:
    batch, seq_len, heads, dim_k, dim_v, chunk_size = 1, 16, 1, 16, 16, 16
    scale = dim_k**-0.5
    values_k = batch * seq_len * heads * dim_k
    values_v = batch * seq_len * heads * dim_v
    q = torch.linspace(-0.2, 0.2, values_k, dtype=torch.float32).reshape(batch, seq_len, heads, dim_k)
    k = torch.linspace(0.2, -0.2, values_k, dtype=torch.float32).reshape_as(q)
    v = torch.linspace(-0.3, 0.3, values_v, dtype=torch.float32).reshape(batch, seq_len, heads, dim_v)
    g = -torch.linspace(0.01, 0.2, values_k, dtype=torch.float32).reshape_as(q)

    actual, _states, _g_cumsum = _run_full_gla_forward(q, k, v, g, chunk_size, scale)
    expected = _gla_reference(q, k, v, g, chunk_size, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
