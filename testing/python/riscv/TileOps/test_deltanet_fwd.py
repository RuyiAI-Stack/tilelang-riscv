from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    get_kernel_class,
    _load_module,
)


def _load_deltanet_fwd_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.deltanet",
        types.ModuleType("tileops.kernels.deltanet"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "deltanet")]
    _load_module(
        "tileops.kernels.deltanet.fused_prepare_compute_w_u",
        TILEOPS_ROOT / "tileops" / "kernels" / "deltanet" / "fused_prepare_compute_w_u.py",
    )
    return _load_module(
        "tileops.kernels.deltanet.deltanet_fwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "deltanet" / "deltanet_fwd.py",
    )


def test_deltanet_fwd_output_o_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_deltanet_fwd_module()
    kernel = module._output_o_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    q = torch.linspace(
        -0.5,
        0.5,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    k = torch.linspace(
        -0.4,
        0.4,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    state = torch.linspace(
        -0.2,
        0.2,
        batch * heads * (seq_len // chunk_size + 1) * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    v_new = torch.linspace(
        -0.3,
        0.3,
        batch * heads * seq_len * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_v)

    actual = kernel(q, k, state, v_new)

    q_chunk = q[0, 0]
    k_chunk = k[0, 0]
    state_0 = state[0, 0, 0]
    v_chunk = v_new[0, 0]
    expected = q_chunk @ state_0 + torch.tril(q_chunk @ k_chunk.T) @ v_chunk
    torch.testing.assert_close(
        actual,
        expected.reshape(batch, heads, seq_len, dim_v),
        rtol=1e-5,
        atol=1e-5,
    )


def test_deltanet_fwd_kernel_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    kernel_cls = get_kernel_class("deltanet.deltanet_fwd", "DeltaNetChunkFwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        torch.float32,
        config={
            "fused_num_stages": 1,
            "fused_threads": 64,
            "h_num_stages": 1,
            "h_threads": 64,
            "h_block_v": 16,
            "o_threads": 64,
        },
    )

    q = torch.linspace(
        -0.5,
        0.5,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    k = torch.linspace(
        -0.4,
        0.4,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    v = torch.linspace(
        -0.3,
        0.3,
        batch * heads * seq_len * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_v)
    beta = torch.tensor([[[0.2, 0.7]]], dtype=torch.float32)

    actual_o, actual_s, _actual_aw, _actual_au, actual_w, actual_u = tileops_kernel(q, k, v, beta)

    gram = k[0, 0] @ k[0, 0].T
    transform = torch.eye(chunk_size, dtype=torch.float32)
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i > j:
                transform[i, j] = -gram[i, j] * beta[0, 0, i]
    expected_w = (transform @ (k[0, 0] * beta[0, 0].unsqueeze(-1))).reshape(batch, heads, seq_len, dim_k)
    expected_u = (transform @ (v[0, 0] * beta[0, 0].unsqueeze(-1))).reshape(batch, heads, seq_len, dim_v)

    state = torch.zeros(batch, heads, dim_k, dim_v, dtype=torch.float32)
    expected_s = torch.empty(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    expected_s[:, :, 0] = state
    expected_v_new = torch.empty_like(expected_u)
    for chunk in range(seq_len // chunk_size):
        start = chunk * chunk_size
        end = start + chunk_size
        k_c = k[:, :, start:end].float()
        w_c = expected_w[:, :, start:end].float()
        u_c = expected_u[:, :, start:end].float()
        v_c = u_c - torch.einsum("bhtk,bhkv->bhtv", w_c, state)
        expected_v_new[:, :, start:end] = v_c
        state = state + torch.einsum("bhtk,bhtv->bhkv", k_c, v_c)
        expected_s[:, :, chunk + 1] = state

    expected_o = q[0, 0] @ expected_s[0, 0, 0]
    expected_o += torch.tril(q[0, 0] @ k[0, 0].T) @ expected_v_new[0, 0]
    expected_o = expected_o.reshape(batch, heads, seq_len, dim_v)

    torch.testing.assert_close(actual_w, expected_w, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_u, expected_u, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_s, expected_s, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_o, expected_o, rtol=1e-5, atol=1e-5)
