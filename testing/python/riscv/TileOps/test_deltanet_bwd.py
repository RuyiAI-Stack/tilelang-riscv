from __future__ import annotations

import torch

from ._harness import get_kernel_class
from .test_deltanet_fwd import _load_deltanet_fwd_module


def _solve_unit_lower_triangular(matrix, rhs):
    rows = []
    for i in range(matrix.shape[-1]):
        value = rhs[..., i, :]
        for j, previous in enumerate(rows):
            value = value - matrix[..., i, j, None] * previous
        rows.append(value)
    return torch.stack(rows, dim=-2)


def _deltanet_reference(q, k, v, beta, chunk_size, grad_output):
    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    beta_ref = beta.float().detach().requires_grad_(True)
    batch, heads, seq_len, dim_k = q.shape
    dim_v = v.shape[-1]
    state = torch.zeros(batch, heads, dim_k, dim_v, dtype=torch.float32)
    outputs = []
    eye = torch.eye(chunk_size, dtype=torch.float32)
    causal = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32))

    for start in range(0, seq_len, chunk_size):
        stop = start + chunk_size
        q_chunk = q_ref[:, :, start:stop]
        k_chunk = k_ref[:, :, start:stop]
        v_chunk = v_ref[:, :, start:stop]
        beta_chunk = beta_ref[:, :, start:stop]
        gram = torch.einsum("bhik,bhjk->bhij", k_chunk, k_chunk)
        transform = eye + torch.tril(beta_chunk.unsqueeze(-1) * gram, diagonal=-1)
        w = _solve_unit_lower_triangular(transform, k_chunk * beta_chunk.unsqueeze(-1))
        u = _solve_unit_lower_triangular(transform, v_chunk * beta_chunk.unsqueeze(-1))
        v_new = u - w @ state
        attention = (q_chunk @ k_chunk.transpose(-2, -1)) * causal
        outputs.append(q_chunk @ state + attention @ v_new)
        state = state + k_chunk.transpose(-2, -1) @ v_new

    loss = (torch.cat(outputs, dim=2) * grad_output.float()).sum()
    return torch.autograd.grad(loss, (q_ref, k_ref, v_ref, beta_ref))


def test_deltanet_bwd_full_float32_runtime_compare() -> None:
    batch, heads, seq_len, dim_k, dim_v, chunk_size = 1, 1, 4, 16, 16, 2
    _load_deltanet_fwd_module()
    fwd_cls = get_kernel_class("deltanet.deltanet_fwd", "DeltaNetChunkFwdKernel")
    bwd_cls = get_kernel_class("deltanet.deltanet_bwd", "DeltaNetChunkBwdKernel")
    config = {
        "fused_num_stages": 1,
        "fused_threads": 64,
        "h_num_stages": 1,
        "h_threads": 64,
        "h_block_v": 16,
        "o_threads": 64,
    }
    fwd = fwd_cls(batch, heads, seq_len, chunk_size, dim_k, dim_v, torch.float32, config=config)
    bwd = bwd_cls(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        torch.float32,
        config={
            "num_stages": 1,
            "threads": 64,
            "parallel_threads": 64,
            "recurrence_threads": 64,
        },
    )

    q = torch.linspace(-0.2, 0.2, batch * heads * seq_len * dim_k).reshape(batch, heads, seq_len, dim_k)
    k = torch.linspace(0.2, -0.2, batch * heads * seq_len * dim_k).reshape_as(q)
    v = torch.linspace(-0.3, 0.3, batch * heads * seq_len * dim_v).reshape(batch, heads, seq_len, dim_v)
    beta = torch.tensor([[[0.25, 0.5, 0.35, 0.7]]], dtype=torch.float32)
    grad_output = torch.linspace(-0.1, 0.1, batch * heads * seq_len * dim_v).reshape_as(v)

    _output, state, aw, au, w, u = fwd(q, k, v, beta)
    actual = bwd(grad_output, q, k, v, beta, state, aw, au, w, u)
    expected = _deltanet_reference(q, k, v, beta, chunk_size, grad_output)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-3, atol=2e-3)
