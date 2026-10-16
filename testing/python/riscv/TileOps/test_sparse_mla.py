from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    kv_stride: int,
    q_start_index_s: int,
) -> torch.Tensor:
    batch, seq_len, heads, fused_dim = q.shape
    dim = fused_dim // 2
    scale = fused_dim**-0.5
    out = torch.empty(batch, seq_len, heads, dim, dtype=q.dtype)
    for b in range(batch):
        for s in range(seq_len):
            q_index = q_start_index_s + s
            max_kv_index = min((q_index + 1 - kv_stride) // kv_stride, kv.shape[1] - 1)
            for h in range(heads):
                selected = indices[b, s, 0]
                selected = selected[(selected >= 0) & (selected <= max_kv_index)].long()
                keys = kv[b, selected, 0]
                scores = (keys.float() * q[b, s, h].float()).sum(dim=-1) * scale
                probs = torch.softmax(scores, dim=0)
                out[b, s, h] = (probs[:, None] * keys[:, :dim].float()).sum(dim=0)
    return out


def test_sparse_mla_basic_float32_runtime_compare():
    batch, seq_len, seq_len_kv = 1, 2, 2
    heads, dim, tail_dim, topk = 16, 16, 16, 2
    kv_stride = 1
    q_start_index_s = 0
    kernel_cls = get_kernel_class(
        "attention.deepseek_dsa_decode",
        "DSADecodeBasicKernel",
    )
    tileops_kernel = kernel_cls(
        batch=batch,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        dtype=torch.float32,
        topk=topk,
        kv_stride=kv_stride,
        q_start_index_s=q_start_index_s,
        kv_group=1,
        config={"block_i": 2, "threads": 32, "num_stages": 0},
    )

    q = torch.linspace(
        -0.5,
        0.5,
        batch * seq_len * heads * (dim + tail_dim),
        dtype=torch.float32,
    ).reshape(batch, seq_len, heads, dim + tail_dim)
    kv = torch.linspace(
        -0.4,
        0.4,
        batch * seq_len_kv * (dim + tail_dim),
        dtype=torch.float32,
    ).reshape(batch, seq_len_kv, 1, dim + tail_dim)
    indices = torch.tensor([[[[0, -1]], [[0, 1]]]], dtype=torch.int32)

    actual = tileops_kernel(q, kv, indices)
    expected = _reference(
        q,
        kv,
        indices,
        kv_stride=kv_stride,
        q_start_index_s=q_start_index_s,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
