from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _apply_rope(
    x: torch.Tensor,
    positions: list[int],
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rotary_dim: int,
):
    y = x.clone()
    half = rotary_dim // 2
    for row, pos in enumerate(positions):
        for head in range(x.shape[1]):
            for d in range(rotary_dim):
                freq_idx = d % half
                paired_d = d + half if d < half else d - half
                paired_val = x[row, head, paired_d]
                rotated = -paired_val if d < half else paired_val
                y[row, head, d] = x[row, head, d] * cos_table[pos, freq_idx] + rotated * sin_table[pos, freq_idx]
    return y


def _gather_pages(
    pages: torch.Tensor,
    block_table: torch.Tensor,
    length: int,
    page_size: int,
):
    dense = torch.empty(1, length, pages.shape[1], pages.shape[2], dtype=pages.dtype)
    for pos in range(length):
        page_idx = pos // page_size
        page_offset = pos % page_size
        physical = block_table[0, page_idx].item() * page_size + page_offset
        dense[0, pos] = pages[physical]
    return dense


def _reference(q_rot: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor, heads_kv: int):
    total_q, heads, dim = q_rot.shape
    heads_per_group = heads // heads_kv
    out = torch.empty_like(q_rot)
    scale = dim**-0.5
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum("qd,kd->qk", q_rot[:, head, :].float(), k_all[0, :, group, :].float()) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, head, :] = torch.einsum("qk,kd->qd", probs, v_all[0, :, group, :].float())
    return out


def test_gqa_prefill_paged_kvcache_rope_fwd_float32_runtime_compare():
    batch, heads, heads_kv = 1, 4, 2
    seq_new, old_len, page_size, max_pages = 3, 2, 4, 2
    dim, rotary_dim = 16, 8
    seqlen_kv = max_pages * page_size
    kernel_cls = get_kernel_class("attention.gqa_fwd", "GQAPrefillPagedWithKVCacheRoPEFwdKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=max_pages,
        page_size=page_size,
        dim=dim,
        max_position=seqlen_kv,
        rotary_dim=rotary_dim,
        is_causal=False,
        dtype=torch.float32,
        config={"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, seq_new * heads * dim, dtype=torch.float32).reshape(seq_new, heads, dim)
    k_new = torch.linspace(-0.4, 0.4, seq_new * heads_kv * dim, dtype=torch.float32).reshape(seq_new, heads_kv, dim)
    v_new = torch.linspace(-0.3, 0.3, seq_new * heads_kv * dim, dtype=torch.float32).reshape(seq_new, heads_kv, dim)
    k_pages = torch.zeros(seqlen_kv, heads_kv, dim, dtype=torch.float32)
    v_pages = torch.zeros_like(k_pages)
    k_pages[:old_len] = torch.linspace(-0.2, 0.2, old_len * heads_kv * dim, dtype=torch.float32).reshape(old_len, heads_kv, dim)
    v_pages[:old_len] = torch.linspace(-0.1, 0.1, old_len * heads_kv * dim, dtype=torch.float32).reshape(old_len, heads_kv, dim)
    cu_seqlens_q = torch.tensor([0, seq_new], dtype=torch.int32)
    cache_seqlens = torch.tensor([old_len], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    half = rotary_dim // 2
    cos_table = torch.linspace(0.7, 1.0, seqlen_kv * half, dtype=torch.float32).reshape(seqlen_kv, half)
    sin_table = torch.linspace(-0.2, 0.2, seqlen_kv * half, dtype=torch.float32).reshape(seqlen_kv, half)

    actual = tileops_kernel(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        None,
        None,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
        seq_new,
        cos_table,
        sin_table,
    )
    positions = list(range(old_len, old_len + seq_new))
    q_rot = _apply_rope(q, positions, cos_table, sin_table, rotary_dim)
    k_new_rot = _apply_rope(k_new, positions, cos_table, sin_table, rotary_dim)
    k_all = torch.cat([_gather_pages(k_pages, block_table, old_len, page_size), k_new_rot.unsqueeze(0)], dim=1)
    v_all = torch.cat([_gather_pages(v_pages, block_table, old_len, page_size), v_new.unsqueeze(0)], dim=1)
    expected = _reference(q_rot, k_all, v_all, heads_kv)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
