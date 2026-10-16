from __future__ import annotations

import torch

from ._harness import compile_tileops_jit


def _reference(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    old_len: int,
    block_table: torch.Tensor,
    page_size: int,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rotary_dim: int,
    physical_tokens: int,
):
    _, heads_kv, dim = k_new.shape
    half = rotary_dim // 2
    k_pages = torch.zeros(physical_tokens, heads_kv, dim, dtype=torch.float32)
    v_pages = torch.zeros_like(k_pages)
    for pos in range(k_new.shape[0]):
        logical = old_len + pos
        physical = block_table[0, logical // page_size].item() * page_size + logical % page_size
        for h in range(heads_kv):
            for d in range(dim):
                val = k_new[pos, h, d]
                if d < rotary_dim:
                    freq_idx = d % half
                    paired_d = d + half if d < half else d - half
                    paired_val = k_new[pos, h, paired_d]
                    rotated = -paired_val if d < half else paired_val
                    val = val * cos_table[logical, freq_idx] + rotated * sin_table[logical, freq_idx]
                k_pages[physical, h, d] = val
                v_pages[physical, h, d] = v_new[pos, h, d]
    return k_pages, v_pages


def test_gqa_prefill_paged_kvcache_rope_append_float32_runtime_compare():
    batch, heads_kv, seq_new, old_len, page_size, max_pages, dim, rotary_dim = (
        1,
        2,
        3,
        2,
        4,
        2,
        16,
        8,
    )
    max_position = old_len + seq_new
    physical_tokens = max_pages * page_size
    from tileops.kernels.attention.gqa.prefill_paged_kv_append import _gqa_prefill_paged_with_kv_cache_rope_append_kernel

    factory = _gqa_prefill_paged_with_kv_cache_rope_append_kernel(
        batch, heads_kv, seq_new, physical_tokens, max_pages, page_size, dim, max_position, rotary_dim, "float32"
    )
    kernel = compile_tileops_jit(factory, dict(block_m=4, threads=128))

    k_new = torch.linspace(-0.4, 0.4, seq_new * heads_kv * dim, dtype=torch.float32).reshape(seq_new, heads_kv, dim)
    v_new = torch.linspace(-0.3, 0.3, seq_new * heads_kv * dim, dtype=torch.float32).reshape(seq_new, heads_kv, dim)
    k_pages = torch.zeros(physical_tokens, heads_kv, dim, dtype=torch.float32)
    v_pages = torch.zeros_like(k_pages)
    cu_seqlens_q = torch.tensor([0, seq_new], dtype=torch.int32)
    cache_seqlens = torch.tensor([old_len], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    half = rotary_dim // 2
    cos_table = torch.linspace(0.7, 1.0, max_position * half, dtype=torch.float32).reshape(max_position, half)
    sin_table = torch.linspace(-0.2, 0.2, max_position * half, dtype=torch.float32).reshape(max_position, half)

    kernel(
        k_new,
        v_new,
        k_pages,
        v_pages,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
        cos_table,
        sin_table,
        seq_new,
    )
    expected_k, expected_v = _reference(
        k_new,
        v_new,
        old_len=old_len,
        block_table=block_table,
        page_size=page_size,
        cos_table=cos_table,
        sin_table=sin_table,
        rotary_dim=rotary_dim,
        physical_tokens=physical_tokens,
    )

    torch.testing.assert_close(k_pages, expected_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_pages, expected_v, rtol=0, atol=0)
