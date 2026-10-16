from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    block_size: int,
    groups: int,
    scale: float,
    selected_block_num: int,
):
    c_seq_len, heads, dim = q.shape
    head_kv = heads // groups
    out = torch.empty(c_seq_len, head_kv, selected_block_num, dtype=torch.int32)
    for t in range(c_seq_len):
        curr = t // block_size
        hist = (t + 1) // block_size
        for kvh in range(head_kv):
            scores: list[tuple[float, int]] = [(float(groups), curr)]
            if curr > 0:
                hist_scores = []
                for g in range(groups):
                    h = kvh * groups + g
                    group_scores = []
                    for c in range(hist):
                        group_scores.append(torch.dot(q[t, h].float(), k_cmp[c, kvh].float()) * scale)
                    hist_scores.append(torch.stack(group_scores))
                b_lse = torch.logsumexp(torch.stack(hist_scores), dim=1)
                for c in range(curr):
                    imp = 0.0
                    for g in range(groups):
                        imp += torch.exp(hist_scores[g][c] - b_lse[g]).item()
                    scores.append((float(groups) if c in (0, curr - 1) else imp, c))
            scores.sort(key=lambda item: (-item[0], item[1]))
            chosen = [idx for _, idx in scores[:selected_block_num]]
            chosen.extend([-1] * (selected_block_num - len(chosen)))
            out[t, kvh] = torch.tensor(chosen[:selected_block_num], dtype=torch.int32)
    return out


def test_nsa_topk_varlen_float32_runtime_compare():
    seq_num, c_seq_len, heads = 1, 16, 2
    dim = 16
    chunk_num, groups = 2, 2
    block_size = bc = bs = 8
    scale = dim**-0.5
    selected_block_num = 2
    kv_group = heads // groups

    kernel_cls = get_kernel_class("attention.deepseek_nsa_topk", "NSATopKVarlenKernel")
    tileops_kernel = kernel_cls(
        seq_num,
        c_seq_len,
        heads,
        dim,
        chunk_num,
        groups,
        scale,
        selected_block_num,
        bc,
        bs,
        torch.float32,
        torch.float32,
        config={"threads": 32},
    )

    q = torch.linspace(-0.5, 0.5, c_seq_len * heads * dim, dtype=torch.float32).reshape(c_seq_len, heads, dim)
    k_cmp = torch.linspace(
        -0.4,
        0.4,
        chunk_num * kv_group * dim,
        dtype=torch.float32,
    ).reshape(chunk_num, kv_group, dim)
    offsets = torch.tensor([0, c_seq_len], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, chunk_num], dtype=torch.int32)
    token_indices = torch.stack((torch.zeros(c_seq_len, dtype=torch.int32), torch.arange(c_seq_len, dtype=torch.int32)), dim=1)

    actual = tileops_kernel(q, k_cmp, offsets, chunk_offsets, token_indices)
    expected = _reference(q, k_cmp, block_size, groups, scale, selected_block_num)
    torch.testing.assert_close(actual, expected)
