from __future__ import annotations

import math

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def _reference_attention_bwd(q, k, v, do, is_causal):
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)

    batch, seq_len, heads, dim = q_ref.shape
    outs = []
    lse = torch.empty((batch, heads, seq_len), dtype=torch.float32)

    for b in range(batch):
        per_head = []
        for h in range(heads):
            scores = (q_ref[b, :, h, :] @ k_ref[b, :, h, :].T) / math.sqrt(dim)
            if is_causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask, -float("inf"))
            lse[b, h] = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
            probs = torch.softmax(scores, dim=-1)
            per_head.append(probs @ v_ref[b, :, h, :])
        outs.append(torch.stack(per_head, dim=1))

    out = torch.stack(outs, dim=0)
    delta = (out.detach() * do).sum(dim=-1).permute(0, 2, 1).contiguous()
    (out * do).sum().backward()
    return q_ref.grad, k_ref.grad, v_ref.grad, lse, delta


def test_mha_bwd_float32_runtime_compare():
    batch, heads, seq_len, dim = 1, 1, 4, 4
    is_causal = False

    q = torch.linspace(-0.4, 0.5, batch * seq_len * heads * dim, dtype=torch.float32).reshape(batch, seq_len, heads, dim)
    k = torch.linspace(0.2, -0.3, batch * seq_len * heads * dim, dtype=torch.float32).reshape(batch, seq_len, heads, dim)
    v = torch.linspace(-0.7, 0.6, batch * seq_len * heads * dim, dtype=torch.float32).reshape(batch, seq_len, heads, dim)
    do = torch.linspace(0.1, 0.8, batch * seq_len * heads * dim, dtype=torch.float32).reshape(batch, seq_len, heads, dim)

    exp_dq, exp_dk, exp_dv, lse, delta = _reference_attention_bwd(q, k, v, do, is_causal)

    kernel_cls = get_kernel_class("attention.gqa_bwd", "GQABwdWGMMAPipelinedKernel")
    tileops_kernel = kernel_cls(
        batch,
        heads,
        heads,
        seq_len,
        dim,
        is_causal,
        torch.float32,
        config={"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        do.contiguous(),
        lse.contiguous(),
        delta.contiguous(),
        dq,
        dk,
        dv,
    )

    torch.testing.assert_close(dq, exp_dq, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk, exp_dk, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv, exp_dv, rtol=1e-4, atol=1e-4)
