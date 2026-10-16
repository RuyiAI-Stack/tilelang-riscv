from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_cb_producer_float16_runtime_compare():
    batch, num_chunks, n_groups, chunk_len, d_state = 1, 2, 2, 2, 4
    seq_len = num_chunks * chunk_len
    c_mat = torch.linspace(-1.0, 1.0, batch * seq_len * n_groups * d_state, dtype=torch.float16).reshape(batch, seq_len, n_groups, d_state)
    b_mat = torch.linspace(1.0, -1.0, batch * seq_len * n_groups * d_state, dtype=torch.float16).reshape(batch, seq_len, n_groups, d_state)

    kernel_cls = get_kernel_class("mamba.cb_producer", "SSDChunkCouplingKernel")
    tileops_kernel = kernel_cls(
        batch,
        num_chunks,
        n_groups,
        chunk_len,
        d_state,
        torch.float16,
        config={"block_l": 1, "block_s": 1, "block_n": 4, "threads": 1},
    )
    actual = compile_tileops_kernel(tileops_kernel)(c_mat.contiguous(), b_mat.contiguous())

    expected = torch.zeros((batch, num_chunks, n_groups, chunk_len, chunk_len), dtype=torch.float16)
    for bb in range(batch):
        for bc in range(num_chunks):
            for g in range(n_groups):
                for l in range(chunk_len):
                    for s in range(chunk_len):
                        if s <= l:
                            lhs = c_mat[bb, bc * chunk_len + l, g]
                            rhs = b_mat[bb, bc * chunk_len + s, g]
                            expected[bb, bc, g, l, s] = torch.sum(lhs * rhs)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
