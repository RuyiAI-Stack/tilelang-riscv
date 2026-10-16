from __future__ import annotations

import pytest
import torch
from tilelang import tvm

from ._harness import compile_tileops_jit, get_kernel_class


def test_mla_decode_ws_rejects_cuda_wgmma_extern():
    batch, heads, kv_heads, seqlen_kv, dim, pe_dim = 1, 8, 1, 8, 16, 8
    kernel_cls = get_kernel_class("attention.deepseek_mla_decode", "MLADecodeWSKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        kv_head_num=kv_heads,
        seqlen_kv=seqlen_kv,
        dim=dim,
        pe_dim=pe_dim,
        dtype=torch.float32,
        config={
            "block_H": 8,
            "block_N": 16,
            "num_split": 1,
            "num_stages": 1,
            "threads": 384,
        },
    )

    with pytest.raises(tvm.error.InternalError, match="Unsupported call_extern.*tileops::uniform_wgmma_bases"):
        compile_tileops_jit(tileops_kernel.kernel, tileops_kernel.config)
