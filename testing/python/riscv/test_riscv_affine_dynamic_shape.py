"""Recover dynamic symbols from flattened or offset tensor extents."""

import pytest
import torch
import tilelang
import tilelang.language as T


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("scale,offset", [(7, 0), (1, 1), (3, 2)])
def test_affine_extent_binds_dynamic_symbol(dtype, scale, offset):
    n = T.dynamic("n", dtype)

    @T.prim_func
    def kernel(A: T.Tensor((scale * n + offset,), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            total = T.alloc_var("int32", init=0)
            for i in T.serial(n):
                total = total + A[scale * i + offset]
            B[0] = total

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        for count in (0, 1, 5, 19):
            data = torch.arange(scale * count + offset, dtype=torch.int32)
            torch.testing.assert_close(compiled(data), data[offset::scale].sum().to(torch.int32).reshape(1))
    finally:
        compiled.close()
