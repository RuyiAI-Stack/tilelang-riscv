"""Integer bit intrinsics used by generic radix kernels."""

import pytest
import torch
import tilelang
import tilelang.language as T


@pytest.mark.parametrize("dtype,bits", [("uint32", 32), ("uint64", 64)])
def test_count_leading_zeros_runtime(dtype, bits):
    @T.prim_func
    def kernel(A: T.Tensor((7,), dtype), B: T.Tensor((7,), "int32")):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(7):
                B[i] = T.clz(A[i])

    values = [0, 1, 2, 3, 255, 1 << (bits - 1), (1 << bits) - 1]
    data = torch.tensor(values, dtype=getattr(torch, dtype))
    expected = torch.tensor([bits - value.bit_length() for value in values], dtype=torch.int32)
    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        torch.testing.assert_close(compiled(data), expected)
    finally:
        compiled.close()


def test_packed_unsigned_maximum_runtime():
    @T.prim_func
    def kernel(A: T.Tensor((6,), "uint32"), B: T.Tensor((6,), "uint32"), C: T.Tensor((6,), "uint32")):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(6):
                C[i] = T.call_extern("uint32", "__vmaxu2", A[i], B[i])

    a = [0, 0xFFFFFFFF, 0x80000001, 0xFFFF0000, 0x7FFF8000, 0x12345678]
    b = [0xFFFFFFFF, 0, 0x00018000, 0x0000FFFF, 0x80007FFF, 0x12345678]
    expected = [max(x & 0xFFFF, y & 0xFFFF) | (max(x >> 16, y >> 16) << 16) for x, y in zip(a, b)]
    compiled = tilelang.compile(kernel, out_idx=[2], target="riscv")
    try:
        torch.testing.assert_close(
            compiled(torch.tensor(a, dtype=torch.uint32), torch.tensor(b, dtype=torch.uint32)),
            torch.tensor(expected, dtype=torch.uint32),
        )
    finally:
        compiled.close()


def test_unsigned_high_multiply_uint32_runtime():
    @T.prim_func
    def kernel(A: T.Tensor((6,), "uint32"), B: T.Tensor((6,), "uint32"), C: T.Tensor((6,), "uint32")):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(6):
                C[i] = T.call_extern("uint32", "__umulhi", A[i], B[i])

    a = [0, 1, 0xFFFFFFFF, 0x80000000, 0x12345678, 0xFFFF0001]
    b = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x80000000, 0x9ABCDEF0, 0x0001FFFF]
    expected = torch.tensor([((x * y) >> 32) & 0xFFFFFFFF for x, y in zip(a, b)], dtype=torch.uint32)
    compiled = tilelang.compile(kernel, out_idx=[2], target="riscv")
    try:
        torch.testing.assert_close(
            compiled(torch.tensor(a, dtype=torch.uint32), torch.tensor(b, dtype=torch.uint32)),
            expected,
        )
    finally:
        compiled.close()
