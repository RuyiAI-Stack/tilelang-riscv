"""Dynamic while loops must observe mutable state and the nearest loop's break."""

import pytest
import torch
import tilelang
import tilelang.language as T


def test_while_dynamic_lane_counts_and_zero_iterations():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            remaining = T.alloc_var("int32", init=A[tid])
            total = T.alloc_var("int32", init=0)
            while remaining > 0:
                total = total + remaining
                remaining = remaining - 1
            B[tid] = total

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        for data in (torch.tensor([0, 1, 9, -2, 3, 16, 2, 5], dtype=torch.int32), torch.arange(8, dtype=torch.int32)):
            positive = data.clamp_min(0)
            torch.testing.assert_close(compiled(data), positive * (positive + 1) // 2)
    finally:
        compiled.close()


def test_while_nested_break_and_buffer_condition():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            i = T.alloc_var("int32", init=0)
            j = T.alloc_var("int32", init=0)
            B[tid] = 0
            while i < A[tid]:
                j = 0
                while j < 10:
                    j = j + 1
                    if j == 3:
                        T.loop_break()
                    B[tid] = B[tid] + 1
                for k in T.serial(5):
                    if k == 2:
                        T.loop_break()
                    B[tid] = B[tid] + 10
                i = i + 1
                if B[tid] >= 66:
                    T.loop_break()
                B[tid] = B[tid] + 100
            # Break from the preceding loop must not escape into this for loop.
            for k in T.serial(2):  # noqa: B007 - exercise a surrounding serial loop
                j = 0
                while j < 4:
                    j = j + 1
                    if j == 2:
                        T.loop_break()
                    B[tid] = B[tid] + 1000

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.tensor([0, 1, 2, 3, -1, 5, 8, 1], dtype=torch.int32)
        expected = torch.tensor([2000 + (0 if n <= 0 else 122 if n == 1 else 144) for n in data.tolist()], dtype=torch.int32)
        torch.testing.assert_close(compiled(data), expected)
    finally:
        compiled.close()


@pytest.mark.parametrize("shuffle", [False, True])
def test_while_cross_thread_cooperation_is_rejected(shuffle):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tid = T.get_thread_binding()
            i = T.alloc_var("int32", init=0)
            value = T.alloc_var("float32", init=A[tid])
            while i < 2:
                if shuffle:
                    value = T.shfl_xor(value, 1)
                else:
                    T.sync_threads()
                i = i + 1
            B[tid] = value

    with pytest.raises(Exception, match="[Ww]hile|phase|synchronization"):
        tilelang.compile(kernel, out_idx=[1], target="riscv")


def test_while_condition_observes_global_buffer_updates():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            B[tid] = 0
            while B[tid] < A[tid]:
                B[tid] = B[tid] + 3

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        values = torch.arange(-2, 6, dtype=torch.int32)
        torch.testing.assert_close(compiled(values), (values.clamp_min(0) + 2) // 3 * 3)
    finally:
        compiled.close()
