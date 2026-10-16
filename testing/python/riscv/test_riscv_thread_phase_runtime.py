"""Runtime regressions for local storage across serialized thread phases."""

import pytest
import torch
import tilelang
import tilelang.language as T

from tilelang.tladapter.toolchain import resolve_llvm_root

pytest.importorskip("tilelang.tladapter._native")
if resolve_llvm_root(required=False) is None:
    pytest.skip("LLVM/MLIR runtime toolchain not available", allow_module_level=True)


def test_fragment_initialization_reaches_every_thread_phase():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            shared = T.alloc_shared((4,), "float32")
            fragment = T.alloc_fragment((4,), "float32")
            total = T.alloc_fragment((1,), "float32")
            T.fill(fragment, 3)
            for i in T.Parallel(4):
                shared[i] = A[i]
            # The shared read creates an implicit phase boundary.
            for i in T.Parallel(4):
                fragment[i] = fragment[i] + shared[i]
            T.reduce_sum(fragment, total, dim=0)
            B[tid] = total[0]

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.arange(1, 5, dtype=torch.float32)
        torch.testing.assert_close(compiled(data), (data.sum() + 12).expand(8))
    finally:
        compiled.close()


@pytest.mark.parametrize("builtin", [False, True])
@pytest.mark.parametrize("width", [4, 8])
def test_shuffle_reads_registers_before_following_lane_updates(builtin, width):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tid = T.get_thread_binding()
            value = T.alloc_local((1,), "float32")
            peer = T.alloc_local((1,), "float32")
            value[0] = A[tid]
            if builtin:
                peer[0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), value[0], (tid + 1) % width, width, 32)
            else:
                peer[0] = T.shfl_xor(value[0], 1, width=width)
            value[0] = value[0] + peer[0]
            B[tid] = value[0]

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.arange(1, 33, dtype=torch.float32)
        tid = torch.arange(32)
        source = tid // width * width + (tid + 1) % width if builtin else tid ^ 1
        torch.testing.assert_close(compiled(data), data + data[source])
    finally:
        compiled.close()


def test_split_loop_keeps_unrelated_local_let_inside_each_thread():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            local = T.alloc_local((1,), "int32")
            shared = T.alloc_shared((4,), "float32")
            local[0] = tid
            offset = local[0] * 2
            for step in T.serial(2):
                shared[tid] = A[tid] + T.cast(offset + step, "float32")
                T.sync_threads()
                B[tid] = shared[(tid + 1) % 4] - T.cast(offset, "float32")
                T.sync_threads()

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.tensor([2.0, 4.0, -1.0, 7.0])
        tid = torch.arange(4)
        neighbor = (tid + 1) % 4
        expected = data[neighbor] + neighbor * 2 + 1 - tid * 2
        torch.testing.assert_close(compiled(data), expected)
    finally:
        compiled.close()


@pytest.mark.parametrize("run_loop", [False, True])
def test_uniform_branch_keeps_unrelated_local_let_inside_each_thread(run_loop):
    @T.prim_func
    def kernel(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            local = T.alloc_local((1,), "int32")
            shared = T.alloc_shared((4,), "float32")
            local[0] = tid
            offset = local[0] * 2
            if A[0] > 0:
                for step in T.serial(2):
                    shared[tid] = A[tid] + T.cast(offset + step, "float32")
                    T.sync_threads()
                    B[tid] = shared[(tid + 1) % 4] - T.cast(offset, "float32")
                    T.sync_threads()
            else:
                B[tid] = T.cast(offset, "float32")

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.tensor([2.0 if run_loop else -2.0, 4.0, -1.0, 7.0])
        tid = torch.arange(4)
        neighbor = (tid + 1) % 4
        expected = data[neighbor] + neighbor * 2 + 1 - tid * 2 if run_loop else tid.float() * 2
        torch.testing.assert_close(compiled(data), expected)
    finally:
        compiled.close()


@pytest.mark.parametrize("run_loop", [False, True])
def test_uniform_branch_around_barrier_loop(run_loop):
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            shared = T.alloc_shared((8,), "int32")
            local = T.alloc_local((1,), "int32")
            local[0] = A[tid]
            if A[0] > 0:
                for step in T.serial(3):
                    shared[tid] = local[0] + step
                    T.sync_threads()
                    local[0] = shared[(tid + 1) % 8]
                    T.sync_threads()
            B[tid] = local[0]

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.arange(1, 9, dtype=torch.int32)
        if not run_loop:
            data[0] = -1
        expected = data.roll(-3) + 3 if run_loop else data
        torch.testing.assert_close(compiled(data), expected)
    finally:
        compiled.close()


def test_conditional_tail_loop_does_not_repeat_preceding_shuffle():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            value = T.alloc_local((1,), "float32")
            copies = T.alloc_local((2,), "float32")
            value[0] = T.if_then_else(tid < 4, A[tid], T.float32(0))
            if tid < 4:
                value[0] = T.shfl_xor(value[0], 1, width=4)
                value[0] = value[0] + 1
                for i in T.unroll(2):
                    copies[i] = value[0]
            if tid < 4:
                B[tid] = copies[0] + copies[1]

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.tensor([2.0, 4.0, -1.0, 7.0])
        torch.testing.assert_close(compiled(data), 2 * (data[torch.arange(4) ^ 1] + 1))
    finally:
        compiled.close()


def test_serialized_threads_reuse_static_scratch_without_stack_growth(monkeypatch):
    monkeypatch.setenv("TILELANG_RISCV_CLANG_OPT_LEVEL", "0")

    @T.prim_func
    def kernel(A: T.Tensor((8192,), "float32"), B: T.Tensor((512,), "float32")):
        with T.Kernel(1, threads=512):
            tid = T.get_thread_binding()
            scratch = T.alloc_local((8192,), "float32")
            T.copy(A, scratch)
            B[tid] = scratch[tid] + T.cast(tid, "float32")

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.arange(8192, dtype=torch.float32) * 0.25
        torch.testing.assert_close(compiled(data), data[:512] + torch.arange(512))
    finally:
        compiled.close()


def test_nested_shuffle_loops_use_each_rows_initialized_values():
    @T.prim_func
    def kernel(A: T.Tensor((2, 4), "float32"), B: T.Tensor((2, 4), "float32")):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            value = T.alloc_local((1,), "float32")
            for row in T.serial(2):
                if tid < 4:
                    value[0] = A[row, tid]
                    for stage in T.serial(2):
                        for substage in T.serial(stage + 1):
                            value[0] = T.max(value[0], T.shfl_xor(value[0], 1 << (stage - substage), width=4))
                    B[row, tid] = value[0]

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        data = torch.tensor([[2.0, -4.0, 9.0, 7.0], [-2.0, -4.0, -1.0, -7.0]])
        torch.testing.assert_close(compiled(data), data.amax(dim=1, keepdim=True).expand(2, 4))
    finally:
        compiled.close()


@pytest.mark.parametrize("threads", [32, 64])
def test_first_warp_suffix_scan_in_multiwarp_launch(threads):
    @T.prim_func
    def kernel(A: T.Tensor((2, 256), "int32"), B: T.Tensor((2, 32), "int32")):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding()
            value = T.alloc_local((1,), "int32")
            for row in T.serial(2):
                if tid < 32:
                    value[0] = 0
                    for i in T.serial(8):
                        value[0] = value[0] + A[row, tid * 8 + i]
                    for stage in T.unroll(5):
                        peer = T.shfl_down(value[0], 1 << stage, width=32)
                        if tid + (1 << stage) < 32:
                            value[0] = value[0] + peer
                    B[row, tid] = value[0]

    compiled = tilelang.compile(kernel, out_idx=[1], target="riscv")
    try:
        torch.manual_seed(135)
        data = torch.randint(0, 4, (2, 256), dtype=torch.int32)
        totals = data.reshape(2, 32, 8).sum(-1)
        expected = totals.flip(-1).cumsum(-1).flip(-1).to(torch.int32)
        torch.testing.assert_close(compiled(data), expected, rtol=0, atol=0)
    finally:
        compiled.close()


@pytest.mark.parametrize("threads", [32, 64])
def test_warp_reads_shared_threshold_before_winner_updates_it(threads):
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding()
            threshold = T.alloc_shared((1,), "int32")
            selected = T.alloc_local((1,), "int32")
            if tid == 0:
                threshold[0] = 16
            T.sync_threads()
            if tid < 32:
                saved_threshold = threshold[0]
                selected[0] = T.cast(tid == saved_threshold, "int32")
                if selected[0] != 0:
                    threshold[0] = tid + 1
            T.sync_threads()
            if tid == 0:
                B[0] = threshold[0]

    compiled = tilelang.compile(kernel, out_idx=[0], target="riscv")
    try:
        torch.testing.assert_close(compiled(), torch.tensor([17], dtype=torch.int32))
    finally:
        compiled.close()


@pytest.mark.parametrize("threads", [32, 64])
def test_shared_let_keeps_value_after_shared_update(threads):
    @T.prim_func
    def kernel(B: T.Tensor((threads,), "int32")):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding()
            shared = T.alloc_shared((threads // 32,), "int32")
            if tid % 32 == 0:
                shared[tid // 32] = 100 + tid // 32
            T.sync_threads()
            before = shared[tid // 32]
            after = before - 1
            if tid % 32 == 0:
                shared[tid // 32] = after
            T.sync_threads()
            B[tid] = after + shared[tid // 32]

    compiled = tilelang.compile(kernel, out_idx=[0], target="riscv")
    try:
        expected = 2 * (99 + torch.arange(threads, dtype=torch.int32) // 32)
        torch.testing.assert_close(compiled(), expected, rtol=0, atol=0)
    finally:
        compiled.close()
