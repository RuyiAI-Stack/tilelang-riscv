from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from harness import T, build_mlir_from_tilelang_prim, lower_tilelang_prim_to_mlir


@dataclass(frozen=True)
class RejectionCase:
    name: str
    factory: Callable[[], object]
    expected: str


def _sync_threads_kernel():
    @T.prim_func
    def sync_threads_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = A[tid]
            T.sync_threads()
            B[tid] = B[tid] + T.float32(1)

    return sync_threads_probe


def _shared_alloc_kernel():
    @T.prim_func
    def shared_alloc_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            scratch[tid] = A[tid]
            B[tid] = scratch[tid]

    return shared_alloc_probe


def _shared_alloc_sync_kernel():
    @T.prim_func
    def shared_alloc_sync_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            scratch[tid] = A[tid]
            T.sync_threads()
            B[tid] = scratch[tid]

    return shared_alloc_sync_probe


def _thread_invariant_shared_alloc_kernel():
    @T.prim_func
    def thread_invariant_shared_alloc_probe(
        A: T.Tensor((4,), "float32"),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=4):
            scratch = T.alloc_shared((4,), "float32")
            for i in T.Parallel(4):
                scratch[i] = A[i]
            for i in T.Parallel(4):
                B[i] = scratch[i]

    return thread_invariant_shared_alloc_probe


def _thread_invariant_shared_alloc_sync_kernel():
    @T.prim_func
    def thread_invariant_shared_alloc_sync_probe(
        A: T.Tensor((4,), "float32"),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=4):
            scratch = T.alloc_shared((4,), "float32")
            for i in T.Parallel(4):
                scratch[i] = A[i]
            T.sync_threads()
            for i in T.Parallel(4):
                scratch[i] = scratch[i] * T.float32(2)
            T.sync_threads()
            for i in T.Parallel(4):
                B[i] = scratch[i]

    return thread_invariant_shared_alloc_sync_probe


def _mixed_shared_local_sync_kernel():
    @T.prim_func
    def mixed_shared_local_sync_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            tmp = T.alloc_local((1,), "float32")
            tmp[0] = A[tid]
            scratch[tid] = tmp[0]
            T.sync_threads()
            B[tid] = scratch[tid]

    return mixed_shared_local_sync_probe


def _local_alloc_cross_sync_kernel():
    @T.prim_func
    def local_alloc_cross_sync_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            tmp = T.alloc_local((1,), "float32")
            tmp[0] = A[tid]
            scratch[tid] = tmp[0]
            T.sync_threads()
            B[tid] = scratch[tid] + tmp[0]

    return local_alloc_cross_sync_probe


def _serial_loop_sync_threads_kernel():
    @T.prim_func
    def serial_loop_sync_threads_probe(
        A: T.Tensor((2, 4), "float32"),
        B: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            tmp = T.alloc_local((1,), "float32")
            tmp[0] = T.float32(0)
            for i in T.serial(0, 2):
                tmp[0] = A[i, tid]
                scratch[tid] = tmp[0]
                T.sync_threads()
                B[i, tid] = scratch[tid] + tmp[0]

    return serial_loop_sync_threads_probe


def _serial_loop_conditional_sync_threads_kernel():
    @T.prim_func
    def serial_loop_conditional_sync_threads_probe(
        A: T.Tensor((2, 4), "float32"),
        B: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            for i in T.serial(0, 2):
                if i == 0:
                    scratch[tid] = A[i, tid]
                    T.sync_threads()
                B[i, tid] = scratch[tid]

    return serial_loop_conditional_sync_threads_probe


def _thread_partitioned_common_loop_leading_sync_kernel():
    @T.prim_func
    def thread_partitioned_common_loop_leading_sync_probe(
        A: T.Tensor((12, 2), "float32"),
        B: T.Tensor((12, 2), "float32"),
    ):
        with T.Kernel(1, threads=12):
            tid = T.get_thread_binding()
            if tid < 4:
                for i in T.serial(0, 2):
                    B[tid, i] = A[tid, i] + T.float32(1)
            elif tid < 8:
                for i in T.serial(0, 2):
                    T.evaluate(T.call_intrin("handle", "tir.tvm_storage_sync", "shared", 4, 4))
                    B[tid, i] = A[tid, i] + T.float32(2)
            else:
                for i in T.serial(0, 2):
                    T.evaluate(T.call_intrin("handle", "tir.tvm_storage_sync", "shared", 5, 4))
                    B[tid, i] = A[tid, i] + T.float32(3)

    return thread_partitioned_common_loop_leading_sync_probe


def _nested_split_serial_loop_sync_threads_kernel():
    @T.prim_func
    def nested_split_serial_loop_sync_threads_probe(
        A: T.Tensor((2, 2, 4), "float32"),
        B: T.Tensor((2, 2, 4), "float32"),
    ):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            tmp = T.alloc_local((1,), "float32")
            for outer in T.serial(0, 2):
                tmp[0] = A[outer, 0, tid]
                T.sync_threads()
                for inner in T.serial(0, 2):
                    scratch[tid] = tmp[0] + A[outer, inner, tid]
                    T.sync_threads()
                    B[outer, inner, tid] = scratch[tid]

    return nested_split_serial_loop_sync_threads_probe


def _single_branch_common_loop_leading_sync_kernel():
    @T.prim_func
    def single_branch_common_loop_leading_sync_probe(
        A: T.Tensor((8, 2), "float32"),
        B: T.Tensor((8, 2), "float32"),
    ):
        with T.Kernel(1, threads=8):
            tid = T.get_thread_binding()
            for i in T.serial(0, 2):
                B[tid, i] = A[tid, i]
            if tid < 4:
                for i in T.serial(0, 2):
                    T.evaluate(T.call_intrin("handle", "tir.tvm_storage_sync", "shared", 4, 4))
                    B[tid, i] = B[tid, i] + T.float32(1)

    return single_branch_common_loop_leading_sync_probe


def _let_wrapped_sync_threads_kernel():
    @T.prim_func
    def let_wrapped_sync_threads_probe(
        A: T.Tensor((2, 4), "float32"),
        B: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            for i in T.serial(0, 2):
                with T.LetStmt(A[i, tid]) as lane_value:
                    scratch[tid] = lane_value
                    T.sync_threads()
                    B[i, tid] = scratch[tid] + lane_value

    return let_wrapped_sync_threads_probe


def _serial_loop_trailing_sync_threads_kernel():
    @T.prim_func
    def serial_loop_trailing_sync_threads_probe(
        A: T.Tensor((2, 4), "float32"),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            for i in T.serial(0, 2):
                scratch[tid] = A[i, tid]
                T.sync_threads()
            B[tid] = scratch[tid] + T.float32(1)

    return serial_loop_trailing_sync_threads_probe


def _sync_warp_kernel():
    @T.prim_func
    def sync_warp_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = A[tid]
            T.sync_warp()
            B[tid] = B[tid] + T.float32(1)

    return sync_warp_probe


def _lane_dependent_sync_warp_loop_kernel():
    @T.prim_func
    def lane_dependent_sync_warp_loop_probe(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
    ):
        with T.Kernel(1, threads=64):
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            scratch = T.alloc_shared((2, 32), "float32")
            for i in T.serial(warp * 64 + lane, warp * 64 + 64, 32):
                scratch[warp, lane] = A[i]
                B[i] = scratch[warp, lane] + T.float32(1)
                T.sync_warp()

    return lane_dependent_sync_warp_loop_probe


def _let_wrapped_lane_dependent_sync_warp_loop_kernel():
    @T.prim_func
    def let_wrapped_lane_dependent_sync_warp_loop_probe(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
    ):
        with T.Kernel(1, threads=64):
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            scratch = T.alloc_shared((2, 32), "float32")
            start = T.alloc_local((1,), "int32")
            end = T.alloc_local((1,), "int32")
            start[0] = warp * 64
            end[0] = warp * 64 + 57
            with T.LetStmt((end[0] + 31) // 32 * 32) as aligned_end:  # noqa: SIM117 - preserve nested TIR scopes
                with T.LetStmt(T.uint32(1 << lane) + T.uint32(1 << lane) - T.uint32(1)) as lane_mask:
                    with T.LetStmt(~lane_mask) as lane_mask_rev:
                        for i in T.serial(start[0] + lane, aligned_end, 32):
                            if i < end[0]:
                                scratch[warp, lane] = A[i]
                                if T.bitwise_and(lane_mask, lane_mask_rev) == T.uint32(0):
                                    B[i] = scratch[warp, lane] + T.float32(1)
                            T.sync_warp()

    return let_wrapped_lane_dependent_sync_warp_loop_probe


def _shfl_sync_kernel():
    @T.prim_func
    def shfl_sync_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            value = A[tid]
            B[tid] = T.shfl_sync(value, 0)

    return shfl_sync_probe


def _shfl_sync_dynamic_source_lane_kernel():
    @T.prim_func
    def shfl_sync_dynamic_source_lane_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            source_lane = tid // 2 * 2
            B[tid] = T.shfl_sync(A[tid], source_lane, width=2)

    return shfl_sync_dynamic_source_lane_probe


def _shfl_sync_local_value_kernel():
    @T.prim_func
    def shfl_sync_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = A[tid]
            B[tid] = T.shfl_sync(tmp[0], 0)

    return shfl_sync_local_value_probe


def _shfl_sync_guarded_local_value_kernel():
    @T.prim_func
    def shfl_sync_guarded_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            if tid < 2:
                tmp[0] = A[tid]
            B[tid] = T.shfl_sync(tmp[0], 0)

    return shfl_sync_guarded_local_value_probe


def _shfl_sync_loop_local_value_kernel():
    @T.prim_func
    def shfl_sync_loop_local_value_probe(A: T.Tensor((8,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = 0
            for i in T.unroll(2):
                tmp[0] = tmp[0] + A[tid * 2 + i]
            B[tid] = T.shfl_sync(tmp[0], 0)

    return shfl_sync_loop_local_value_probe


def _shfl_sync_dynamic_loop_local_value_kernel():
    @T.prim_func
    def shfl_sync_dynamic_loop_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = 0
            for i in T.serial(0, tid + 1):
                tmp[0] = tmp[0] + A[tid] + i
            B[tid] = T.shfl_sync(tmp[0], 0)

    return shfl_sync_dynamic_loop_local_value_probe


def _shfl_sync_loop_float_local_value_kernel():
    @T.prim_func
    def shfl_sync_loop_float_local_value_probe(A: T.Tensor((8,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "float32")
            tmp[0] = 0
            for i in T.unroll(2):
                tmp[0] = tmp[0] + A[tid * 2 + i]
            B[tid] = T.shfl_sync(tmp[0], 0)

    return shfl_sync_loop_float_local_value_probe


def _shfl_xor_kernel():
    @T.prim_func
    def shfl_xor_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            value = A[tid]
            B[tid] = T.shfl_xor(value, 1)

    return shfl_xor_probe


def _shfl_xor_local_value_kernel():
    @T.prim_func
    def shfl_xor_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = A[tid]
            B[tid] = T.shfl_xor(tmp[0], 1)

    return shfl_xor_local_value_probe


def _shfl_xor_loop_local_value_kernel():
    @T.prim_func
    def shfl_xor_loop_local_value_probe(A: T.Tensor((8,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = 0
            for i in T.unroll(2):
                tmp[0] = tmp[0] + A[tid * 2 + i]
            B[tid] = T.shfl_xor(tmp[0], 1)

    return shfl_xor_loop_local_value_probe


def _shfl_xor_dynamic_loop_local_value_kernel():
    @T.prim_func
    def shfl_xor_dynamic_loop_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = 0
            for i in T.serial(0, tid + 1):
                tmp[0] = tmp[0] + A[tid] + i
            B[tid] = T.shfl_xor(tmp[0], 1)

    return shfl_xor_dynamic_loop_local_value_probe


def _shfl_down_kernel():
    @T.prim_func
    def shfl_down_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.shfl_down(A[tid], 1)

    return shfl_down_probe


def _shfl_down_local_value_kernel():
    @T.prim_func
    def shfl_down_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = A[tid]
            B[tid] = T.shfl_down(tmp[0], 1)

    return shfl_down_local_value_probe


def _shfl_up_kernel():
    @T.prim_func
    def shfl_up_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.shfl_up(A[tid], 1)

    return shfl_up_probe


def _shfl_up_local_value_kernel():
    @T.prim_func
    def shfl_up_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = A[tid]
            B[tid] = T.shfl_up(tmp[0], 1)

    return shfl_up_local_value_probe


def _builtin_warp_shuffle_up_kernel():
    @T.prim_func
    def builtin_warp_shuffle_up_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.tvm_warp_shuffle_up(T.uint32(4294967295), A[tid], 1, 32, 32)

    return builtin_warp_shuffle_up_probe


def _warp_reduce_kernel():
    @T.prim_func
    def warp_reduce_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.warp_reduce_sum(A[tid])

    return warp_reduce_probe


def _warp_reduce_local_value_kernel():
    @T.prim_func
    def warp_reduce_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = A[tid]
            B[tid] = T.warp_reduce_sum(tmp[0])

    return warp_reduce_local_value_probe


def _warp_reduce_loop_local_value_kernel():
    @T.prim_func
    def warp_reduce_loop_local_value_probe(A: T.Tensor((8,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = 0
            for i in T.unroll(2):
                tmp[0] = tmp[0] + A[tid * 2 + i]
            B[tid] = T.warp_reduce_sum(tmp[0])

    return warp_reduce_loop_local_value_probe


def _warp_reduce_dynamic_loop_local_value_kernel():
    @T.prim_func
    def warp_reduce_dynamic_loop_local_value_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "int32")
            tmp[0] = 0
            for i in T.serial(0, tid + 1):
                tmp[0] = tmp[0] + A[tid] + i
            B[tid] = T.warp_reduce_sum(tmp[0])

    return warp_reduce_dynamic_loop_local_value_probe


def _warp_reduce_max_kernel():
    @T.prim_func
    def warp_reduce_max_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.warp_reduce_max(A[tid])

    return warp_reduce_max_probe


def _warp_reduce_min_kernel():
    @T.prim_func
    def warp_reduce_min_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.warp_reduce_min(A[tid])

    return warp_reduce_min_probe


def _warp_reduce_bitand_kernel():
    @T.prim_func
    def warp_reduce_bitand_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.warp_reduce_bitand(A[tid])

    return warp_reduce_bitand_probe


def _warp_reduce_bitor_kernel():
    @T.prim_func
    def warp_reduce_bitor_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.warp_reduce_bitor(A[tid])

    return warp_reduce_bitor_probe


def _warp_reduce_bitxor_kernel():
    @T.prim_func
    def warp_reduce_bitxor_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.warp_reduce_bitxor(A[tid])

    return warp_reduce_bitxor_probe


def _warp_reduce_max_local_vector_kernel():
    @T.prim_func
    def warp_reduce_max_local_vector_probe(A: T.Tensor((8,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            vals = T.alloc_local((2,), "int32")
            acc = T.alloc_local((1,), "int32")
            for i in T.vectorized(2):
                vals[i] = A[tid * 2 + i]
            acc[0] = T.int32(-2147483648)
            for i in T.unroll(2):
                acc[0] = T.max(acc[0], vals[i])
            B[tid] = T.warp_reduce_max(acc[0])

    return warp_reduce_max_local_vector_probe


def _warp_reduce_max_float_local_vector_kernel():
    @T.prim_func
    def warp_reduce_max_float_local_vector_probe(A: T.Tensor((8,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            vals = T.alloc_local((2,), "float32")
            acc = T.alloc_local((1,), "float32")
            for i in T.vectorized(2):
                vals[i] = A[tid * 2 + i]
            acc[0] = 0
            for i in T.unroll(2):
                acc[0] = T.max(acc[0], vals[i])
            B[tid] = T.warp_reduce_max(acc[0])

    return warp_reduce_max_float_local_vector_probe


def _warp_reduce_sum_thread_index_helper_kernel():
    @T.prim_func
    def warp_reduce_sum_thread_index_helper_probe(A: T.Tensor((4, 2, 32, 4), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1, threads=256):
            tid = T.get_thread_binding()
            lane = T.get_lane_idx()
            warp = T.get_warp_idx()
            group = warp % 2
            head = warp // 2
            vals = T.alloc_local((4,), "float32")
            acc = T.alloc_local((1,), "float32")
            for i in T.vectorized(4):
                vals[i] = A[head, group, lane, i]
            acc[0] = 0
            for i in T.unroll(4):
                acc[0] = acc[0] + vals[i]
            B[tid] = T.warp_reduce_sum(acc[0])

    return warp_reduce_sum_thread_index_helper_probe


def _warp_reduce_sum_self_store_kernel():
    @T.prim_func
    def warp_reduce_sum_self_store_probe(A: T.Tensor((8,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            tmp = T.alloc_local((1,), "float32")
            tmp[0] = 0
            for i in T.unroll(2):
                tmp[0] = tmp[0] + A[tid * 2 + i]
            tmp[0] = T.warp_reduce_sum(tmp[0])
            B[tid] = tmp[0]

    return warp_reduce_sum_self_store_probe


def _warp_reduce_max_self_store_kernel():
    @T.prim_func
    def warp_reduce_max_self_store_probe(A: T.Tensor((8,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            vals = T.alloc_local((2,), "float32")
            tmp = T.alloc_local((1,), "float32")
            for i in T.vectorized(2):
                vals[i] = A[tid * 2 + i]
            tmp[0] = T.float32(-3.4028234663852886e38)
            for i in T.unroll(2):
                tmp[0] = T.max(tmp[0], vals[i])
            tmp[0] = T.warp_reduce_max(tmp[0])
            B[tid] = tmp[0]

    return warp_reduce_max_self_store_probe


def _warp_reduce_sum_self_store_shared_phase_kernel():
    @T.prim_func
    def warp_reduce_sum_self_store_shared_phase_probe(A: T.Tensor((2, 64), "float32"), B: T.Tensor((2, 64), "float32")):
        with T.Kernel(1, threads=64):
            tid = T.get_thread_binding()
            lane = T.get_lane_idx()
            warp = T.get_warp_idx()
            scratch = T.alloc_shared((2,), "float32")
            tmp = T.alloc_local((1,), "float32")
            for i in T.serial(0, 2):
                tmp[0] = A[i, tid]
                tmp[0] = T.warp_reduce_sum(tmp[0])
                if lane == 0:
                    scratch[warp] = tmp[0]
                T.sync_threads()
                B[i, tid] = scratch[warp]

    return warp_reduce_sum_self_store_shared_phase_probe


def _warp_vote_kernel():
    @T.prim_func
    def warp_vote_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.any_sync(A[tid] > 0)

    return warp_vote_probe


def _all_sync_kernel():
    @T.prim_func
    def all_sync_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.all_sync(A[tid] > 0)

    return all_sync_probe


def _ballot_kernel():
    @T.prim_func
    def ballot_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "uint64")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.ballot_sync(A[tid] > 0)

    return ballot_probe


def _ballot_full_mask_kernel():
    @T.prim_func
    def ballot_full_mask_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "uint64")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.ballot(A[tid] > 0)

    return ballot_full_mask_probe


def _activemask_kernel():
    @T.prim_func
    def activemask_probe(B: T.Tensor((4,), "uint64")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.activemask()

    return activemask_probe


def _match_any_kernel():
    @T.prim_func
    def match_any_probe(A: T.Tensor((4,), "uint32"), B: T.Tensor((4,), "uint32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.match_any_sync(A[tid])

    return match_any_probe


def _match_any_non_linear_index_kernel():
    @T.prim_func
    def match_any_non_linear_index_probe(A: T.Tensor((8,), "uint32"), B: T.Tensor((4,), "uint32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.match_any_sync(A[tid * 2])

    return match_any_non_linear_index_probe


def _match_all_kernel():
    @T.prim_func
    def match_all_probe(A: T.Tensor((4,), "uint32"), B: T.Tensor((4,), "uint32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.match_all_sync(A[tid])

    return match_all_probe


def _match_all_non_linear_index_kernel():
    @T.prim_func
    def match_all_non_linear_index_probe(A: T.Tensor((8,), "uint32"), B: T.Tensor((4,), "uint32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.match_all_sync(A[tid * 2])

    return match_all_non_linear_index_probe


def _syncthreads_or_kernel():
    @T.prim_func
    def syncthreads_or_probe(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            B[tid] = T.syncthreads_or(A[tid] > 0)

    return syncthreads_or_probe


def _shuffle_elect_kernel():
    @T.prim_func
    def shuffle_elect_probe(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            if T.shuffle_elect(4):
                B[tid] = 1
            else:
                B[tid] = 0

    return shuffle_elect_probe


def _single_thread_cooperative_expr_kernel():
    @T.prim_func
    def single_thread_cooperative_expr_probe(
        A: T.Tensor((1,), "uint32"),
        P: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
        C: T.Tensor((1,), "uint64"),
        D: T.Tensor((1,), "uint32"),
        E: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            pred = P[0] > 0
            B[0] = T.any_sync(pred) + T.all_sync(pred) + T.syncthreads_or(pred)
            C[0] = T.activemask() + T.ballot_sync(pred) + T.ballot(pred)
            D[0] = T.match_any_sync(A[0]) + T.match_all_sync(A[0])
            if T.shuffle_elect(1):
                E[0] = 1
            else:
                E[0] = 0

    return single_thread_cooperative_expr_probe


def _sync_grid_kernel():
    @T.prim_func
    def sync_grid_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(4, threads=1) as (pid,):
            B[pid] = A[pid]
            T.sync_grid()
            B[pid] = B[pid] + T.float32(1)

    return sync_grid_probe


def _single_block_sync_grid_kernel():
    @T.prim_func
    def single_block_sync_grid_probe(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            tid = T.get_thread_binding()
            scratch = T.alloc_shared((4,), "float32")
            scratch[tid] = A[tid]
            T.sync_grid()
            B[tid] = scratch[tid] + T.float32(1)

    return single_block_sync_grid_probe


CASES = (
    RejectionCase(
        name="match_any_sync_non_linear_index",
        factory=_match_any_non_linear_index_kernel,
        expected="tl.match_any_sync expression is not supported yet",
    ),
    RejectionCase(
        name="match_all_sync_non_linear_index",
        factory=_match_all_non_linear_index_kernel,
        expected="tl.match_all_sync expression is not supported yet",
    ),
    RejectionCase(
        name="shuffle_elect",
        factory=_shuffle_elect_kernel,
        expected="tl.tl_shuffle_elect expression is not supported yet",
    ),
    RejectionCase(
        name="sync_grid",
        factory=_sync_grid_kernel,
        expected="tl.sync_grid inside non-unit launch is not supported yet.*cooperative grid runtime",
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_cooperative_thread_intrinsics_are_rejected(case: RejectionCase):
    with pytest.raises(Exception, match=case.expected):
        build_mlir_from_tilelang_prim(case.factory(), case.name)


def test_sync_threads_lowers_by_splitting_serial_thread_phases():
    source = lower_tilelang_prim_to_mlir(
        _sync_threads_kernel(),
        "sync_threads",
        "cooperative/sync_threads",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("scf.for") >= 2


def test_match_any_sync_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _match_any_kernel(),
        "match_any_sync",
        "cooperative/match_any_sync",
    )

    assert "tl.match_any_sync" not in source
    assert "Unsupported" not in source
    assert "arith.andi" in source
    assert "arith.select" in source
    assert "scf.if" in source


def test_match_all_sync_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _match_all_kernel(),
        "match_all_sync",
        "cooperative/match_all_sync",
    )

    assert "tl.match_all_sync" not in source
    assert "Unsupported" not in source
    assert "arith.andi" in source
    assert "arith.ori" in source
    assert "arith.select" in source
    assert "scf.if" in source


def test_any_sync_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_vote_kernel(),
        "any_sync",
        "cooperative/any_sync",
    )

    assert "tl.any_sync" not in source
    assert "Unsupported" not in source
    assert "arith.ori" in source
    assert "scf.if" in source


def test_all_sync_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _all_sync_kernel(),
        "all_sync",
        "cooperative/all_sync",
    )

    assert "tl.all_sync" not in source
    assert "Unsupported" not in source
    assert "arith.andi" in source
    assert "scf.if" in source


def test_ballot_sync_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _ballot_kernel(),
        "ballot_sync",
        "cooperative/ballot_sync",
    )

    assert "tl.ballot_sync" not in source
    assert "Unsupported" not in source
    assert "arith.shli" in source
    assert "arith.ori" in source
    assert "scf.if" in source


def test_ballot_full_mask_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _ballot_full_mask_kernel(),
        "ballot_full_mask",
        "cooperative/ballot_full_mask",
    )

    assert "tl.ballot" not in source
    assert "Unsupported" not in source
    assert "arith.shli" in source
    assert "arith.ori" in source
    assert "scf.if" in source


def test_activemask_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _activemask_kernel(),
        "activemask",
        "cooperative/activemask",
    )

    assert "tl.activemask" not in source
    assert "Unsupported" not in source
    assert "arith.shli" in source
    assert "arith.ori" in source


def test_syncthreads_or_lowers_by_serialized_thread_replay():
    source = lower_tilelang_prim_to_mlir(
        _syncthreads_or_kernel(),
        "syncthreads_or",
        "cooperative/syncthreads_or",
    )

    assert "tl.syncthreads_or" not in source
    assert "Unsupported" not in source
    assert "scf.for" in source
    assert "arith.ori" in source


def test_shfl_sync_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_sync_kernel(),
        "shfl_sync",
        "cooperative/shfl_sync",
    )

    assert "tl.shfl_sync" not in source
    assert "Unsupported" not in source
    assert "scf.if" in source
    assert "arith.shli" in source
    assert "memref.load" in source


def test_shfl_sync_dynamic_source_lane_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_sync_dynamic_source_lane_kernel(),
        "shfl_sync_dynamic_source_lane",
        "cooperative/shfl_sync_dynamic_source_lane",
    )

    assert "tl.shfl_sync" not in source
    assert "Unsupported" not in source
    assert "arith.remui" in source
    assert "memref.load" in source


def test_shfl_sync_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_sync_local_value_kernel(),
        "shfl_sync_local_value",
        "cooperative/shfl_sync_local_value",
    )

    assert "tl.shfl_sync" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source


def test_shfl_sync_guarded_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_sync_guarded_local_value_kernel(),
        "shfl_sync_guarded_local_value",
        "cooperative/shfl_sync_guarded_local_value",
    )

    assert "tl.shfl_sync" not in source
    assert "Unsupported" not in source
    assert "scf.if" in source


def test_shfl_sync_loop_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_sync_loop_local_value_kernel(),
        "shfl_sync_loop_local_value",
        "cooperative/shfl_sync_loop_local_value",
    )

    assert "tl.shfl_sync" not in source
    assert "Unsupported" not in source
    assert "arith.addi" in source


def test_shfl_sync_loop_float_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_sync_loop_float_local_value_kernel(),
        "shfl_sync_loop_float_local_value",
        "cooperative/shfl_sync_loop_float_local_value",
    )

    assert "tl.shfl_sync" not in source
    assert "Unsupported" not in source
    assert "arith.addf" in source


def test_shfl_sync_dynamic_loop_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_sync_dynamic_loop_local_value_kernel(),
        "shfl_sync_dynamic_loop_local_value",
        "cooperative/shfl_sync_dynamic_loop_local_value",
    )

    assert "tl.shfl_sync" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source
    assert "scf.for" in source


def test_shfl_xor_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_xor_kernel(),
        "shfl_xor",
        "cooperative/shfl_xor",
    )

    assert "tl.shfl_xor_sync" not in source
    assert "Unsupported" not in source
    assert "arith.xori" in source
    assert "scf.if" in source


def test_shfl_xor_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_xor_local_value_kernel(),
        "shfl_xor_local_value",
        "cooperative/shfl_xor_local_value",
    )

    assert "tl.shfl_xor_sync" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source


def test_shfl_xor_loop_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_xor_loop_local_value_kernel(),
        "shfl_xor_loop_local_value",
        "cooperative/shfl_xor_loop_local_value",
    )

    assert "tl.shfl_xor_sync" not in source
    assert "Unsupported" not in source
    assert "arith.addi" in source


def test_shfl_xor_dynamic_loop_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_xor_dynamic_loop_local_value_kernel(),
        "shfl_xor_dynamic_loop_local_value",
        "cooperative/shfl_xor_dynamic_loop_local_value",
    )

    assert "tl.shfl_xor_sync" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source
    assert "scf.for" in source


def test_shfl_down_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_down_kernel(),
        "shfl_down",
        "cooperative/shfl_down",
    )

    assert "tl.shfl_down_sync" not in source
    assert "Unsupported" not in source
    assert "arith.addi" in source
    assert "scf.if" in source


def test_shfl_down_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_down_local_value_kernel(),
        "shfl_down_local_value",
        "cooperative/shfl_down_local_value",
    )

    assert "tl.shfl_down_sync" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source


def test_shfl_up_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_up_kernel(),
        "shfl_up",
        "cooperative/shfl_up",
    )

    assert "tl.shfl_up_sync" not in source
    assert "Unsupported" not in source
    assert "arith.subi" in source
    assert "scf.if" in source


def test_shfl_up_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _shfl_up_local_value_kernel(),
        "shfl_up_local_value",
        "cooperative/shfl_up_local_value",
    )

    assert "tl.shfl_up_sync" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source


def test_builtin_tvm_warp_shuffle_up_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _builtin_warp_shuffle_up_kernel(),
        "builtin_warp_shuffle_up",
        "cooperative/builtin_warp_shuffle_up",
    )

    assert "tir.tvm_warp_shuffle_up" not in source
    assert "Unsupported" not in source
    assert "arith.subi" in source
    assert "scf.if" in source
    assert "memref.load" in source


def test_warp_reduce_sum_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_kernel(),
        "warp_reduce_sum",
        "cooperative/warp_reduce_sum",
    )

    assert "tl.warp_reduce_sum" not in source
    assert "Unsupported" not in source
    assert "arith.addi" in source
    assert "scf.if" in source


def test_warp_reduce_sum_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_local_value_kernel(),
        "warp_reduce_sum_local_value",
        "cooperative/warp_reduce_sum_local_value",
    )

    assert "tl.warp_reduce_sum" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source
    assert "scf.if" in source


def test_warp_reduce_sum_loop_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_loop_local_value_kernel(),
        "warp_reduce_sum_loop_local_value",
        "cooperative/warp_reduce_sum_loop_local_value",
    )

    assert "tl.warp_reduce_sum" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source
    assert source.count("scf.for") >= 2


def test_warp_reduce_sum_dynamic_loop_local_value_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_dynamic_loop_local_value_kernel(),
        "warp_reduce_sum_dynamic_loop_local_value",
        "cooperative/warp_reduce_sum_dynamic_loop_local_value",
    )

    assert "tl.warp_reduce_sum" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source
    assert source.count("scf.for") >= 2


def test_warp_reduce_max_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_max_kernel(),
        "warp_reduce_max",
        "cooperative/warp_reduce_max",
    )

    assert "tl.warp_reduce_max" not in source
    assert "Unsupported" not in source
    assert "arith.select" in source
    assert "scf.if" in source


def test_warp_reduce_min_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_min_kernel(),
        "warp_reduce_min",
        "cooperative/warp_reduce_min",
    )

    assert "tl.warp_reduce_min" not in source
    assert "Unsupported" not in source
    assert "arith.select" in source
    assert "scf.if" in source


def test_warp_reduce_bitand_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_bitand_kernel(),
        "warp_reduce_bitand",
        "cooperative/warp_reduce_bitand",
    )

    assert "tl.warp_reduce_bitand" not in source
    assert "Unsupported" not in source
    assert "arith.andi" in source
    assert "scf.if" in source


def test_warp_reduce_bitor_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_bitor_kernel(),
        "warp_reduce_bitor",
        "cooperative/warp_reduce_bitor",
    )

    assert "tl.warp_reduce_bitor" not in source
    assert "Unsupported" not in source
    assert "arith.ori" in source
    assert "scf.if" in source


def test_warp_reduce_bitxor_lowers_by_serialized_warp_replay():
    with pytest.raises(AttributeError, match="warp_reduce_bitxor"):
        _warp_reduce_bitxor_kernel()


def test_warp_reduce_max_local_vector_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_max_local_vector_kernel(),
        "warp_reduce_max_local_vector",
        "cooperative/warp_reduce_max_local_vector",
    )

    assert "tl.warp_reduce_max" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source
    assert "arith.select" in source


def test_warp_reduce_max_float_local_vector_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_max_float_local_vector_kernel(),
        "warp_reduce_max_float_local_vector",
        "cooperative/warp_reduce_max_float_local_vector",
    )

    assert "tl.warp_reduce_max" not in source
    assert "Unsupported" not in source
    assert "memref.load" in source
    assert "arith.select" in source


def test_warp_reduce_sum_thread_index_helper_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_sum_thread_index_helper_kernel(),
        "warp_reduce_sum_thread_index_helper",
        "cooperative/warp_reduce_sum_thread_index_helper",
    )

    assert "tl.warp_reduce_sum" not in source
    assert "Unsupported" not in source
    assert "arith.remui" in source
    assert "scf.if" in source


def test_warp_reduce_sum_self_store_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_sum_self_store_kernel(),
        "warp_reduce_sum_self_store",
        "cooperative/warp_reduce_sum_self_store",
    )

    assert "tl.warp_reduce_sum" not in source
    assert "Unsupported" not in source
    assert "memref.store" in source
    assert "scf.if" in source


def test_warp_reduce_max_self_store_lowers_by_serialized_warp_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_max_self_store_kernel(),
        "warp_reduce_max_self_store",
        "cooperative/warp_reduce_max_self_store",
    )

    assert "tl.warp_reduce_max" not in source
    assert "Unsupported" not in source
    assert "memref.store" in source
    assert "arith.select" in source


def test_warp_reduce_sum_self_store_shared_phase_lowers_by_serialized_replay():
    source = lower_tilelang_prim_to_mlir(
        _warp_reduce_sum_self_store_shared_phase_kernel(),
        "warp_reduce_sum_self_store_shared_phase",
        "cooperative/warp_reduce_sum_self_store_shared_phase",
    )

    assert "tl.warp_reduce_sum" not in source
    assert "tvm_storage_sync" not in source
    assert "Unsupported" not in source
    assert source.count("scf.for") >= 2
    assert "memref.subview" in source
    assert "memref.store" in source


def test_sync_warp_lowers_by_splitting_serial_thread_phases():
    source = lower_tilelang_prim_to_mlir(
        _sync_warp_kernel(),
        "sync_warp",
        "cooperative/sync_warp",
    )

    assert "tl.sync_warp" not in source
    assert source.count("scf.for") >= 2
    assert "arith.addf" in source


def test_lane_dependent_sync_warp_loop_lowers_by_warp_round_replay():
    source = lower_tilelang_prim_to_mlir(
        _lane_dependent_sync_warp_loop_kernel(),
        "lane_dependent_sync_warp_loop",
        "cooperative/lane_dependent_sync_warp_loop",
    )

    assert "tl.sync_warp" not in source
    assert source.count("scf.for") >= 2
    assert "memref.alloca" in source
    assert "arith.addf" in source


def test_let_wrapped_lane_dependent_sync_warp_loop_lowers_by_warp_round_replay():
    source = lower_tilelang_prim_to_mlir(
        _let_wrapped_lane_dependent_sync_warp_loop_kernel(),
        "let_wrapped_lane_dependent_sync_warp_loop",
        "cooperative/let_wrapped_lane_dependent_sync_warp_loop",
    )

    assert "tl.sync_warp" not in source
    assert source.count("scf.for") >= 2
    assert "arith.andi" in source
    assert "memref.alloca" in source
    assert "arith.addf" in source


def test_shared_alloc_lowers_with_cta_scoped_buffer_without_sync():
    source = lower_tilelang_prim_to_mlir(
        _shared_alloc_kernel(),
        "shared_alloc",
        "cooperative/shared_alloc",
    )

    assert "tvm_storage_sync" not in source
    assert "memref.alloca" in source
    assert "memref.store" in source
    assert "memref.load" in source
    assert "scf.for" in source


def test_shared_alloc_sync_threads_lowers_with_cta_scoped_buffer():
    source = lower_tilelang_prim_to_mlir(
        _shared_alloc_sync_kernel(),
        "shared_alloc_sync",
        "cooperative/shared_alloc_sync",
    )

    assert "tvm_storage_sync" not in source
    assert "memref.alloca" in source
    assert source.count("scf.for") >= 2


def test_thread_invariant_shared_alloc_lowers_with_cta_scoped_buffer():
    source = lower_tilelang_prim_to_mlir(
        _thread_invariant_shared_alloc_kernel(),
        "thread_invariant_shared_alloc",
        "cooperative/thread_invariant_shared_alloc",
    )

    assert "tvm_storage_sync" not in source
    assert "memref.alloca" in source
    assert "scf.parallel" in source


def test_thread_invariant_shared_sync_phases_execute_once_per_cta():
    source = lower_tilelang_prim_to_mlir(
        _thread_invariant_shared_alloc_sync_kernel(),
        "thread_invariant_shared_alloc_sync",
        "cooperative/thread_invariant_shared_alloc_sync",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("scf.parallel") == 3
    # The entire thread-invariant launch runs once, including all barriers.
    assert source.count("scf.for") == 2


def test_mixed_shared_local_alloc_sync_threads_lowers_when_local_is_phase_private():
    source = lower_tilelang_prim_to_mlir(
        _mixed_shared_local_sync_kernel(),
        "mixed_shared_local_sync",
        "cooperative/mixed_shared_local_sync",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("memref.alloca") >= 2
    assert source.count("scf.for") >= 2


def test_local_alloc_cross_sync_lowers_with_thread_private_backing():
    source = lower_tilelang_prim_to_mlir(
        _local_alloc_cross_sync_kernel(),
        "local_alloc_cross_sync",
        "cooperative/local_alloc_cross_sync",
    )

    assert "tvm_storage_sync" not in source
    assert "memref.subview" in source
    assert source.count("memref.alloca") >= 2
    assert source.count("scf.for") >= 2
    assert "arith.addf" in source


def test_thread_invariant_serial_loop_sync_threads_lowers_by_iteration_phases():
    source = lower_tilelang_prim_to_mlir(
        _serial_loop_sync_threads_kernel(),
        "serial_loop_sync_threads",
        "cooperative/serial_loop_sync_threads",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("scf.for") >= 3
    assert "memref.alloca" in source
    assert "memref.load" in source
    assert "arith.addf" in source


def test_thread_invariant_conditional_sync_threads_lowers_by_iteration_phases():
    source = lower_tilelang_prim_to_mlir(
        _serial_loop_conditional_sync_threads_kernel(),
        "serial_loop_conditional_sync_threads",
        "cooperative/serial_loop_conditional_sync_threads",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("scf.for") >= 3
    assert "scf.if" in source
    assert "memref.load" in source


def test_thread_partitioned_common_loop_with_leading_sync_lowers_by_iteration_phases():
    source = lower_tilelang_prim_to_mlir(
        _thread_partitioned_common_loop_leading_sync_kernel(),
        "thread_partitioned_common_loop_leading_sync",
        "cooperative/thread_partitioned_common_loop_leading_sync",
    )

    assert "tvm_storage_sync" not in source
    assert "Unsupported" not in source
    assert "scf.if" in source
    assert source.count("scf.for") >= 3
    assert source.count("memref.store") >= 2


def test_nested_split_serial_loop_sync_threads_lowers_by_iteration_phases():
    source = lower_tilelang_prim_to_mlir(
        _nested_split_serial_loop_sync_threads_kernel(),
        "nested_split_serial_loop_sync_threads",
        "cooperative/nested_split_serial_loop_sync_threads",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("scf.for") >= 4
    assert "memref.alloca" in source
    assert "memref.load" in source
    assert "arith.addf" in source


def test_single_branch_common_loop_with_leading_sync_lowers_by_iteration_phases():
    source = lower_tilelang_prim_to_mlir(
        _single_branch_common_loop_leading_sync_kernel(),
        "single_branch_common_loop_leading_sync",
        "cooperative/single_branch_common_loop_leading_sync",
    )

    assert "tvm_storage_sync" not in source
    assert "Unsupported" not in source
    assert "scf.if" in source
    assert source.count("scf.for") >= 3
    assert source.count("memref.store") >= 2


def test_let_wrapped_sync_threads_lowers_by_iteration_phases():
    source = lower_tilelang_prim_to_mlir(
        _let_wrapped_sync_threads_kernel(),
        "let_wrapped_sync_threads",
        "cooperative/let_wrapped_sync_threads",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("scf.for") >= 3
    assert "arith.addf" in source
    assert "memref.load" in source


def test_thread_invariant_trailing_sync_threads_lowers_by_iteration_phases():
    source = lower_tilelang_prim_to_mlir(
        _serial_loop_trailing_sync_threads_kernel(),
        "serial_loop_trailing_sync_threads",
        "cooperative/serial_loop_trailing_sync_threads",
    )

    assert "tvm_storage_sync" not in source
    assert source.count("scf.for") >= 3
    assert "memref.alloca" in source
    assert "memref.load" in source
    assert "arith.addf" in source


def test_single_block_sync_grid_lowers_as_serialized_phase_boundary():
    source = lower_tilelang_prim_to_mlir(
        _single_block_sync_grid_kernel(),
        "single_block_sync_grid",
        "cooperative/single_block_sync_grid",
    )

    assert "tl.sync_grid" not in source
    assert "memref.alloca" in source
    assert source.count("scf.for") >= 2
    assert "arith.addf" in source


def test_single_thread_cooperative_expressions_lower_to_scalar_equivalents():
    source = lower_tilelang_prim_to_mlir(
        _single_thread_cooperative_expr_kernel(),
        "single_thread_cooperative_expr",
        "cooperative/single_thread_cooperative_expr",
    )

    assert "tl.any_sync" not in source
    assert "tl.all_sync" not in source
    assert "tl.ballot" not in source
    assert "tl.activemask" not in source
    assert "tl.match_any_sync" not in source
    assert "tl.match_all_sync" not in source
    assert "tl.tl_shuffle_elect" not in source
    assert "arith.constant 1" in source
    assert "arith.cmpi" in source
