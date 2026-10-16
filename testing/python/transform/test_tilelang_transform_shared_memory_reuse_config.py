"""The shared-memory reuse switch must affect the actual optimization pipeline."""

import pytest
import tilelang
import tilelang.language as T
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


@pytest.mark.parametrize("disable", [False, True])
def test_shared_memory_reuse_switch(disable):
    @T.prim_func
    def shared_copy(x: T.Tensor((32,), "float32"), y: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            a = T.alloc_shared((32,), "float32")
            for i in T.Parallel(32):
                a[i] = x[i]
            T.sync_threads()
            for i in T.Parallel(32):
                y[i] = a[i] + 1
            T.sync_threads()
            b = T.alloc_shared((32,), "float32")
            for i in T.Parallel(32):
                b[i] = x[i] + 2
            T.sync_threads()
            for i in T.Parallel(32):
                y[i] += b[i]

    target = tilelang.tvm.target.Target("cuda -arch=sm_80", host="llvm")
    with (
        target,
        tilelang.transform.PassContext(
            config={
                tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: disable,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            }
        ),
    ):
        module = tilelang.tvm.IRModule({"shared_copy": shared_copy})
        module = OptimizeForTarget(LowerAndLegalize(module, target), target)
    allocations = []
    for func in module.functions.values():
        tilelang.tvm.tir.stmt_functor.post_order_visit(
            func.body, lambda node: allocations.append(node) if isinstance(node, tilelang.tvm.tir.Allocate) else None
        )
    shared = [node for node in allocations if node.buffer_var.type_annotation.storage_scope.startswith("shared")]
    assert len(shared) == 1
    # Both paths must remain one legal dynamic allocation. Disjoint lifetimes
    # share 128 bytes by default and occupy distinct 128-byte regions if disabled.
    assert int(shared[0].extents[0]) * shared[0].dtype.bits // 8 == (256 if disable else 128)
