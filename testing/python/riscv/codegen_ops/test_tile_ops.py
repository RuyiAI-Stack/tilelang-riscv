from __future__ import annotations

import pytest

from harness import (
    T,
    assert_contains_all,
    build_mlir_from_tilelang_prim,
    lower_tilelang_prim_to_mlir,
)


def _has_transpose_a_matmul(source: str) -> bool:
    return "linalg.matmul_transpose_a" in source or (
        "linalg.matmul indexing_maps" in source
        and "affine_map<(d0, d1, d2) -> (d2, d0)>" in source
        and "affine_map<(d0, d1, d2) -> (d2, d1)>" in source
    )


def _has_transpose_b_matmul(source: str) -> bool:
    return "linalg.matmul_transpose_b" in source or (
        "linalg.matmul indexing_maps" in source
        and "affine_map<(d0, d1, d2) -> (d0, d2)>" in source
        and "affine_map<(d0, d1, d2) -> (d1, d2)>" in source
    )


def test_tile_copy_codegen_op():
    @T.prim_func
    def tile_copy(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4,), "float32")
            T.copy(A, A_shared)
            T.copy(A_shared, B)

    source = lower_tilelang_prim_to_mlir(tile_copy, "tile_copy", "tile/copy")

    assert_contains_all(
        source,
        (
            "func.func @tile_copy",
            "memref.alloca() : memref<4xf32>",
            "memref.copy",
        ),
    )
    assert source.count("memref.copy") == 2


def test_tile_dynamic_copy_codegen_op():
    batch = T.dynamic("batch")

    @T.prim_func
    def tile_dynamic_copy(
        A: T.Tensor((batch,), "float32"),
        B: T.Tensor((batch,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((batch,), "float32")
            T.copy(A, A_shared)
            T.copy(A_shared, B)

    source = lower_tilelang_prim_to_mlir(tile_dynamic_copy, "tile_dynamic_copy", "tile/dynamic_copy")

    assert_contains_all(
        source,
        (
            "func.func @tile_dynamic_copy(%arg0: memref<?xf32>, %arg1: memref<?xf32>)",
            "memref.copy",
            "memref.subview",
        ),
    )
    assert source.count("memref.copy") == 2


def test_tile_copy_disable_tma_codegen_op():
    @T.prim_func
    def tile_copy_disable_tma(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            T.copy(A, B, disable_tma=True)

    source = lower_tilelang_prim_to_mlir(
        tile_copy_disable_tma,
        "tile_copy_disable_tma",
        "tile/copy_disable_tma",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_copy_disable_tma",
            "memref.copy",
        ),
    )
    assert "tl.tma" not in source
    assert "tl.tileop.tma" not in source
    assert "tl.create_tma" not in source
    assert "Unsupported CUDA/target-specific intrinsic" not in source


def test_tile_async_copy_codegen_op():
    @T.prim_func
    def tile_async_copy(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4,), "float32")
            T.async_copy(A, A_shared)
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.copy(A_shared, B)

    source = lower_tilelang_prim_to_mlir(
        tile_async_copy,
        "tile_async_copy",
        "tile/async_copy",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_async_copy",
            "memref.alloca() : memref<4xf32>",
            "memref.copy",
        ),
    )
    assert source.count("memref.copy") == 2
    assert "tl.tileop.async_copy" not in source
    assert "tir.ptx_commit_group" not in source
    assert "tir.ptx_wait_group" not in source
    assert "Unsupported CUDA/target-specific intrinsic" not in source


def test_tile_copy_strided_param_codegen_op():
    @T.prim_func
    def tile_copy_strided(
        A: T.Buffer((4,), "float32", strides=(2,)),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    source = lower_tilelang_prim_to_mlir(
        tile_copy_strided,
        "tile_copy_strided",
        "tile/copy_strided_param",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_copy_strided",
            "memref<4xf32, strided<[2]>>",
            "memref.copy",
        ),
    )


def test_tile_rank_reduced_copy_strided_param_codegen_op():
    @T.prim_func
    def tile_rank_reduced_copy_strided(
        A: T.Buffer((1, 4), "float32", strides=(8, 2)),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    source = lower_tilelang_prim_to_mlir(
        tile_rank_reduced_copy_strided,
        "tile_rank_reduced_copy_strided",
        "tile/rank_reduced_copy_strided_param",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_rank_reduced_copy_strided",
            "memref<1x4xf32, strided<[8, 2]>>",
            "memref.copy",
        ),
    )


def test_tile_decl_buffer_rank_change_copy_codegen_op():
    @T.prim_func
    def tile_decl_buffer_rank_change_copy(
        A: T.Buffer((2, 2, 2), "float32"),
        B: T.Tensor((2, 4), "float32"),
    ):
        A_flat = T.Buffer((2, 4), "float32", data=A.data, strides=(4, 1))
        with T.Kernel(1, threads=1):
            T.copy(A_flat, B)

    source = lower_tilelang_prim_to_mlir(
        tile_decl_buffer_rank_change_copy,
        "tile_decl_buffer_rank_change_copy",
        "tile/decl_buffer_rank_change_copy",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_decl_buffer_rank_change_copy",
            "memref<2x2x2xf32>",
            "memref.reinterpret_cast",
            "memref.copy",
        ),
    )


def test_tile_copy_strided_static_one_broadcast_param_codegen_op():
    @T.prim_func
    def tile_copy_strided_static_one_broadcast(
        A: T.Buffer((1, 4), "float32", strides=(8, 2)),
        B: T.Tensor((3, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    source = lower_tilelang_prim_to_mlir(
        tile_copy_strided_static_one_broadcast,
        "tile_copy_strided_static_one_broadcast",
        "tile/copy_strided_static_one_broadcast_param",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_copy_strided_static_one_broadcast",
            "memref<1x4xf32, strided<[8, 2]>>",
            "scf.for",
            "memref.load",
            "memref.store",
        ),
    )


def test_tile_copy_non_static_broadcast_is_rejected():
    @T.prim_func
    def tile_copy_non_static_broadcast(
        A: T.Tensor((2, 4), "float32"),
        B: T.Tensor((3, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    with pytest.raises(
        Exception,
        match="tl.copy currently requires matching extents, except for static-1 broadcast",
    ):
        build_mlir_from_tilelang_prim(
            tile_copy_non_static_broadcast,
            "tile_copy_non_static_broadcast",
        )


def test_tile_rank_reduced_copy_non_static_dim_is_rejected():
    @T.prim_func
    def tile_rank_reduced_copy_non_static_dim(
        A: T.Tensor((2, 4), "float32"),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    with pytest.raises(
        Exception,
        match="rank-changing tl.copy only supports dropping static-1 dimensions",
    ):
        build_mlir_from_tilelang_prim(
            tile_rank_reduced_copy_non_static_dim,
            "tile_rank_reduced_copy_non_static_dim",
        )


def test_tile_rank_reduced_copy_extent_mismatch_is_rejected():
    @T.prim_func
    def tile_rank_reduced_copy_extent_mismatch(
        A: T.Tensor((1, 5), "float32"),
        B: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    with pytest.raises(
        Exception,
        match="rank-reduced tl.copy extents must match after dropping static-1 dimensions",
    ):
        build_mlir_from_tilelang_prim(
            tile_rank_reduced_copy_extent_mismatch,
            "tile_rank_reduced_copy_extent_mismatch",
        )


def test_tile_gemm_codegen_op():
    @T.prim_func
    def tile_matmul(
        A: T.Tensor((4, 4), "float32"),
        B: T.Tensor((4, 4), "float32"),
        C: T.Tensor((4, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4, 4), "float32")
            B_shared = T.alloc_shared((4, 4), "float32")
            C_local = T.alloc_fragment((4, 4), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    source = lower_tilelang_prim_to_mlir(tile_matmul, "tile_matmul", "tile/gemm")

    assert_contains_all(
        source,
        (
            "func.func @tile_matmul",
            "linalg.matmul",
            "memref.alloca() : memref<4x4xf32>",
            "memref.copy",
        ),
    )
    assert source.count("memref.copy") >= 3


def test_tile_gemm_strided_param_codegen_op():
    @T.prim_func
    def tile_matmul_strided(
        A: T.Buffer((4, 4), "float32", strides=(8, 2)),
        B: T.Tensor((4, 4), "float32"),
        C: T.Tensor((4, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.gemm(A, B, C)

    source = lower_tilelang_prim_to_mlir(
        tile_matmul_strided,
        "tile_matmul_strided",
        "tile/gemm_strided_param",
    )

    assert_contains_all(
        source,
        (
            "func.func @tile_matmul_strided",
            "memref<4x4xf32, strided<[8, 2]>>",
            "linalg.matmul",
        ),
    )


def test_tile_gemm_transpose_b_strided_param_codegen_op():
    @T.prim_func
    def tile_matmul_transpose_b_strided(
        A: T.Tensor((2, 3), "float32"),
        B: T.Buffer((4, 3), "float32", strides=(8, 2)),
        C: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.gemm(A, B, C, transpose_B=True)

    source = lower_tilelang_prim_to_mlir(
        tile_matmul_transpose_b_strided,
        "tile_matmul_transpose_b_strided",
        "tile/gemm_transpose_b_strided_param",
    )

    assert "func.func @tile_matmul_transpose_b_strided" in source
    assert "memref<4x3xf32, strided<[8, 2]>>" in source
    assert _has_transpose_b_matmul(source)


def test_tile_gemm_transpose_a_strided_param_codegen_op():
    @T.prim_func
    def tile_matmul_transpose_a_strided(
        A: T.Buffer((3, 2), "float32", strides=(8, 2)),
        B: T.Tensor((3, 4), "float32"),
        C: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.gemm(A, B, C, transpose_A=True)

    source = lower_tilelang_prim_to_mlir(
        tile_matmul_transpose_a_strided,
        "tile_matmul_transpose_a_strided",
        "tile/gemm_transpose_a_strided_param",
    )

    assert "func.func @tile_matmul_transpose_a_strided" in source
    assert "memref<3x2xf32, strided<[8, 2]>>" in source
    assert _has_transpose_a_matmul(source)


def test_tile_gemm_transpose_ab_strided_param_codegen_op():
    @T.prim_func
    def tile_matmul_transpose_ab_strided(
        A: T.Buffer((3, 2), "float32", strides=(8, 2)),
        B: T.Buffer((4, 3), "float32", strides=(9, 2)),
        C: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            T.gemm(A, B, C, transpose_A=True, transpose_B=True)

    source = lower_tilelang_prim_to_mlir(
        tile_matmul_transpose_ab_strided,
        "tile_matmul_transpose_ab_strided",
        "tile/gemm_transpose_ab_strided_param",
    )

    assert "func.func @tile_matmul_transpose_ab_strided" in source
    assert "memref<3x2xf32, strided<[8, 2]>>" in source
    assert "memref<4x3xf32, strided<[9, 2]>>" in source
    assert "memref.alloca() : memref<2x3xf32>" in source
    assert _has_transpose_b_matmul(source)
    assert source.count("scf.for") >= 2
    assert source.count("memref.store") >= 1


def test_tile_gemm_transpose_b_codegen_op():
    @T.prim_func
    def tile_matmul_transpose_b(
        A: T.Tensor((2, 3), "float32"),
        B: T.Tensor((4, 3), "float32"),
        C: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((2, 3), "float32")
            B_shared = T.alloc_shared((4, 3), "float32")
            C_local = T.alloc_fragment((2, 4), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    source = lower_tilelang_prim_to_mlir(
        tile_matmul_transpose_b,
        "tile_matmul_transpose_b",
        "tile/gemm_transpose_b",
    )

    assert "func.func @tile_matmul_transpose_b" in source
    assert "memref<2x3xf32>" in source
    assert "memref<4x3xf32>" in source
    assert _has_transpose_b_matmul(source)


def test_tile_fixed_f32_max_reduction_uses_vector_reduction():
    @T.prim_func
    def tile_fixed_f32_max(A: T.Tensor((4,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            values = T.alloc_fragment((4,), "float32")
            result = T.alloc_fragment((1,), "float32")
            values[T.Ramp(0, 1, 4)] = A[T.Ramp(0, 1, 4)]
            T.reduce_max(values, result, dim=0, clear=True)
            B[0] = result[0]

    source = lower_tilelang_prim_to_mlir(
        tile_fixed_f32_max,
        "tile_fixed_f32_max",
        "tile/fixed_f32_max",
    )
    assert_contains_all(
        source,
        (
            "vector.transfer_read",
            "vector.reduction <maximumf>",
            "vector<4xf32> into f32",
        ),
    )


@pytest.mark.parametrize(
    ("dtype", "combining_kind", "mlir_type"),
    (("float64", "maximumf", "vector<4xf64> into f64"),),
)
def test_tile_fixed_typed_max_reduction_uses_vector_reduction(dtype, combining_kind, mlir_type):
    @T.prim_func
    def tile_fixed_typed_max(A: T.Tensor((4,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=1):
            values = T.alloc_fragment((4,), dtype)
            result = T.alloc_fragment((1,), dtype)
            values[T.Ramp(0, 1, 4)] = A[T.Ramp(0, 1, 4)]
            T.reduce_max(values, result, dim=0, clear=True)
            B[0] = result[0]

    source = lower_tilelang_prim_to_mlir(
        tile_fixed_typed_max,
        "tile_fixed_typed_max",
        f"tile/fixed_{dtype}_max",
    )
    assert_contains_all(
        source,
        (
            "vector.transfer_read",
            f"vector.reduction <{combining_kind}>",
            mlir_type,
        ),
    )


def test_tile_fixed_f32_max_clear_false_uses_scalar_fallback():
    @T.prim_func
    def tile_fixed_f32_max_accumulate(A: T.Tensor((4,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            values = T.alloc_fragment((4,), "float32")
            result = T.alloc_fragment((1,), "float32")
            values[T.Ramp(0, 1, 4)] = A[T.Ramp(0, 1, 4)]
            result[0] = B[0]
            T.reduce_max(values, result, dim=0, clear=False)
            B[0] = result[0]

    source = lower_tilelang_prim_to_mlir(
        tile_fixed_f32_max_accumulate,
        "tile_fixed_f32_max_accumulate",
        "tile/fixed_f32_max_accumulate",
    )
    assert "vector.reduction" not in source
    assert "arith.cmpf ogt" in source
