from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness import (
    T,
    assert_contains_all,
    build_mlir_from_source,
    lower_source_to_mlir,
    lower_tilelang_prim_to_mlir,
)


@dataclass(frozen=True)
class SourceCase:
    name: str
    symbol: str
    source: str
    expected: tuple[str, ...]


CASES = (
    SourceCase(
        name="float32x4_structured_add",
        symbol="vec4_structured_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_structured_add(
    A: T.Buffer((4,), "float32x4"),
    B: T.Buffer((4,), "float32x4"),
    C: T.Buffer((4,), "float32x4"),
):
    for i in T.serial(4):
        with T.block("add"):
            vi = T.axis.spatial(4, i)
            C[vi] = A[vi] + B[vi]
""",
        expected=(
            "func.func @vec4_structured_add",
            "memref<4xvector<4xf32>>",
            "linalg.generic",
            "arith.addf",
        ),
    ),
    SourceCase(
        name="int32x4_to_float32x4_cast",
        symbol="vec4_cast",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_cast(A: T.Buffer((4,), "int32x4"), B: T.Buffer((4,), "float32x4")):
    for i in T.serial(4):
        B[i] = T.cast(A[i], "float32x4")
""",
        expected=(
            "func.func @vec4_cast",
            "memref<4xvector<4xi32>>",
            "arith.sitofp",
            ": vector<4xi32> to vector<4xf32>",
            "memref.store",
        ),
    ),
    SourceCase(
        name="uint8x16_to_float32x16_cast",
        symbol="vec16_u8_to_f32_cast",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec16_u8_to_f32_cast(A: T.Buffer((4,), "uint8x16"), B: T.Buffer((4,), "float32x16")):
    for i in T.serial(4):
        B[i] = T.cast(A[i], "float32x16")
""",
        expected=(
            "func.func @vec16_u8_to_f32_cast",
            "memref<4xvector<16xi8>>",
            "memref<4xvector<16xf32>>",
            "arith.uitofp",
            ": vector<16xi8> to vector<16xf32>",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float16x4_add",
        symbol="vec4_f16_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_f16_add(
    A: T.Buffer((4,), "float16x4"),
    B: T.Buffer((4,), "float16x4"),
    C: T.Buffer((4,), "float16x4"),
):
    for i in T.serial(4):
        C[i] = A[i] + B[i]
""",
        expected=(
            "func.func @vec4_f16_add",
            "memref<4xvector<4xf16>>",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float64x2_add",
        symbol="vec2_f64_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec2_f64_add(
    A: T.Buffer((4,), "float64x2"),
    B: T.Buffer((4,), "float64x2"),
    C: T.Buffer((4,), "float64x2"),
):
    for i in T.serial(4):
        C[i] = A[i] + B[i]
""",
        expected=(
            "func.func @vec2_f64_add",
            "memref<4xvector<2xf64>>",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="int32x2_add",
        symbol="vec2_i32_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec2_i32_add(
    A: T.Buffer((4,), "int32x2"),
    B: T.Buffer((4,), "int32x2"),
    C: T.Buffer((4,), "int32x2"),
):
    for i in T.serial(4):
        with T.block("add"):
            vi = T.axis.spatial(4, i)
            C[vi] = A[vi] + B[vi]
""",
        expected=(
            "func.func @vec2_i32_add",
            "memref<4xvector<2xi32>>",
            "linalg.generic",
            "arith.addi",
        ),
    ),
    SourceCase(
        name="int16x8_add",
        symbol="vec8_i16_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec8_i16_add(
    A: T.Buffer((4,), "int16x8"),
    B: T.Buffer((4,), "int16x8"),
    C: T.Buffer((4,), "int16x8"),
):
    for i in T.serial(4):
        C[i] = A[i] + B[i]
""",
        expected=(
            "vec8_i16_add",
            "memref<4xvector<8xi16>>",
            "arith.addi",
            "memref.store",
        ),
    ),
    SourceCase(
        name="int8x16_add",
        symbol="vec16_i8_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec16_i8_add(
    A: T.Buffer((4,), "int8x16"),
    B: T.Buffer((4,), "int8x16"),
    C: T.Buffer((4,), "int8x16"),
):
    for i in T.serial(4):
        C[i] = A[i] + B[i]
""",
        expected=(
            "vec16_i8_add",
            "memref<4xvector<16xi8>>",
            "arith.addi",
            "memref.store",
        ),
    ),
    SourceCase(
        name="uint8x16_add",
        symbol="vec16_u8_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec16_u8_add(
    A: T.Buffer((4,), "uint8x16"),
    B: T.Buffer((4,), "uint8x16"),
    C: T.Buffer((4,), "uint8x16"),
):
    for i in T.serial(4):
        C[i] = A[i] + B[i]
""",
        expected=(
            "vec16_u8_add",
            "memref<4xvector<16xi8>>",
            "arith.addi",
            "memref.store",
        ),
    ),
    SourceCase(
        name="uint16x8_add",
        symbol="vec8_u16_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec8_u16_add(
    A: T.Buffer((4,), "uint16x8"),
    B: T.Buffer((4,), "uint16x8"),
    C: T.Buffer((4,), "uint16x8"),
):
    for i in T.serial(4):
        C[i] = A[i] + B[i]
""",
        expected=(
            "vec8_u16_add",
            "memref<4xvector<8xi16>>",
            "arith.addi",
            "memref.store",
        ),
    ),
    SourceCase(
        name="bfloat16x4_add",
        symbol="vec4_bf16_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_bf16_add(
    A: T.Buffer((4,), "bfloat16x4"),
    B: T.Buffer((4,), "bfloat16x4"),
    C: T.Buffer((4,), "bfloat16x4"),
):
    for i in T.serial(4):
        C[i] = A[i] + B[i]
""",
        expected=(
            "vec4_bf16_add",
            "memref<4xvector<4xbf16>>",
            "arith.addf",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float32x4_broadcast",
        symbol="vec4_broadcast",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_broadcast(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32x4")):
    for i in T.serial(4):
        B[i] = T.Broadcast(A[i], 4)
""",
        expected=(
            "func.func @vec4_broadcast",
            "memref<4xvector<4xf32>>",
            "vector.broadcast",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float32_to_float32x4_cast",
        symbol="scalar_to_vec_cast",
        source="""
# from tvm.script import tir as T
@T.prim_func
def scalar_to_vec_cast(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32x4")):
    for i in T.serial(4):
        B[i] = T.cast(A[i], "float32x4")
""",
        expected=(
            "func.func @scalar_to_vec_cast",
            "memref<4xf32>",
            "memref<4xvector<4xf32>>",
            "vector.broadcast",
            "memref.store",
        ),
    ),
    SourceCase(
        name="int32x4_ramp",
        symbol="vec4_ramp",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp(B: T.Buffer((4,), "int32x4")):
    for i in T.serial(4):
        B[i] = T.Ramp(i * 4, 1, 4)
""",
        expected=(
            "func.func @vec4_ramp",
            "memref<4xvector<4xi32>>",
            "arith.muli",
            "arith.addi",
            "vector.from_elements",
        ),
    ),
    SourceCase(
        name="uint32x4_ramp",
        symbol="vec4_u32_ramp",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_u32_ramp(B: T.Buffer((4,), "uint32x4")):
    for i in T.serial(4):
        B[i] = T.cast(T.Ramp(i * 4, 1, 4), "uint32x4")
""",
        expected=(
            "func.func @vec4_u32_ramp",
            "memref<4xvector<4xi32>>",
            "arith.muli",
            "vector.from_elements",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float32x4_shuffle",
        symbol="vec4_shuffle",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_shuffle(A: T.Buffer((4,), "float32x4"), B: T.Buffer((4,), "float32x4")):
    for i in T.serial(4):
        B[i] = T.Shuffle([A[i]], [3, 2, 1, 0])
""",
        expected=(
            "func.func @vec4_shuffle",
            "vector.extract",
            "vector.from_elements",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float32x4_ramp_slice_copy",
        symbol="vec4_ramp_slice_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp_slice_copy(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
    for i in T.serial(4):
        idx = T.Ramp(i * 4, 1, 4)
        B[idx] = A[idx]
""",
        expected=(
            "func.func @vec4_ramp_slice_copy",
            "memref<16xf32>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="float32x4_ramp_slice_stride2_copy",
        symbol="vec4_ramp_slice_stride2_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp_slice_stride2_copy(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")):
    for i in T.serial(4):
        idx = T.Ramp(i * 8, 2, 4)
        B[idx] = A[idx]
""",
        expected=(
            "func.func @vec4_ramp_slice_stride2_copy",
            "memref<32xf32>",
            "arith.muli",
            "arith.addi",
            "vector.from_elements",
            "vector.extract",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float32x4_ramp_slice_dynamic_stride_copy",
        symbol="vec4_ramp_slice_dynamic_stride_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp_slice_dynamic_stride_copy(
    A: T.Buffer((64,), "float32"),
    B: T.Buffer((64,), "float32"),
    stride: T.int32,
):
    for i in T.serial(4):
        idx = T.Ramp(i * 8, stride, 4)
        B[idx] = A[idx]
""",
        expected=(
            "func.func @vec4_ramp_slice_dynamic_stride_copy",
            "memref<64xf32>",
            "arith.index_cast",
            "arith.muli",
            "arith.addi",
            "vector.from_elements",
            "vector.extract",
            "memref.store",
        ),
    ),
    SourceCase(
        name="float32x4_ramp_slice_2d_last_dim_copy",
        symbol="vec4_ramp_slice_2d_last_dim_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp_slice_2d_last_dim_copy(
    A: T.Buffer((4, 16), "float32"),
    B: T.Buffer((4, 16), "float32"),
):
    for i, j in T.grid(4, 4):
        idx = T.Ramp(j * 4, 1, 4)
        B[i, idx] = A[i, idx]
""",
        expected=(
            "func.func @vec4_ramp_slice_2d_last_dim_copy",
            "memref<4x16xf32>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="float32x4_ramp_slice_3d_last_dim_copy",
        symbol="vec4_ramp_slice_3d_last_dim_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp_slice_3d_last_dim_copy(
    A: T.Buffer((2, 4, 16), "float32"),
    B: T.Buffer((2, 4, 16), "float32"),
):
    for b, i, j in T.grid(2, 4, 4):
        idx = T.Ramp(j * 4, 1, 4)
        B[b, i, idx] = A[b, i, idx]
""",
        expected=(
            "func.func @vec4_ramp_slice_3d_last_dim_copy",
            "memref<2x4x16xf32>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="float32x8_ramp_slice_copy",
        symbol="vec8_ramp_slice_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec8_ramp_slice_copy(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")):
    for i in T.serial(4):
        idx = T.Ramp(i * 8, 1, 8)
        B[idx] = A[idx]
""",
        expected=(
            "func.func @vec8_ramp_slice_copy",
            "memref<32xf32>",
            "vector<8xf32>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="float32x4_ramp_slice_add",
        symbol="vec4_ramp_slice_add",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp_slice_add(
    A: T.Buffer((16,), "float32"),
    B: T.Buffer((16,), "float32"),
    C: T.Buffer((16,), "float32"),
):
    for i in T.serial(4):
        idx = T.Ramp(i * 4, 1, 4)
        C[idx] = A[idx] + B[idx]
""",
        expected=(
            "func.func @vec4_ramp_slice_add",
            "memref<16xf32>",
            "vector.transfer_read",
            "arith.addf",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="float32x4_ramp_slice_local_alloc_copy",
        symbol="vec4_ramp_slice_local_alloc_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_ramp_slice_local_alloc_copy(
    A: T.Buffer((16,), "float32"),
    B: T.Buffer((16,), "float32"),
):
    Tbuf = T.alloc_buffer((16,), "float32")
    for i in T.serial(4):
        idx = T.Ramp(i * 4, 1, 4)
        Tbuf[idx] = A[idx]
    for i in T.serial(4):
        idx = T.Ramp(i * 4, 1, 4)
        B[idx] = Tbuf[idx]
""",
        expected=(
            "func.func @vec4_ramp_slice_local_alloc_copy",
            "memref.alloca() : memref<16xf32>",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
    SourceCase(
        name="float32x4_guarded_ramp_slice_copy",
        symbol="vec4_guarded_ramp_slice_copy",
        source="""
# from tvm.script import tir as T
@T.prim_func
def vec4_guarded_ramp_slice_copy(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
    for i in T.serial(4):
        if i < 2:
            idx = T.Ramp(i * 4, 1, 4)
            B[idx] = A[idx]
""",
        expected=(
            "func.func @vec4_guarded_ramp_slice_copy",
            "arith.cmpi slt",
            "scf.if",
            "vector.transfer_read",
            "vector.transfer_write",
            "{in_bounds = [true]}",
        ),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_vector_codegen_ops(case: SourceCase):
    source = lower_source_to_mlir(case.source, case.symbol, f"vector/{case.name}")
    assert_contains_all(source, case.expected)


def test_packed_scalar_view_roundtrip_codegen_op():
    @T.prim_func
    def packed_scalar_view_roundtrip(
        A: T.Tensor((4,), T.float32),
        B: T.Tensor((4,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            P = T.alloc_local((2,), T.float32x2)
            V = T.view(P, (4,), T.float32)
            for i in T.serial(4):
                V[i] = A[i]
            for i in T.serial(4):
                B[i] = V[i]

    source = lower_tilelang_prim_to_mlir(
        packed_scalar_view_roundtrip,
        "packed_scalar_view_roundtrip",
        "vector/packed_scalar_view_roundtrip",
    )

    assert_contains_all(
        source,
        (
            "func.func @packed_scalar_view_roundtrip",
            "memref.alloca() : memref<2xvector<2xf32>>",
            "arith.divui",
            "arith.remui",
            "vector.insert",
            "vector.extract",
            "memref.store",
        ),
    )


def test_packed_bfloat16_scalar_view_roundtrip_codegen_op():
    @T.prim_func
    def packed_bfloat16_scalar_view_roundtrip(
        A: T.Tensor((4,), T.bfloat16),
        B: T.Tensor((4,), T.bfloat16),
    ):
        with T.Kernel(1, threads=1):
            P = T.alloc_local((2,), T.bfloat16x2)
            V = T.view(P, (4,), T.bfloat16)
            for i in T.serial(4):
                V[i] = A[i]
            for i in T.serial(4):
                B[i] = V[i]

    source = lower_tilelang_prim_to_mlir(
        packed_bfloat16_scalar_view_roundtrip,
        "packed_bfloat16_scalar_view_roundtrip",
        "vector/packed_bfloat16_scalar_view_roundtrip",
    )

    assert_contains_all(
        source,
        (
            "func.func @packed_bfloat16_scalar_view_roundtrip",
            "memref.alloca() : memref<2xvector<2xbf16>>",
            "arith.divui",
            "arith.remui",
            "vector.insert",
            "vector.extract",
            "memref.store",
        ),
    )


def test_packed_float32x4_scalar_view_roundtrip_codegen_op():
    @T.prim_func
    def packed_float32x4_scalar_view_roundtrip(
        A: T.Tensor((8,), T.float32),
        B: T.Tensor((8,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            P = T.alloc_local((2,), T.float32x4)
            V = T.view(P, (8,), T.float32)
            for i in T.serial(8):
                V[i] = A[i]
            for i in T.serial(8):
                B[i] = V[i]

    source = lower_tilelang_prim_to_mlir(
        packed_float32x4_scalar_view_roundtrip,
        "packed_float32x4_scalar_view_roundtrip",
        "vector/packed_float32x4_scalar_view_roundtrip",
    )

    assert_contains_all(
        source,
        (
            "func.func @packed_float32x4_scalar_view_roundtrip",
            "memref.alloca() : memref<2xvector<4xf32>>",
            "arith.divui",
            "arith.remui",
            "vector.insert",
            "vector.extract",
            "memref.store",
        ),
    )


def test_packed_float32x4_scalar_view_ramp_slice_roundtrip_codegen_op():
    @T.prim_func
    def packed_float32x4_scalar_view_ramp_slice_roundtrip(
        A: T.Tensor((8,), T.float32),
        B: T.Tensor((8,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            P = T.alloc_local((2,), T.float32x4)
            V = T.view(P, (8,), T.float32)
            for i in T.serial(2):
                idx = T.Ramp(i * 4, 1, 4)
                V[idx] = A[idx]
            for i in T.serial(2):
                idx = T.Ramp(i * 4, 1, 4)
                B[idx] = V[idx]

    source = lower_tilelang_prim_to_mlir(
        packed_float32x4_scalar_view_ramp_slice_roundtrip,
        "packed_float32x4_scalar_view_ramp_slice_roundtrip",
        "vector/packed_float32x4_scalar_view_ramp_slice_roundtrip",
    )

    assert_contains_all(
        source,
        (
            "func.func @packed_float32x4_scalar_view_ramp_slice_roundtrip",
            "memref.alloca() : memref<2xvector<4xf32>>",
            "vector.from_elements",
            "vector.extract",
            "vector.insert",
            "arith.divui",
            "arith.remui",
            "memref.store",
        ),
    )


def test_packed_float32x4_scalar_view_2d_roundtrip_codegen_op():
    @T.prim_func
    def packed_float32x4_scalar_view_2d_roundtrip(
        A: T.Tensor((2, 4), T.float32),
        B: T.Tensor((2, 4), T.float32),
    ):
        with T.Kernel(1, threads=1):
            P = T.alloc_local((2,), T.float32x4)
            V = T.view(P, (2, 4), T.float32)
            for i, j in T.grid(2, 4):
                V[i, j] = A[i, j]
            for i, j in T.grid(2, 4):
                B[i, j] = V[i, j]

    source = lower_tilelang_prim_to_mlir(
        packed_float32x4_scalar_view_2d_roundtrip,
        "packed_float32x4_scalar_view_2d_roundtrip",
        "vector/packed_float32x4_scalar_view_2d_roundtrip",
    )

    assert_contains_all(
        source,
        (
            "func.func @packed_float32x4_scalar_view_2d_roundtrip",
            "memref.alloca() : memref<2xvector<4xf32>>",
            "arith.muli",
            "arith.addi",
            "arith.divui",
            "arith.remui",
            "vector.insert",
            "vector.extract",
            "memref.store",
        ),
    )


def test_packed_float32x4_scalar_view_2d_ramp_slice_roundtrip_codegen_op():
    @T.prim_func
    def packed_float32x4_scalar_view_2d_ramp_slice_roundtrip(
        A: T.Tensor((2, 4), T.float32),
        B: T.Tensor((2, 4), T.float32),
    ):
        with T.Kernel(1, threads=1):
            P = T.alloc_local((2,), T.float32x4)
            V = T.view(P, (2, 4), T.float32)
            for i in T.serial(2):
                idx = T.Ramp(0, 1, 4)
                V[i, idx] = A[i, idx]
            for i in T.serial(2):
                idx = T.Ramp(0, 1, 4)
                B[i, idx] = V[i, idx]

    source = lower_tilelang_prim_to_mlir(
        packed_float32x4_scalar_view_2d_ramp_slice_roundtrip,
        "packed_float32x4_scalar_view_2d_ramp_slice_roundtrip",
        "vector/packed_float32x4_scalar_view_2d_ramp_slice_roundtrip",
    )

    assert_contains_all(
        source,
        (
            "func.func @packed_float32x4_scalar_view_2d_ramp_slice_roundtrip",
            "memref.alloca() : memref<2xvector<4xf32>>",
            "vector.from_elements",
            "vector.extract",
            "vector.insert",
            "arith.muli",
            "arith.addi",
            "arith.divui",
            "arith.remui",
            "memref.store",
        ),
    )


@pytest.mark.parametrize(
    ("dtype", "mlir_type"),
    (
        ("float8_e3m4", "f8E3M4"),
        ("float8_e4m3", "f8E4M3"),
        ("float8_e4m3b11fnuz", "f8E4M3B11FNUZ"),
        ("float8_e4m3fn", "f8E4M3FN"),
        ("float8_e4m3fnuz", "f8E4M3FNUZ"),
        ("float8_e5m2", "f8E5M2"),
        ("float8_e5m2fnuz", "f8E5M2FNUZ"),
        ("float8_e8m0fnu", "f8E8M0FNU"),
        ("float4_e2m1fn", "f4E2M1FN"),
    ),
)
def test_low_precision_scalar_buffer_dtype_lowers(dtype: str, mlir_type: str):
    symbol = f"scalar_{dtype.replace('_', '')}_copy"
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def {symbol}(A: T.Buffer((4,), "{dtype}"), B: T.Buffer((4,), "{dtype}")):
    for i in T.serial(4):
        B[i] = A[i]
"""

    mlir = lower_source_to_mlir(source, symbol)
    assert f"memref<4x{mlir_type}>" in mlir
    assert "memref.load" in mlir
    assert "memref.store" in mlir


@pytest.mark.parametrize(
    ("dtype", "mlir_type"),
    (
        ("float8_e3m4x4", "vector<4xf8E3M4>"),
        ("float8_e4m3x4", "vector<4xf8E4M3>"),
        ("float8_e4m3fnx4", "vector<4xf8E4M3FN>"),
        ("float8_e4m3fnuzx4", "vector<4xf8E4M3FNUZ>"),
        ("float8_e5m2x4", "vector<4xf8E5M2>"),
        ("float8_e5m2fnuzx4", "vector<4xf8E5M2FNUZ>"),
        ("float4_e2m1fnx16", "vector<16xf4E2M1FN>"),
    ),
)
def test_low_precision_vector_buffer_dtype_lowers(dtype: str, mlir_type: str):
    symbol = f"{dtype.replace('_', '')}_copy"
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def {symbol}(A: T.Buffer((4,), "{dtype}"), B: T.Buffer((4,), "{dtype}")):
    for i in T.serial(4):
        B[i] = A[i]
"""

    mlir = lower_source_to_mlir(source, symbol)
    assert f"memref<4x{mlir_type}>" in mlir
    assert "memref.load" in mlir
    assert "memref.store" in mlir


@pytest.mark.parametrize(
    ("dtype", "mlir_type"),
    (
        ("float8_e4m3", "f8E4M3"),
        ("float8_e4m3fn", "f8E4M3FN"),
        ("float8_e4m3fnuz", "f8E4M3FNUZ"),
        ("float8_e5m2", "f8E5M2"),
        ("float8_e5m2fnuz", "f8E5M2FNUZ"),
        ("float4_e2m1fn", "f4E2M1FN"),
    ),
)
def test_low_precision_float_casts_lower(dtype: str, mlir_type: str):
    symbol = f"float32_to_{dtype.replace('_', '')}_cast"
    source = f"""
# from tvm.script import tir as T
@T.prim_func
def {symbol}(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "{dtype}"), C: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        B[i] = T.cast(A[i], "{dtype}")
        C[i] = T.cast(B[i], "float32")
"""

    mlir = lower_source_to_mlir(source, symbol)
    assert f"arith.truncf" in mlir
    assert f"to {mlir_type}" in mlir
    assert f"arith.extf" in mlir
    assert f"{mlir_type} to f32" in mlir


def test_non_last_ramp_dimension_is_rejected_by_tir(capfd):
    source = """
# from tvm.script import tir as T
@T.prim_func
def multi_ramp_probe(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
    idx_i = T.Ramp(0, 1, 4)
    idx_j = T.Ramp(0, 1, 4)
    B[idx_i, idx_j] = A[idx_i, idx_j]
"""

    with pytest.raises(Exception):
        build_mlir_from_source(source, "multi_ramp_probe")
    assert "Only the last index of a buffer access may be a vector type" in capfd.readouterr().err


def test_fixed_vector_mask_logic_and_select_codegen():
    source = """
# from tvm.script import tir as T
@T.prim_func
def vector_mask_logic(
    A: T.Buffer((1,), "float32x4"),
    B: T.Buffer((1,), "float32x4"),
    M: T.Buffer((1,), "int32x4"),
    N: T.Buffer((1,), "int32x4"),
    O: T.Buffer((1,), "float32x4"),
):
    O[0] = T.Select(
        T.Not(T.And(M[0] > T.Broadcast(0, 4), N[0] > T.Broadcast(0, 4))),
        A[0],
        T.Select(
            T.Or(M[0] > T.Broadcast(0, 4), N[0] > T.Broadcast(0, 4)),
            B[0],
            A[0],
        ),
    )
"""

    mlir = lower_source_to_mlir(source, "vector_mask_logic", "vector/mask_logic")
    assert_contains_all(
        mlir,
        (
            "vector<4xi1>",
            "arith.cmpi sgt",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "vector.broadcast",
            "arith.select",
        ),
    )


def test_scalar_select_condition_regression():
    source = """
# from tvm.script import tir as T
@T.prim_func
def scalar_select_condition(A: T.Buffer((1,), "float32"), B: T.Buffer((1,), "float32")):
    B[0] = T.Select(A[0] > T.float32(0), A[0], T.float32(0))
"""

    mlir = lower_source_to_mlir(source, "scalar_select_condition", "vector/scalar_select")
    assert "scf.if" in mlir
