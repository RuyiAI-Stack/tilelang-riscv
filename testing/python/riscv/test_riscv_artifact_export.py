from __future__ import annotations

import numpy as np
import pytest
import torch

from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.riscv import (
    RiscvKernelAdapter,
    build_qemu_executable,
    emit_asm,
    emit_llvm_ir,
    emit_mlir,
    emit_object,
    load_host_module,
    lower_to_llvm_dialect_mlir,
)
from tilelang.tladapter.toolchain import ToolchainNotFoundError


pytest.importorskip("tilelang.tladapter._native")


def _build_mlir_module(source: str, global_symbol: str):
    func = tvm.script.from_source(source).with_attr("global_symbol", global_symbol)
    mod = tvm.IRModule({global_symbol: func})
    target = tvm.target.Target("riscv")
    return func, tvm.ffi.get_global_func("target.build.tilelang_riscv")(mod, target)


def _build_copy_module():
    return _build_mlir_module(
        """
# from tvm.script import tir as T
@T.prim_func
def copy(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        with T.block("copy"):
            vi = T.axis.spatial(4, i)
            B[vi] = A[vi]
""",
        "copy",
    )


def _real_mlir_module_or_skip():
    func, rt_mod = _build_copy_module()
    source = rt_mod.inspect_source()
    if "Placeholder MLIR module" in source:
        pytest.skip("vendored MLIR lowering is disabled in this build")
    return func, rt_mod


def test_riscv_export_helpers_emit_all_artifacts(tmp_path):
    _, rt_mod = _real_mlir_module_or_skip()

    mlir_path = tmp_path / "copy.mlir"
    ll_path = tmp_path / "copy.ll"
    asm_path = tmp_path / "copy.s"
    obj_path = tmp_path / "copy.o"

    try:
        mlir_source = emit_mlir(rt_mod, mlir_path)
        llvm_ir = emit_llvm_ir(rt_mod, ll_path)
        asm_text = emit_asm(rt_mod, asm_path)
        obj_bytes = emit_object(rt_mod, obj_path)
    except ToolchainNotFoundError as err:
        pytest.skip(str(err))

    assert mlir_path.read_text() == mlir_source
    assert "func.func @copy(" in mlir_source

    assert ll_path.read_text() == llvm_ir
    assert "define void @copy(" in llvm_ir

    assert asm_path.read_text() == asm_text
    assert ".globl copy" in asm_text or ".globl\tcopy" in asm_text

    assert obj_path.read_bytes() == obj_bytes
    assert obj_path.stat().st_size > 0


def test_riscv_fp8_matmul_legalizes_conversions_after_linalg_lowering():
    source = """
module {
  func.func @fp8_matmul(
      %a: memref<1x1xf8E4M3FN>,
      %b: memref<1x1xf8E4M3FN>,
      %c: memref<1x1xf32>) {
    linalg.matmul ins(%a, %b : memref<1x1xf8E4M3FN>, memref<1x1xf8E4M3FN>)
                  outs(%c : memref<1x1xf32>)
    return
  }
}
"""

    llvm_mlir = lower_to_llvm_dialect_mlir(source)

    assert "tilelang_riscv_f8e4m3fn_to_f32" in llvm_mlir
    assert "llvm.fpext" not in llvm_mlir


def test_riscv_fp4_constants_preserve_all_encodings():
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    values += [-v for v in values]
    statements = []
    for i, value in enumerate(values):
        statements.extend(
            [
                f"%i{i} = arith.constant {i} : index",
                f"%v{i} = arith.constant {value:.6e} : f4E2M1FN",
                f"memref.store %v{i}, %out[%i{i}] : memref<16xf4E2M1FN>",
            ]
        )
    source = "module {\n func.func @fp4_constants(%out: memref<16xf4E2M1FN>) {\n"
    source += "\n".join(statements) + "\n return\n }\n}\n"
    library = load_host_module(source)
    try:
        out = np.full(16, -1, dtype=np.int8)
        library(out)
        np.testing.assert_array_equal(out, np.arange(16, dtype=np.int8))
    finally:
        library.close()


def test_riscv_legalizes_copysign_and_ctpop_without_external_helpers():
    source = """
module {
  func.func @math_legalization(%magnitude: f32, %sign: f32, %bits: i32) -> (f32, i32) {
    %signed = math.copysign %magnitude, %sign : f32
    %count = math.ctpop %bits : i32
    return %signed, %count : f32, i32
  }
}
"""

    llvm_mlir = lower_to_llvm_dialect_mlir(source)

    assert "math.copysign" not in llvm_mlir
    assert "math.ctpop" not in llvm_mlir
    assert "llvm.bitcast" in llvm_mlir
    assert "llvm.and" in llvm_mlir
    assert "llvm.lshr" in llvm_mlir


def test_riscv_host_sim_executes_copy_on_native_cpu(tmp_path):
    _, rt_mod = _real_mlir_module_or_skip()
    so_path = tmp_path / "copy.so"

    try:
        library = load_host_module(rt_mod, path=so_path)
    except ToolchainNotFoundError as err:
        pytest.skip(str(err))

    data = np.arange(4, dtype=np.float32)
    out = np.zeros_like(data)
    library(data, out)

    assert so_path.is_file()
    np.testing.assert_array_equal(out, data)


def test_riscv_qemu_builder_emits_freestanding_elf(tmp_path):
    _, rt_mod = _real_mlir_module_or_skip()
    exe_path = tmp_path / "copy.qemu.elf"
    data = np.arange(4, dtype=np.float32)
    out = np.zeros_like(data)

    try:
        built = build_qemu_executable(rt_mod, data, out, path=exe_path)
    except ToolchainNotFoundError as err:
        pytest.skip(str(err))

    assert built == exe_path
    assert exe_path.is_file()
    assert exe_path.read_bytes().startswith(b"\x7fELF")


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_riscv_unsigned_memref_preserves_high_bits_and_output_writes(bits):
    _, rt_mod = _build_mlir_module(
        f"""
# from tvm.script import tir as T
@T.prim_func
def unsigned_shift(A: T.Buffer((4,), "uint{bits}"), B: T.Buffer((4,), "uint{bits}")):
    for i in T.serial(4):
        B[i] = T.shift_right(A[i], T.uint{bits}(1))
""",
        "unsigned_shift",
    )
    data = np.array([0, 1, 1 << (bits - 1), (1 << bits) - 1], dtype=f"uint{bits}")
    out = np.zeros_like(data)
    library = load_host_module(rt_mod)
    try:
        library(data, out)
        np.testing.assert_array_equal(out, data >> 1)
        with pytest.raises(TypeError, match="Expected"):
            library(data.astype(np.float32), out)
    finally:
        library.close()


def test_riscv_kernel_adapter_executes_copy_on_cpu_torch():
    func, rt_mod = _real_mlir_module_or_skip()
    params = []
    for param in func.params:
        if param in func.buffer_map:
            params.append(KernelParam.from_buffer(func.buffer_map[param]))
        else:
            params.append(KernelParam.from_var(param))

    try:
        adapter = RiscvKernelAdapter(
            params=params,
            result_idx=[1],
            target="riscv",
            func_or_mod=func,
            rt_mod=rt_mod,
        )
    except ToolchainNotFoundError as err:
        pytest.skip(str(err))

    data = torch.arange(4, dtype=torch.float32)
    out = adapter(data)
    adapter.close()

    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, data)
