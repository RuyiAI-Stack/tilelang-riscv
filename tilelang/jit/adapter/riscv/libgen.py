"""Helpers for MLIR -> LLVM -> RISC-V artifact generation."""

from __future__ import annotations

import os
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tilelang.tladapter.toolchain import resolve_tool
from tilelang.tladapter.utils import Pipeline


DEFAULT_RISCV_TRIPLE = os.environ.get("TILELANG_RISCV_TRIPLE", "riscv64-unknown-linux-gnu")


_F8_EXTF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.extf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*(?P<src>f8E4M3FN(?:UZ)?)\s+to\s+(?P<dst>f16|bf16|f32)\s*$"
)
_F8_TRUNCF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.truncf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*(?P<src>f16|bf16|f32)\s+to\s+(?P<dst>f8E4M3FN(?:UZ)?)\s*$"
)
_F4_EXTF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.extf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*f4E2M1FN\s+to\s+f32\s*$"
)
_F4_TRUNCF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.truncf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*f32\s+to\s+f4E2M1FN\s*$"
)
_F4_CONSTANT_RE = re.compile(
    r"^(?P<prefix>\s*%[\w\d_.$-]+\s*=\s*arith\.constant\s+)"
    r"(?P<value>\S+)\s*:\s*f4E2M1FN\s*$"
)
_MATH_COPYSIGN_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*math\.copysign\s+"
    r"(?P<magnitude>%[\w\d_.$-]+)\s*,\s*(?P<sign>%[\w\d_.$-]+)\s*:\s*"
    r"(?P<dtype>f32|f64)\s*$"
)
_MATH_CTPOP_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*math\.ctpop\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*(?P<dtype>i8|i16|i32|i64)\s*$"
)


def _f8_helper_suffix(dtype: str) -> str:
    return dtype.lower()


def _ssa_suffix(value: str, suffix: str) -> str:
    base = value[1:] if value.startswith("%") else value
    base = base.replace(".", "_").replace("$", "_").replace("-", "_")
    return f"%{suffix}_{base}"


def _legalize_low_precision_conversions_for_riscv(mlir_source: str) -> str:
    """Lower MLIR FP8 convert ops to explicit byte-level helper calls.

    The LLVM dialect lowers FP8 memrefs to byte storage on the RISC-V host path, but
    upstream MLIR does not legalize bare arith.extf/truncf for these FP8 types to
    LLVM. Keep FP8 storage in MLIR and only replace scalar conversions.
    """

    helper_decls: set[str] = set()
    rewritten: list[str] = []
    for line in mlir_source.splitlines():
        f4_constant = _F4_CONSTANT_RE.match(line)
        if f4_constant:
            literal = f4_constant.group("value")
            if literal.startswith("0x"):
                bits = int(literal, 16)
            else:
                value = float(literal)
                magnitude = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0).index(abs(value))
                bits = magnitude | (8 if math.copysign(1.0, value) < 0 else 0)
            rewritten.append(f"{f4_constant.group('prefix')}{bits} : i8")
            continue
        ext_match = _F8_EXTF_RE.match(line)
        if ext_match:
            indent = ext_match.group("indent")
            result = ext_match.group("result")
            value = ext_match.group("value")
            src = ext_match.group("src")
            dst = ext_match.group("dst")
            suffix = _f8_helper_suffix(src)
            bits = _ssa_suffix(result, "bits")
            helper = f"tilelang_riscv_{suffix}_to_f32"
            helper_decls.add(f"  func.func private @{helper}(i8) -> f32")
            rewritten.append(f"{indent}{bits} = arith.bitcast {value} : {src} to i8")
            decoded = result if dst == "f32" else _ssa_suffix(result, "decoded_f32")
            rewritten.append(f"{indent}{decoded} = func.call @{helper}({bits}) : (i8) -> f32")
            if dst != "f32":
                rewritten.append(f"{indent}{result} = arith.truncf {decoded} : f32 to {dst}")
            continue

        trunc_match = _F8_TRUNCF_RE.match(line)
        if trunc_match:
            indent = trunc_match.group("indent")
            result = trunc_match.group("result")
            value = trunc_match.group("value")
            src = trunc_match.group("src")
            dst = trunc_match.group("dst")
            suffix = _f8_helper_suffix(dst)
            bits = _ssa_suffix(result, "bits")
            helper = f"tilelang_riscv_f32_to_{suffix}"
            helper_decls.add(f"  func.func private @{helper}(f32) -> i8")
            if src != "f32":
                # FP16/BF16 values widen exactly; only the FP8 helper rounds.
                widened = _ssa_suffix(result, "widened_f32")
                rewritten.append(f"{indent}{widened} = arith.extf {value} : {src} to f32")
                value = widened
            rewritten.append(f"{indent}{bits} = func.call @{helper}({value}) : (f32) -> i8")
            rewritten.append(f"{indent}{result} = arith.bitcast {bits} : i8 to {dst}")
            continue

        f4_ext_match = _F4_EXTF_RE.match(line)
        if f4_ext_match:
            indent = f4_ext_match.group("indent")
            result = f4_ext_match.group("result")
            value = f4_ext_match.group("value")
            helper = "tilelang_riscv_f4e2m1fn_to_f32"
            helper_decls.add(f"  func.func private @{helper}(i8) -> f32")
            rewritten.append(f"{indent}{result} = func.call @{helper}({value}) : (i8) -> f32")
            continue

        f4_trunc_match = _F4_TRUNCF_RE.match(line)
        if f4_trunc_match:
            indent = f4_trunc_match.group("indent")
            result = f4_trunc_match.group("result")
            value = f4_trunc_match.group("value")
            helper = "tilelang_riscv_f32_to_f4e2m1fn"
            helper_decls.add(f"  func.func private @{helper}(f32) -> i8")
            rewritten.append(f"{indent}{result} = func.call @{helper}({value}) : (f32) -> i8")
            continue

        rewritten.append(line)

    output: list[str] = []
    inserted = False
    for line in rewritten:
        output.append(line)
        if not inserted and line.strip() == "module {":
            output.extend(sorted(helper_decls))
            inserted = True
    if not inserted:
        output = sorted(helper_decls) + output
    return ("\n".join(output) + ("\n" if mlir_source.endswith("\n") else "")).replace("f4E2M1FN", "i8")


def _legalize_unsupported_math_ops_for_riscv(mlir_source: str) -> str:
    """Expand scalar math ops not covered by this MLIR toolchain's conversions."""

    rewritten: list[str] = []
    for line in mlir_source.splitlines():
        copysign_match = _MATH_COPYSIGN_RE.match(line)
        if copysign_match:
            indent = copysign_match.group("indent")
            result = copysign_match.group("result")
            magnitude = copysign_match.group("magnitude")
            sign = copysign_match.group("sign")
            dtype = copysign_match.group("dtype")
            integer_dtype = "i32" if dtype == "f32" else "i64"
            abs_mask = 2 ** (int(integer_dtype[1:]) - 1) - 1
            sign_mask = -(2 ** (int(integer_dtype[1:]) - 1))
            magnitude_bits = _ssa_suffix(result, "copysign_magnitude_bits")
            sign_bits = _ssa_suffix(result, "copysign_sign_bits")
            abs_mask_value = _ssa_suffix(result, "copysign_abs_mask")
            sign_mask_value = _ssa_suffix(result, "copysign_sign_mask")
            absolute_bits = _ssa_suffix(result, "copysign_absolute_bits")
            signed_bits = _ssa_suffix(result, "copysign_signed_bits")
            combined_bits = _ssa_suffix(result, "copysign_combined_bits")
            rewritten.extend(
                [
                    f"{indent}{magnitude_bits} = arith.bitcast {magnitude} : {dtype} to {integer_dtype}",
                    f"{indent}{sign_bits} = arith.bitcast {sign} : {dtype} to {integer_dtype}",
                    f"{indent}{abs_mask_value} = arith.constant {abs_mask} : {integer_dtype}",
                    f"{indent}{sign_mask_value} = arith.constant {sign_mask} : {integer_dtype}",
                    f"{indent}{absolute_bits} = arith.andi {magnitude_bits}, {abs_mask_value} : {integer_dtype}",
                    f"{indent}{signed_bits} = arith.andi {sign_bits}, {sign_mask_value} : {integer_dtype}",
                    f"{indent}{combined_bits} = arith.ori {absolute_bits}, {signed_bits} : {integer_dtype}",
                    f"{indent}{result} = arith.bitcast {combined_bits} : {integer_dtype} to {dtype}",
                ]
            )
            continue

        ctpop_match = _MATH_CTPOP_RE.match(line)
        if ctpop_match:
            indent = ctpop_match.group("indent")
            result = ctpop_match.group("result")
            value = ctpop_match.group("value")
            dtype = ctpop_match.group("dtype")
            width = int(dtype[1:])
            masks = {
                "pair": int("55" * (width // 8), 16),
                "quad": int("33" * (width // 8), 16),
                "nibble": int("0f" * (width // 8), 16),
            }
            pair_mask = _ssa_suffix(result, "ctpop_pair_mask")
            pair_shift_amount = _ssa_suffix(result, "ctpop_pair_shift_amount")
            pair_shift = _ssa_suffix(result, "ctpop_pair_shift")
            pair_bits = _ssa_suffix(result, "ctpop_pair_bits")
            pair_sum = _ssa_suffix(result, "ctpop_pair_sum")
            quad_mask = _ssa_suffix(result, "ctpop_quad_mask")
            quad_low = _ssa_suffix(result, "ctpop_quad_low")
            quad_shift_amount = _ssa_suffix(result, "ctpop_quad_shift_amount")
            quad_shift = _ssa_suffix(result, "ctpop_quad_shift")
            quad_high = _ssa_suffix(result, "ctpop_quad_high")
            quad_sum = _ssa_suffix(result, "ctpop_quad_sum")
            nibble_shift_amount = _ssa_suffix(result, "ctpop_nibble_shift_amount")
            nibble_shift = _ssa_suffix(result, "ctpop_nibble_shift")
            nibble_sum = _ssa_suffix(result, "ctpop_nibble_sum")
            nibble_mask = _ssa_suffix(result, "ctpop_nibble_mask")
            current = _ssa_suffix(result, "ctpop_nibbles")
            rewritten.extend(
                [
                    f"{indent}{pair_mask} = arith.constant {masks['pair']} : {dtype}",
                    f"{indent}{pair_shift_amount} = arith.constant 1 : {dtype}",
                    f"{indent}{pair_shift} = arith.shrui {value}, {pair_shift_amount} : {dtype}",
                    f"{indent}{pair_bits} = arith.andi {pair_shift}, {pair_mask} : {dtype}",
                    f"{indent}{pair_sum} = arith.subi {value}, {pair_bits} : {dtype}",
                    f"{indent}{quad_mask} = arith.constant {masks['quad']} : {dtype}",
                    f"{indent}{quad_low} = arith.andi {pair_sum}, {quad_mask} : {dtype}",
                    f"{indent}{quad_shift_amount} = arith.constant 2 : {dtype}",
                    f"{indent}{quad_shift} = arith.shrui {pair_sum}, {quad_shift_amount} : {dtype}",
                    f"{indent}{quad_high} = arith.andi {quad_shift}, {quad_mask} : {dtype}",
                    f"{indent}{quad_sum} = arith.addi {quad_low}, {quad_high} : {dtype}",
                    f"{indent}{nibble_shift_amount} = arith.constant 4 : {dtype}",
                    f"{indent}{nibble_shift} = arith.shrui {quad_sum}, {nibble_shift_amount} : {dtype}",
                    f"{indent}{nibble_sum} = arith.addi {quad_sum}, {nibble_shift} : {dtype}",
                    f"{indent}{nibble_mask} = arith.constant {masks['nibble']} : {dtype}",
                    f"{indent}{current} = arith.andi {nibble_sum}, {nibble_mask} : {dtype}",
                ]
            )
            for shift in (8, 16, 32):
                if shift >= width:
                    continue
                shift_amount = _ssa_suffix(result, f"ctpop_shift_amount_{shift}")
                shifted = _ssa_suffix(result, f"ctpop_shift_{shift}")
                summed = _ssa_suffix(result, f"ctpop_sum_{shift}")
                rewritten.append(f"{indent}{shift_amount} = arith.constant {shift} : {dtype}")
                rewritten.append(f"{indent}{shifted} = arith.shrui {current}, {shift_amount} : {dtype}")
                rewritten.append(f"{indent}{summed} = arith.addi {current}, {shifted} : {dtype}")
                current = summed
            final_mask = _ssa_suffix(result, "ctpop_final_mask")
            rewritten.append(f"{indent}{final_mask} = arith.constant {2 * width - 1} : {dtype}")
            rewritten.append(f"{indent}{result} = arith.andi {current}, {final_mask} : {dtype}")
            continue

        rewritten.append(line)

    return "\n".join(rewritten) + ("\n" if mlir_source.endswith("\n") else "")


def _extract_mlir_source(value: Any) -> str:
    if isinstance(value, str):
        return value

    inspect_source = getattr(value, "inspect_source", None)
    if callable(inspect_source):
        return inspect_source()

    rt_mod = getattr(value, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "inspect_source"):
        return rt_mod.inspect_source()

    kernel_source = getattr(value, "kernel_source", None)
    if isinstance(kernel_source, str):
        return kernel_source

    raise TypeError(f"Cannot extract MLIR source from value of type {type(value)}")


def _write_text_if_requested(text: str, path: str | os.PathLike[str] | None) -> str:
    if path is not None:
        Path(path).write_text(text)
    return text


def _run_checked(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=input_text is not None,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "unknown tool failure"
        raise RuntimeError(f"`{' '.join(cmd)}` failed: {message}")
    return proc


_PRE_LOW_PRECISION_LEGALIZATION_PASSES = (
    "canonicalize",
    "cse",
    "func.func(convert-linalg-to-loops)",
    "canonicalize",
    "cse",
)

_POST_LOW_PRECISION_LEGALIZATION_PASSES = (
    "convert-scf-to-cf",
    "expand-strided-metadata",
    "lower-affine",
    "finalize-memref-to-llvm",
    "arith-expand{include-f4e2m1=true}",
    "convert-math-to-libm",
    "convert-math-to-llvm",
    "convert-arith-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
)


def _build_pipeline(pass_names: tuple[str, ...]) -> Pipeline:
    pipeline = Pipeline()
    for pass_name in pass_names:
        pipeline.add(pass_name)
    return pipeline


def build_debug_pipeline() -> Pipeline:
    return _build_pipeline(_PRE_LOW_PRECISION_LEGALIZATION_PASSES + _POST_LOW_PRECISION_LEGALIZATION_PASSES)


def lower_to_llvm_dialect_mlir(value: Any, *, pipeline: Pipeline | None = None) -> str:
    mlir_source = _extract_mlir_source(value)
    if pipeline is not None:
        mlir_source = _legalize_low_precision_conversions_for_riscv(mlir_source)
        mlir_source = _legalize_unsupported_math_ops_for_riscv(mlir_source)
        return pipeline.run(mlir_source)

    # Linalg lowering materializes the scalar FP8 extension operations used by
    # mixed-precision matmul. Legalize those operations only after they exist,
    # then continue with the LLVM conversion passes.
    mlir_source = _build_pipeline(_PRE_LOW_PRECISION_LEGALIZATION_PASSES).run(mlir_source)
    mlir_source = _legalize_low_precision_conversions_for_riscv(mlir_source)
    mlir_source = _legalize_unsupported_math_ops_for_riscv(mlir_source)
    return _build_pipeline(_POST_LOW_PRECISION_LEGALIZATION_PASSES).run(mlir_source)


def emit_mlir(value: Any, path: str | os.PathLike[str] | None = None) -> str:
    return _write_text_if_requested(_extract_mlir_source(value), path)


def emit_llvm_ir(
    value: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    pipeline: Pipeline | None = None,
) -> str:
    llvm_dialect_mlir = lower_to_llvm_dialect_mlir(value, pipeline=pipeline)
    proc = _run_checked([str(resolve_tool("mlir-translate")), "--mlir-to-llvmir"], input_text=llvm_dialect_mlir)
    return _write_text_if_requested(proc.stdout, path)


def _emit_llc_artifact(
    value: Any,
    *,
    filetype: str,
    triple: str = DEFAULT_RISCV_TRIPLE,
    path: str | os.PathLike[str] | None = None,
    pipeline: Pipeline | None = None,
) -> bytes | str:
    llvm_ir = emit_llvm_ir(value, pipeline=pipeline)
    with tempfile.TemporaryDirectory(prefix="tilelang-riscv-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        ll_path = temp_dir_path / "kernel.ll"
        out_path = temp_dir_path / f"kernel.{'s' if filetype == 'asm' else 'o'}"
        ll_path.write_text(llvm_ir)
        _run_checked(
            [
                str(resolve_tool("llc")),
                f"-mtriple={triple}",
                f"-filetype={filetype}",
                str(ll_path),
                "-o",
                str(out_path),
            ]
        )
        if filetype == "asm":
            text = out_path.read_text()
            if path is not None:
                Path(path).write_text(text)
            return text
        data = out_path.read_bytes()
        if path is not None:
            Path(path).write_bytes(data)
        return data


def emit_asm(
    value: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    triple: str = DEFAULT_RISCV_TRIPLE,
    pipeline: Pipeline | None = None,
) -> str:
    return _emit_llc_artifact(value, filetype="asm", triple=triple, path=path, pipeline=pipeline)


def emit_object(
    value: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    triple: str = DEFAULT_RISCV_TRIPLE,
    pipeline: Pipeline | None = None,
) -> bytes:
    return _emit_llc_artifact(value, filetype="obj", triple=triple, path=path, pipeline=pipeline)


__all__ = [
    "DEFAULT_RISCV_TRIPLE",
    "build_debug_pipeline",
    "emit_asm",
    "emit_llvm_ir",
    "emit_mlir",
    "emit_object",
    "lower_to_llvm_dialect_mlir",
]
