"""Helpers for compiling MLIR-backed kernels into native host libraries."""

from __future__ import annotations

import ctypes
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tilelang.tladapter.toolchain import ToolchainNotFoundError, resolve_llvm_root, resolve_tool

from .libgen import DEFAULT_RISCV_TRIPLE, emit_llvm_ir, emit_mlir, emit_object


@dataclass(frozen=True)
class _MemRefParam:
    shape: tuple[int | None, ...]
    dtype: np.dtype
    mlir_type: str

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_fp4(self) -> bool:
        return False


@dataclass(frozen=True)
class _ScalarParam:
    dtype_token: str
    ctype: type[ctypes._SimpleCData]
    mlir_type: str


@dataclass(frozen=True)
class _FunctionSignature:
    name: str
    params: tuple[_MemRefParam | _ScalarParam, ...]


class RiscvRunnerError(RuntimeError):
    """Raised when the freestanding RISC-V runner cannot be built or executed."""


class RiscvRunnerNotFoundError(RiscvRunnerError):
    """Raised when no runnable RISC-V simulator is available."""


_QEMU_OUTPUT_MAGIC = b"TLRVQ001"
_RETAINED_HOST_LIBRARY_TEMPDIRS: list[tempfile.TemporaryDirectory[str]] = []


def _run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "unknown tool failure"
        raise RuntimeError(f"`{' '.join(cmd)}` failed: {message}")
    return proc


def _run_checked_binary(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        message = stderr or stdout or "unknown tool failure"
        raise RiscvRunnerError(f"`{' '.join(cmd)}` failed: {message}")
    return proc


_PTR_TO_MEMREF_DECL_RE = re.compile(
    r"declare\s+\{\s*ptr,\s*ptr,\s*i64,\s*\[(?P<rank>\d+)\s+x\s+i64\],\s*"
    r"\[(?P=rank)\s+x\s+i64\]\s*\}\s+@(?P<name>tilelang_riscv_ptr_i64_to_memref_[^(]+)"
    r"\((?P<args>[^)]*)\)"
)


def _decode_ptr_to_memref_shape(symbol: str) -> list[int | None]:
    marker = "_r"
    rank_pos = symbol.rfind(marker)
    if rank_pos < 0:
        raise ValueError(f"Malformed pointer-backed memref helper symbol: {symbol}")
    rest = symbol[rank_pos + len(marker) :]
    pieces = rest.split("_")
    if not pieces or not pieces[0].isdigit():
        raise ValueError(f"Malformed pointer-backed memref helper rank: {symbol}")
    shape: list[int | None] = []
    i = 1
    while i < len(pieces):
        tag = pieces[i]
        if tag == "d":
            shape.append(None)
            i += 1
            continue
        if tag.startswith("s") and len(tag) > 1:
            shape.append(int(tag[1:]))
            i += 1
            continue
        if tag == "s" and i + 1 < len(pieces):
            shape.append(int(pieces[i + 1]))
            i += 2
            continue
        raise ValueError(f"Malformed pointer-backed memref helper shape: {symbol}")
    if len(shape) != int(pieces[0]):
        raise ValueError(f"Pointer-backed memref helper rank/shape mismatch: {symbol}")
    return shape


def _row_major_strides(shape_exprs: list[str]) -> list[str]:
    strides = ["1"] * len(shape_exprs)
    running = "1"
    for i in range(len(shape_exprs) - 1, -1, -1):
        strides[i] = running
        running = f"({running}) * ({shape_exprs[i]})"
    return strides


def _emit_ptr_to_memref_helpers(llvm_ir: str) -> str:
    definitions: list[str] = [
        "#include <stdint.h>",
        "",
    ]
    seen: set[str] = set()
    emitted_structs: set[str] = set()
    for match in _PTR_TO_MEMREF_DECL_RE.finditer(llvm_ir):
        symbol = match.group("name")
        if symbol in seen:
            continue
        seen.add(symbol)
        rank = int(match.group("rank"))
        shape = _decode_ptr_to_memref_shape(symbol)
        dyn_arg_count = sum(dim is None for dim in shape)
        struct_name = f"tilelang_riscv_memref_r{rank}"
        if struct_name not in emitted_structs:
            emitted_structs.add(struct_name)
            definitions.extend(
                [
                    f"struct {struct_name} {{",
                    "  void* allocated;",
                    "  void* aligned;",
                    "  int64_t offset;",
                    f"  int64_t sizes[{rank}];",
                    f"  int64_t strides[{rank}];",
                    "};",
                    "",
                ]
            )
        args = ["int64_t raw_addr"] + [f"int64_t d{i}" for i in range(dyn_arg_count)]
        shape_exprs: list[str] = []
        dyn_index = 0
        for dim in shape:
            if dim is None:
                shape_exprs.append(f"d{dyn_index}")
                dyn_index += 1
            else:
                shape_exprs.append(str(dim))
        stride_exprs = _row_major_strides(shape_exprs)
        definitions.extend(
            [
                f"struct {struct_name} {symbol}({', '.join(args)}) {{",
                f"  struct {struct_name} result;",
                "  result.allocated = (void*)(uintptr_t)raw_addr;",
                "  result.aligned = (void*)(uintptr_t)raw_addr;",
                "  result.offset = 0;",
            ]
        )
        for i, expr in enumerate(shape_exprs):
            definitions.append(f"  result.sizes[{i}] = {expr};")
        for i, expr in enumerate(stride_exprs):
            definitions.append(f"  result.strides[{i}] = {expr};")
        definitions.extend(
            [
                "  return result;",
                "}",
                "",
            ]
        )
    return "\n".join(definitions) if seen else ""


def _emit_f8_conversion_helpers(llvm_ir: str) -> str:
    """Emit E4M3 casts with nearest-even rounding and finite saturation."""
    needed = {
        "tilelang_riscv_f8e4m3fnuz_to_f32": "tilelang_riscv_f8e4m3fnuz_to_f32" in llvm_ir,
        "tilelang_riscv_f32_to_f8e4m3fnuz": "tilelang_riscv_f32_to_f8e4m3fnuz" in llvm_ir,
        "tilelang_riscv_f8e4m3fn_to_f32": "tilelang_riscv_f8e4m3fn_to_f32" in llvm_ir,
        "tilelang_riscv_f32_to_f8e4m3fn": "tilelang_riscv_f32_to_f8e4m3fn" in llvm_ir,
    }
    if not any(needed.values()):
        return ""

    definitions = [
        "#include <stdint.h>",
        "",
        "static float tilelang_riscv_pow2_i32(int exp) {",
        "  float value = 1.0f;",
        "  if (exp >= 0) {",
        "    for (int i = 0; i < exp; ++i) value *= 2.0f;",
        "  } else {",
        "    for (int i = 0; i < -exp; ++i) value *= 0.5f;",
        "  }",
        "  return value;",
        "}",
        "",
        "static float tilelang_riscv_decode_e4m3(uint8_t raw, int bias, int fnuz) {",
        '  if (fnuz ? raw == 0x80 : (raw & 0x7f) == 0x7f) return __builtin_nanf("");',
        "  if ((raw & 0x7f) == 0) return (raw & 0x80) ? -0.0f : 0.0f;",
        "  int sign = (raw & 0x80) ? -1 : 1;",
        "  int exp = (raw >> 3) & 0x0f;",
        "  int mant = raw & 0x07;",
        "  float value;",
        "  if (exp == 0) {",
        "    value = ((float)mant / 8.0f) * tilelang_riscv_pow2_i32(1 - bias);",
        "  } else {",
        "    value = (1.0f + (float)mant / 8.0f) * tilelang_riscv_pow2_i32(exp - bias);",
        "  }",
        "  return sign < 0 ? -value : value;",
        "}",
        "",
        "static uint8_t tilelang_riscv_encode_e4m3(float value, int bias, int fnuz) {",
        "  uint8_t sign = __builtin_signbit(value) ? 0x80 : 0;",
        "  if (!(value == value)) return fnuz ? 0x80 : (sign | 0x7f);",
        "  float magnitude = __builtin_fabsf(value);",
        "  uint8_t max_code = fnuz ? 0x7f : 0x7e;",
        "  float max_value = tilelang_riscv_decode_e4m3(max_code, bias, fnuz);",
        "  if (magnitude >= max_value) return sign | max_code;",
        "  float best_diff = magnitude;",
        "  uint8_t best = 0;",
        "  for (int code = 1; code <= max_code; ++code) {",
        "    float decoded = tilelang_riscv_decode_e4m3((uint8_t)code, bias, fnuz);",
        "    float diff = decoded - magnitude;",
        "    if (diff < 0.0f) diff = -diff;",
        "    if (diff < best_diff || (diff == best_diff && (code & 1) == 0 && (best & 1) != 0)) {",
        "      best_diff = diff;",
        "      best = (uint8_t)code;",
        "    }",
        "  }",
        "  return (fnuz && best == 0) ? 0 : (sign | best);",
        "}",
        "",
    ]
    if needed["tilelang_riscv_f8e4m3fnuz_to_f32"]:
        definitions.extend(
            [
                "float tilelang_riscv_f8e4m3fnuz_to_f32(uint8_t raw) {",
                "  return tilelang_riscv_decode_e4m3(raw, 8, 1);",
                "}",
                "",
            ]
        )
    if needed["tilelang_riscv_f32_to_f8e4m3fnuz"]:
        definitions.extend(
            [
                "uint8_t tilelang_riscv_f32_to_f8e4m3fnuz(float value) {",
                "  return tilelang_riscv_encode_e4m3(value, 8, 1);",
                "}",
                "",
            ]
        )
    if needed["tilelang_riscv_f8e4m3fn_to_f32"]:
        definitions.extend(
            [
                "float tilelang_riscv_f8e4m3fn_to_f32(uint8_t raw) {",
                "  return tilelang_riscv_decode_e4m3(raw, 7, 0);",
                "}",
                "",
            ]
        )
    if needed["tilelang_riscv_f32_to_f8e4m3fn"]:
        definitions.extend(
            [
                "uint8_t tilelang_riscv_f32_to_f8e4m3fn(float value) {",
                "  return tilelang_riscv_encode_e4m3(value, 7, 0);",
                "}",
                "",
            ]
        )
    return "\n".join(definitions)


def _emit_f4_conversion_helpers(llvm_ir: str) -> str:
    need_decode = "tilelang_riscv_f4e2m1fn_to_f32" in llvm_ir
    need_encode = "tilelang_riscv_f32_to_f4e2m1fn" in llvm_ir
    if not need_decode and not need_encode:
        return ""
    definitions = [
        "#include <stdint.h>",
        "",
        "float tilelang_riscv_f4e2m1fn_to_f32(uint8_t raw) {",
        "  raw &= 0x0f;",
        "  float sign = (raw & 0x08) ? -1.0f : 1.0f;",
        "  int exp = (raw >> 1) & 0x03;",
        "  int mant = raw & 0x01;",
        "  float value = 0.0f;",
        "  if (exp == 0) {",
        "    value = 0.5f * (float)mant;",
        "  } else {",
        "    float pow2 = exp == 1 ? 1.0f : (exp == 2 ? 2.0f : 4.0f);",
        "    value = (1.0f + 0.5f * (float)mant) * pow2;",
        "  }",
        "  return sign * value;",
        "}",
        "",
        "uint8_t tilelang_riscv_f32_to_f4e2m1fn(float value) {",
        "  float best_diff = 3.4028234663852886e38f;",
        "  uint8_t best = 0;",
        "  for (int code = 0; code < 16; ++code) {",
        "    float decoded = tilelang_riscv_f4e2m1fn_to_f32((uint8_t)code);",
        "    float diff = decoded - value;",
        "    if (diff < 0.0f) diff = -diff;",
        "    if (diff < best_diff || (diff == best_diff && (code & 1) == 0 && (best & 1) != 0)) {",
        "      best_diff = diff;",
        "      best = (uint8_t)code;",
        "    }",
        "  }",
        "  return best;",
        "}",
        "",
    ]
    return "\n".join(definitions)


def _emit_math_helpers(llvm_ir: str) -> str:
    if not re.search(r"\brsqrtf\b", llvm_ir):
        return ""
    return "\n".join(
        [
            "#include <math.h>",
            "",
            "float rsqrtf(float value) {",
            "  return 1.0f / sqrtf(value);",
            "}",
            "",
        ]
    )


def _emit_host_helper_source(llvm_ir: str) -> str:
    helper_parts = [
        _emit_ptr_to_memref_helpers(llvm_ir),
        _emit_f8_conversion_helpers(llvm_ir),
        _emit_f4_conversion_helpers(llvm_ir),
        _emit_math_helpers(llvm_ir),
    ]
    return "\n".join(part for part in helper_parts if part)


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())

    for char in text:
        if char in pairs:
            depth += 1
        elif char in closers:
            depth -= 1
        if char == delimiter and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _dtype_token_to_numpy(dtype_token: str) -> np.dtype:
    if dtype_token == "f4E2M1FN":
        return np.dtype(np.int8)
    if dtype_token == "f16":
        return np.dtype(np.float16)
    if dtype_token == "f32":
        return np.dtype(np.float32)
    if dtype_token == "f64":
        return np.dtype(np.float64)
    if dtype_token == "bf16":
        return np.dtype("bfloat16")
    if dtype_token in ("f8E4M3FN", "f8E5M2", "f8E4M3FNUZ", "f8E5M2FNUZ"):
        return np.dtype(np.uint8)
    if dtype_token == "i1":
        return np.dtype(np.bool_)
    if dtype_token == "index":
        return np.dtype(np.int64)
    if dtype_token.startswith("ui"):
        return np.dtype(f"uint{int(dtype_token[2:])}")
    if dtype_token.startswith("i"):
        return np.dtype(f"int{int(dtype_token[1:])}")
    raise TypeError(f"Unsupported MLIR dtype for host wrapper: {dtype_token}")


def _dtype_token_to_ctype(dtype_token: str) -> type[ctypes._SimpleCData]:
    mapping: dict[str, type[ctypes._SimpleCData]] = {
        "f32": ctypes.c_float,
        "f64": ctypes.c_double,
        "i1": ctypes.c_bool,
        "i8": ctypes.c_int8,
        "i16": ctypes.c_int16,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "ui8": ctypes.c_uint8,
        "ui16": ctypes.c_uint16,
        "ui32": ctypes.c_uint32,
        "ui64": ctypes.c_uint64,
        "index": ctypes.c_int64,
    }
    if dtype_token in mapping:
        return mapping[dtype_token]
    raise TypeError(f"Unsupported MLIR scalar dtype for host wrapper: {dtype_token}")


def _dtype_token_to_c_type(dtype_token: str) -> str:
    mapping = {
        "f32": "float",
        "f64": "double",
        "i1": "_Bool",
        "i8": "signed char",
        "i16": "short",
        "i32": "int",
        "i64": "long long",
        "ui8": "unsigned char",
        "ui16": "unsigned short",
        "ui32": "unsigned int",
        "ui64": "unsigned long long",
        "index": "long long",
    }
    if dtype_token in mapping:
        return mapping[dtype_token]
    raise TypeError(f"Unsupported MLIR scalar dtype for freestanding RISC-V runner: {dtype_token}")


def _parse_memref_type(type_text: str) -> _MemRefParam:
    assert type_text.startswith("memref<") and type_text.endswith(">")
    inner = type_text[len("memref<") : -1].strip()
    shape_and_dtype = _split_top_level(inner)[0]
    dims_and_dtype = [piece.strip() for piece in shape_and_dtype.split("x") if piece.strip()]
    if len(dims_and_dtype) == 1:
        shape: tuple[int | None, ...] = ()
        dtype_token = dims_and_dtype[0]
    else:
        raw_shape = dims_and_dtype[:-1]
        dtype_token = dims_and_dtype[-1]
        shape = tuple(None if dim == "?" else int(dim) for dim in raw_shape)
    return _MemRefParam(shape=shape, dtype=_dtype_token_to_numpy(dtype_token), mlir_type=type_text)


def _parse_param_type(type_text: str) -> _MemRefParam | _ScalarParam:
    normalized = type_text.strip()
    if normalized.startswith("memref<"):
        return _parse_memref_type(normalized)
    return _ScalarParam(dtype_token=normalized, ctype=_dtype_token_to_ctype(normalized), mlir_type=normalized)


def _parse_function_signatures(mlir_source: str) -> list[_FunctionSignature]:
    matches = list(re.finditer(r"func\.func\s+@(?P<name>[\w$.-]+)\((?P<params>[^)]*)\)\s*(?:->\s*[^({]+)?\{", mlir_source))
    signatures: list[_FunctionSignature] = []
    for match in matches:
        params_text = match.group("params").strip()
        params: list[_MemRefParam | _ScalarParam] = []
        if params_text:
            for param_text in _split_top_level(params_text):
                _, type_text = param_text.split(":", maxsplit=1)
                params.append(_parse_param_type(type_text))
        signatures.append(_FunctionSignature(name=match.group("name"), params=tuple(params)))
    generic_matches = re.finditer(
        r'"func\.func"\(\)\s*<\{(?P<attrs>.*?)\}>\s*\(',
        mlir_source,
        flags=re.DOTALL,
    )
    for match in generic_matches:
        attrs = match.group("attrs")
        name_match = re.search(r'sym_name\s*=\s*"(?P<name>[\w$.-]+)"', attrs)
        type_match = re.search(r"function_type\s*=\s*\((?P<params>.*?)\)\s*->", attrs, flags=re.DOTALL)
        if name_match is None or type_match is None:
            continue
        params_text = type_match.group("params").strip()
        params: list[_MemRefParam | _ScalarParam] = []
        if params_text:
            for type_text in _split_top_level(params_text):
                params.append(_parse_param_type(type_text))
        if not any(signature.name == name_match.group("name") for signature in signatures):
            signatures.append(_FunctionSignature(name=name_match.group("name"), params=tuple(params)))
    return signatures


def _select_signature(mlir_source: str, function_name: str | None) -> _FunctionSignature:
    signatures = _parse_function_signatures(mlir_source)
    if not signatures:
        raise ValueError("No `func.func` definitions found in the MLIR module")
    if function_name is None:
        return signatures[0]
    for signature in signatures:
        if signature.name == function_name:
            return signature
    raise ValueError(f"Function `{function_name}` not found in the MLIR module")


def resolve_host_triple() -> str:
    if os.environ.get("TILELANG_HOST_TRIPLE"):
        return os.environ["TILELANG_HOST_TRIPLE"]
    clang = resolve_tool("clang")
    proc = _run_checked([str(clang), "-dumpmachine"])
    return proc.stdout.strip()


def _normalize_memref_array(spec: _MemRefParam, value: Any) -> np.ndarray:
    array = np.asarray(value)
    # MLIR's signless integer storage also represents TIR unsigned buffers.
    # Preserve the bit pattern and shared storage (including output writes).
    if array.dtype.kind == "u" and spec.dtype.kind == "i" and array.dtype.itemsize == spec.dtype.itemsize:
        array = array.view(spec.dtype)
    if array.dtype != spec.dtype:
        raise TypeError(f"Expected {spec.dtype} for {spec.mlir_type}, but got {array.dtype}")
    if array.ndim != spec.rank:
        raise ValueError(f"Expected rank-{spec.rank} array for {spec.mlir_type}, but got rank {array.ndim}")
    if not array.flags.c_contiguous:
        raise ValueError("Execution currently requires C-contiguous NumPy arrays")
    expected_shape = _physical_memref_shape(spec)
    for expected_dim, actual_dim in zip(expected_shape, array.shape):
        if expected_dim is not None and expected_dim != actual_dim:
            raise ValueError(f"Expected shape {expected_shape} for {spec.mlir_type}, but got {tuple(int(dim) for dim in array.shape)}")
    return array


def _physical_memref_shape(spec: _MemRefParam) -> tuple[int | None, ...]:
    if not spec.is_fp4 or not spec.shape:
        return spec.shape
    shape = list(spec.shape)
    last = shape[-1]
    shape[-1] = None if last is None else (last + 1) // 2
    return tuple(shape)


def _logical_memref_shape(spec: _MemRefParam, array: np.ndarray) -> list[int]:
    shape = [int(dim) for dim in array.shape]
    if spec.is_fp4 and shape:
        shape[-1] *= 2
    return shape


def _logical_row_major_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = running
        running *= shape[i]
    return strides


def _normalize_scalar_value(spec: _ScalarParam, value: Any) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    try:
        return spec.ctype(scalar).value
    except Exception as err:  # pragma: no cover - ctypes error details are platform-specific
        raise TypeError(f"Cannot convert value {value!r} to scalar type {spec.mlir_type}") from err


def _format_c_scalar_literal(spec: _ScalarParam, value: Any) -> str:
    normalized = _normalize_scalar_value(spec, value)
    if spec.dtype_token in ("f32", "f64"):
        if not np.isfinite(normalized):
            raise ValueError("Freestanding RISC-V runner does not support NaN/Inf scalar literals")
        return float(normalized).hex()
    if spec.dtype_token == "i1":
        return "1" if normalized else "0"
    return str(int(normalized))


def _resolve_command_parts(command: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(command, str):
        parts = shlex.split(command)
    else:
        parts = [str(part) for part in command]
    if not parts:
        raise RiscvRunnerNotFoundError("Empty RISC-V runner command")
    resolved = shutil.which(parts[0]) if not os.path.isabs(parts[0]) else parts[0]
    if resolved is None:
        raise RiscvRunnerNotFoundError(f"RISC-V runner `{parts[0]}` not found. Install qemu-user or set TILELANG_RISCV_RUNNER.")
    return [resolved, *parts[1:]]


def resolve_riscv_runner(
    runner: str | list[str] | tuple[str, ...] | None = None,
    *,
    required: bool = True,
) -> list[str] | None:
    try:
        if runner is not None:
            cmd = _resolve_command_parts(runner)
        elif os.environ.get("TILELANG_RISCV_RUNNER"):
            cmd = _resolve_command_parts(os.environ["TILELANG_RISCV_RUNNER"])
        else:
            qemu = shutil.which("qemu-riscv64")
            if qemu is None:
                if not required:
                    return None
                raise RiscvRunnerNotFoundError(
                    'RISC-V runner `qemu-riscv64` not found. Install qemu-user or set `TILELANG_RISCV_RUNNER="spike pk"`.'
                )
            cmd = [qemu]
        if os.environ.get("TILELANG_RISCV_RUNNER_FLAGS"):
            cmd.extend(shlex.split(os.environ["TILELANG_RISCV_RUNNER_FLAGS"]))
        return cmd
    except RiscvRunnerNotFoundError:
        if required:
            raise
        return None


def resolve_riscv_linker(*, required: bool = True) -> Path | None:
    override = os.environ.get("TILELANG_RISCV_LINKER")
    if override:
        parts = shlex.split(override)
        if len(parts) != 1:
            raise ToolchainNotFoundError("TILELANG_RISCV_LINKER must point to a single linker executable")
        linker_path = Path(parts[0]).expanduser()
        if not linker_path.is_file():
            raise ToolchainNotFoundError(f"Configured RISC-V linker not found: {linker_path}")
        return linker_path.resolve()

    lld = resolve_tool("ld.lld", required=False)
    if lld is not None:
        return lld

    for linker_name in ("riscv64-unknown-linux-gnu-ld", "riscv64-linux-gnu-ld"):
        linker_path = shutil.which(linker_name)
        if linker_path:
            return Path(linker_path).resolve()

    system_ld = shutil.which("ld")
    if system_ld:
        proc = subprocess.run([system_ld, "-V"], text=True, capture_output=True, check=False)
        version_text = proc.stdout + proc.stderr
        if "elf64lriscv" in version_text:
            return Path(system_ld).resolve()

    if required:
        raise ToolchainNotFoundError(
            "No RISC-V linker found. Build the vendored LLVM toolchain with lld enabled or set TILELANG_RISCV_LINKER to a cross linker."
        )
    return None


def _resolve_riscv_gcc_root() -> Path | None:
    for env_name in ("TILELANG_RISCV_GCC_ROOT", "GCC_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate = Path(env_value).expanduser().resolve()
            if candidate.is_dir():
                return candidate

    for env_name in ("CC", "CXX"):
        compiler = os.environ.get(env_name)
        if not compiler:
            continue
        compiler_path = Path(shlex.split(compiler)[0]).expanduser()
        if compiler_path.is_file() and compiler_path.parent.name == "bin":
            candidate = compiler_path.parent.parent.resolve()
            if candidate.is_dir():
                return candidate

    default_root = Path("/opt/gcc-native")
    if default_root.is_dir():
        return default_root.resolve()
    return None


def _riscv_gcc_runtime_library_dirs(gcc_root: Path) -> list[Path]:
    abi = os.environ.get("TILELANG_RISCV_ABI")
    abi_names = [abi] if abi else ["lp64d", "lp64f", "lp64", "ilp32d", "ilp32f", "ilp32"]
    candidates: list[Path] = []
    for abi_name in abi_names:
        candidates.extend(
            [
                gcc_root / "lib64" / abi_name,
                gcc_root / "lib32" / abi_name,
                gcc_root / "lib" / "lib64" / abi_name,
                gcc_root / "lib" / "lib32" / abi_name,
                gcc_root / "sysroot" / "lib64" / abi_name,
                gcc_root / "sysroot" / "lib32" / abi_name,
                gcc_root / "sysroot" / "usr" / "lib64" / abi_name,
                gcc_root / "sysroot" / "usr" / "lib32" / abi_name,
            ]
        )

    runtime_dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if (candidate / "libgcc_s.so").is_file() and candidate not in seen:
            runtime_dirs.append(candidate)
            seen.add(candidate)

    if runtime_dirs:
        return runtime_dirs

    for runtime_lib in gcc_root.rglob("libgcc_s.so"):
        runtime_dir = runtime_lib.parent
        if runtime_dir not in seen:
            runtime_dirs.append(runtime_dir)
            seen.add(runtime_dir)
    return runtime_dirs


def _riscv_clang_flags(
    *,
    include_runtime_library_dirs: bool = False,
    include_abi: bool = True,
) -> list[str]:
    flags: list[str] = []
    gcc_root = _resolve_riscv_gcc_root()
    if gcc_root is not None:
        flags.append(f"--gcc-toolchain={gcc_root}")
        sysroot = gcc_root / "sysroot"
        if sysroot.is_dir():
            flags.append(f"--sysroot={sysroot}")
        if include_runtime_library_dirs:
            for runtime_dir in _riscv_gcc_runtime_library_dirs(gcc_root):
                flags.append(f"-L{runtime_dir}")
                flags.append(f"-Wl,-rpath,{runtime_dir}")
    if os.environ.get("TILELANG_RISCV_MARCH"):
        flags.append(f"-march={os.environ['TILELANG_RISCV_MARCH']}")
    if include_abi and os.environ.get("TILELANG_RISCV_ABI"):
        flags.append(f"-mabi={os.environ['TILELANG_RISCV_ABI']}")
    if os.environ.get("TILELANG_RISCV_CPU"):
        flags.append(f"-mcpu={os.environ['TILELANG_RISCV_CPU']}")
    if os.environ.get("TILELANG_RISCV_CLANG_FLAGS"):
        flags.extend(shlex.split(os.environ["TILELANG_RISCV_CLANG_FLAGS"]))
    return flags


def _infer_riscv_abi_flag(obj_path: Path) -> list[str]:
    llvm_readelf = resolve_tool("llvm-readelf", required=False)
    if llvm_readelf is None:
        return []
    proc = _run_checked([str(llvm_readelf), "-A", str(obj_path)])
    match = re.search(r"TagName:\s+arch\s+Value:\s+([^\s]+)", proc.stdout)
    if match is None:
        return []
    arch = re.sub(r"\d+p\d+", "", match.group(1))
    if "g" in arch or "d" in arch:
        return ["-mabi=lp64d"]
    if "f" in arch:
        return ["-mabi=lp64f"]
    return ["-mabi=lp64"]


def _freestanding_runner_source(signature: _FunctionSignature, args: tuple[Any, ...]) -> tuple[str, list[np.ndarray]]:
    if len(args) != len(signature.params):
        raise ValueError(f"Expected {len(signature.params)} arguments, but got {len(args)}")

    memref_arrays: list[np.ndarray] = []
    storage_decls: list[str] = []
    write_backs: list[str] = []
    call_args: list[str] = []

    for index, (spec, value) in enumerate(zip(signature.params, args)):
        arg_name = f"arg{index}"
        if isinstance(spec, _MemRefParam):
            array = _normalize_memref_array(spec, value)
            memref_arrays.append(array)
            raw_bytes = list(array.tobytes(order="C"))
            storage_size = max(len(raw_bytes), 1)
            init_bytes = ", ".join(str(byte) for byte in raw_bytes) if raw_bytes else "0"
            storage_decls.append(
                "static union {\n"
                "  unsigned long long align;\n"
                f"  unsigned char bytes[{storage_size}];\n"
                f"}} {arg_name}_storage = {{ .bytes = {{ {init_bytes} }} }};"
            )
            call_args.extend(
                [
                    f"(void *){arg_name}_storage.bytes",
                    f"(void *){arg_name}_storage.bytes",
                    "0",
                ]
            )
            call_args.extend(str(int(dim)) for dim in array.shape)
            call_args.extend(str(int(stride // array.dtype.itemsize)) for stride in array.strides)
            write_backs.append(f"  tl_write_all({arg_name}_storage.bytes, {array.nbytes}ul);")
        else:
            call_args.append(_format_c_scalar_literal(spec, value))

    prototype_params: list[str] = []
    for index, spec in enumerate(signature.params):
        arg_name = f"arg{index}"
        if isinstance(spec, _MemRefParam):
            prototype_params.extend(
                [
                    f"void *{arg_name}_allocated",
                    f"void *{arg_name}_aligned",
                    f"long long {arg_name}_offset",
                ]
            )
            prototype_params.extend(f"long long {arg_name}_size{dim}" for dim in range(spec.rank))
            prototype_params.extend(f"long long {arg_name}_stride{dim}" for dim in range(spec.rank))
        else:
            prototype_params.append(f"{_dtype_token_to_c_type(spec.dtype_token)} {arg_name}")
    prototype = ", ".join(prototype_params) if prototype_params else "void"
    symbol_name = signature.name.replace("\\", "\\\\").replace('"', '\\"')
    storage_block = "\n\n".join(storage_decls)
    call_expr = ", ".join(call_args)
    magic_bytes = ", ".join(str(byte) for byte in _QEMU_OUTPUT_MAGIC)
    write_block = "\n".join(write_backs)

    source = f"""typedef unsigned long tl_size;
typedef long tl_long;

extern void tilelang_kernel_entry({prototype}) __asm__("{symbol_name}");

static const unsigned char tl_magic[{len(_QEMU_OUTPUT_MAGIC)}] = {{{magic_bytes}}};

{storage_block}

static tl_long tl_syscall3(tl_long number, tl_long arg0, tl_long arg1, tl_long arg2) {{
  register tl_long a0 __asm__("a0") = arg0;
  register tl_long a1 __asm__("a1") = arg1;
  register tl_long a2 __asm__("a2") = arg2;
  register tl_long a7 __asm__("a7") = number;
  __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
  return a0;
}}

static void tl_exit(tl_long code) {{
  register tl_long a0 __asm__("a0") = code;
  register tl_long a7 __asm__("a7") = 93;
  __asm__ volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
  for (;;) {{
  }}
}}

static void tl_write_all(const void *buffer, tl_size size) {{
  const unsigned char *cursor = (const unsigned char *)buffer;
  while (size > 0) {{
    tl_long written = tl_syscall3(64, 1, (tl_long)cursor, (tl_long)size);
    if (written <= 0) {{
      tl_exit(3);
    }}
    cursor += (tl_size)written;
    size -= (tl_size)written;
  }}
}}

void _start(void) {{
  tilelang_kernel_entry({call_expr});
  tl_write_all(tl_magic, {len(_QEMU_OUTPUT_MAGIC)}ul);
{write_block}
  tl_exit(0);
}}
"""
    return source, memref_arrays


def build_qemu_executable(
    value: Any,
    *args: Any,
    function_name: str | None = None,
    path: str | os.PathLike[str],
    triple: str = DEFAULT_RISCV_TRIPLE,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> Path:
    mlir_source = emit_mlir(value)
    signature = _select_signature(mlir_source, function_name)
    source, _ = _freestanding_runner_source(signature, args)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tilelang-riscv-qemu-build-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        obj_path = temp_dir_path / f"{signature.name}.o"
        harness_path = temp_dir_path / f"{signature.name}_runner.c"
        linker_bin_dir = temp_dir_path / "linker-bin"
        linker_bin_dir.mkdir()
        linker_path = resolve_riscv_linker(required=True)
        os.symlink(linker_path, linker_bin_dir / "ld")
        emit_object(value, path=obj_path, triple=triple, pipeline=pipeline)
        harness_path.write_text(source)
        cmd = [
            str(resolve_tool("clang")),
            f"--target={triple}",
            f"-B{linker_bin_dir}",
            "-nostdlib",
            "-static",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-stack-protector",
            "-O2",
            str(harness_path),
            str(obj_path),
            "-Wl,-e,_start",
            "-o",
            str(out_path),
        ]
        cmd.extend(_infer_riscv_abi_flag(obj_path))
        cmd.extend(_riscv_clang_flags(include_abi=False))
        if clang_flags:
            cmd.extend(clang_flags)
        try:
            _run_checked(cmd)
        except RuntimeError as err:
            raise RiscvRunnerError(str(err)) from err
    return out_path


def build_host_shared_library(
    value: Any,
    path: str | os.PathLike[str],
    *,
    triple: str | None = None,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> Path:
    llvm_ir = emit_llvm_ir(value, pipeline=pipeline)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llvm_root = resolve_llvm_root()
    llvm_lib_dir = llvm_root / "lib"
    with tempfile.TemporaryDirectory(prefix="tilelang-host-compile-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        ll_path = temp_dir_path / f"{out_path.stem}.ll"
        ll_path.write_text(llvm_ir)
        helper_source = _emit_host_helper_source(llvm_ir)
        opt_level = os.environ.get("TILELANG_RISCV_CLANG_OPT_LEVEL", "2").lower()
        if opt_level not in {"0", "1", "2", "3", "s", "z"}:
            raise RiscvRunnerError("TILELANG_RISCV_CLANG_OPT_LEVEL must be one of 0, 1, 2, 3, s, or z")
        cmd = [
            str(resolve_tool("clang")),
            "-shared",
            "-fPIC",
            f"-O{opt_level}",
            "-Wno-override-module",
            "-x",
            "ir",
            str(ll_path),
            "-o",
            str(out_path),
            f"-L{llvm_lib_dir}",
            f"-Wl,-rpath,{llvm_lib_dir}",
            "-lmlir_c_runner_utils",
            "-lmlir_runner_utils",
            "-lm",
        ]
        if helper_source:
            helper_path = temp_dir_path / f"{out_path.stem}_ptr_helpers.c"
            helper_path.write_text(helper_source)
            out_flag_index = cmd.index("-o")
            cmd[out_flag_index:out_flag_index] = ["-x", "c", str(helper_path)]
        active_triple = triple or resolve_host_triple()
        if active_triple:
            cmd.extend(["-target", active_triple])
        cmd.extend(_riscv_clang_flags(include_runtime_library_dirs=True))
        if clang_flags:
            cmd.extend(clang_flags)
        _run_checked(cmd)
    return out_path


class HostKernelLibrary:
    def __init__(
        self,
        library_path: str | os.PathLike[str],
        signature: _FunctionSignature,
        *,
        owned_tempdir: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.path = Path(library_path)
        self.signature = signature
        self.function_name = signature.name
        self._owned_tempdir = owned_tempdir
        self._cdll = ctypes.CDLL(str(self.path))
        self._entry = getattr(self._cdll, self.function_name)
        self._entry.argtypes = self._build_argtypes()
        self._entry.restype = None

    def close(self) -> None:
        if self._owned_tempdir is not None:
            # ctypes does not expose a portable dlclose. Keep the directory alive
            # while the process may still hold the shared object mapped.
            _RETAINED_HOST_LIBRARY_TEMPDIRS.append(self._owned_tempdir)
            self._owned_tempdir = None

    def __del__(self) -> None:
        try:  # noqa: SIM105 - ignore errors during interpreter teardown
            self.close()
        except Exception:
            pass

    def _build_argtypes(self) -> list[type[Any]]:
        argtypes: list[type[Any]] = []
        for spec in self.signature.params:
            if isinstance(spec, _MemRefParam):
                argtypes.extend([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64])
                argtypes.extend([ctypes.c_int64] * spec.rank)
                argtypes.extend([ctypes.c_int64] * spec.rank)
            else:
                argtypes.append(spec.ctype)
        return argtypes

    def _flatten_memref_arg(self, spec: _MemRefParam, value: Any) -> list[Any]:
        array = _normalize_memref_array(spec, value)
        ptr = array.ctypes.data
        shape = _logical_memref_shape(spec, array)
        strides = _logical_row_major_strides(shape) if spec.is_fp4 else [stride // array.dtype.itemsize for stride in array.strides]
        flattened = [ctypes.c_void_p(ptr), ctypes.c_void_p(ptr), ctypes.c_int64(0)]
        flattened.extend(ctypes.c_int64(int(dim)) for dim in shape)
        flattened.extend(ctypes.c_int64(int(stride)) for stride in strides)
        return flattened

    def _flatten_scalar_arg(self, spec: _ScalarParam, value: Any) -> Any:
        return spec.ctype(_normalize_scalar_value(spec, value))

    def _prepare_args(self, *args: Any) -> list[Any]:
        if len(args) != len(self.signature.params):
            raise ValueError(f"Expected {len(self.signature.params)} arguments, but got {len(args)}")
        flattened: list[Any] = []
        for spec, value in zip(self.signature.params, args):
            if isinstance(spec, _MemRefParam):
                flattened.extend(self._flatten_memref_arg(spec, value))
            else:
                flattened.append(self._flatten_scalar_arg(spec, value))
        return flattened

    def __call__(self, *args: Any) -> None:
        self._entry(*self._prepare_args(*args))


def load_host_module(
    value: Any,
    *,
    function_name: str | None = None,
    path: str | os.PathLike[str] | None = None,
    triple: str | None = None,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> HostKernelLibrary:
    mlir_source = emit_mlir(value)
    signature = _select_signature(mlir_source, function_name)
    owned_tempdir: tempfile.TemporaryDirectory[str] | None = None
    if path is None:
        owned_tempdir = tempfile.TemporaryDirectory(prefix="tilelang-host-lib-")
        out_path = Path(owned_tempdir.name) / f"{signature.name}.so"
    else:
        out_path = Path(path)
    build_host_shared_library(
        value,
        out_path,
        triple=triple,
        pipeline=pipeline,
        clang_flags=clang_flags,
    )
    return HostKernelLibrary(out_path, signature, owned_tempdir=owned_tempdir)


def load_host_module_from_binary(
    mlir_source: str,
    path: str | os.PathLike[str],
    *,
    function_name: str | None = None,
) -> HostKernelLibrary:
    signature = _select_signature(mlir_source, function_name)
    return HostKernelLibrary(path, signature)


def run_host(
    value: Any,
    *args: Any,
    function_name: str | None = None,
    triple: str | None = None,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> None:
    load_host_module(
        value,
        function_name=function_name,
        triple=triple,
        pipeline=pipeline,
        clang_flags=clang_flags,
    )(*args)


def run_qemu(
    value: Any,
    *args: Any,
    function_name: str | None = None,
    path: str | os.PathLike[str] | None = None,
    runner: str | list[str] | tuple[str, ...] | None = None,
    triple: str = DEFAULT_RISCV_TRIPLE,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> None:
    mlir_source = emit_mlir(value)
    signature = _select_signature(mlir_source, function_name)
    _, memref_arrays = _freestanding_runner_source(signature, args)
    owned_tempdir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if path is None:
            owned_tempdir = tempfile.TemporaryDirectory(prefix="tilelang-riscv-qemu-run-")
            exe_path = Path(owned_tempdir.name) / f"{signature.name}.elf"
        else:
            exe_path = Path(path)
        build_qemu_executable(
            value,
            *args,
            function_name=function_name,
            path=exe_path,
            triple=triple,
            pipeline=pipeline,
            clang_flags=clang_flags,
        )
        cmd = resolve_riscv_runner(runner, required=True)
        proc = _run_checked_binary([*cmd, str(exe_path)])
        payload = proc.stdout
        if not payload.startswith(_QEMU_OUTPUT_MAGIC):
            raise RiscvRunnerError("RISC-V runner output did not start with the expected TileLang marker")
        cursor = len(_QEMU_OUTPUT_MAGIC)
        for array in memref_arrays:
            end = cursor + array.nbytes
            if end > len(payload):
                raise RiscvRunnerError("RISC-V runner output was truncated while decoding memref results")
            updated = np.frombuffer(payload[cursor:end], dtype=array.dtype, count=array.size).reshape(array.shape)
            np.copyto(array, updated)
            cursor = end
        if cursor != len(payload):
            raise RiscvRunnerError("RISC-V runner produced trailing bytes after the expected memref payload")
    finally:
        if owned_tempdir is not None:
            owned_tempdir.cleanup()


__all__ = [
    "HostKernelLibrary",
    "RiscvRunnerError",
    "RiscvRunnerNotFoundError",
    "build_host_shared_library",
    "build_qemu_executable",
    "load_host_module_from_binary",
    "load_host_module",
    "resolve_riscv_runner",
    "resolve_riscv_linker",
    "resolve_host_triple",
    "run_host",
    "run_qemu",
]
