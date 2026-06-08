"""Helpers for MLIR -> LLVM -> RISC-V artifact generation."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tilelang.tladapter.toolchain import resolve_tool
from tilelang.tladapter.utils import Pipeline


DEFAULT_RISCV_TRIPLE = os.environ.get("TILELANG_RISCV_TRIPLE", "riscv64-unknown-linux-gnu")
DEFAULT_RISCV_LLC_MATTR = os.environ.get("TILELANG_RISCV_LLC_MATTR", "+m,+a,+f,+d,+c")


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


def build_debug_pipeline() -> Pipeline:
    pipeline = Pipeline()
    for pass_name in (
        "canonicalize",
        "cse",
        "func.func(convert-linalg-to-loops)",
        "canonicalize",
        "cse",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "finalize-memref-to-llvm",
        "convert-math-to-llvm",
        "convert-arith-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
    ):
        pipeline.add(pass_name)
    return pipeline


def lower_to_llvm_dialect_mlir(value: Any, *, pipeline: Pipeline | None = None) -> str:
    mlir_source = _extract_mlir_source(value)
    active_pipeline = pipeline if pipeline is not None else build_debug_pipeline()
    return active_pipeline.run(mlir_source)


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
        out_path = temp_dir_path / f"kernel.{ 's' if filetype == 'asm' else 'o' }"
        ll_path.write_text(llvm_ir)
        cmd = [
            str(resolve_tool("llc")),
            f"-mtriple={triple}",
            f"-filetype={filetype}",
        ]
        if DEFAULT_RISCV_LLC_MATTR:
            cmd.append(f"-mattr={DEFAULT_RISCV_LLC_MATTR}")
        if os.environ.get("TILELANG_RISCV_LLC_FLAGS"):
            cmd.extend(shlex.split(os.environ["TILELANG_RISCV_LLC_FLAGS"]))
        cmd.extend([str(ll_path), "-o", str(out_path)])
        _run_checked(cmd)
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
