from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import tilelang
from tilelang.jit.adapter.riscv import emit_asm, emit_llvm_ir, emit_mlir, emit_object, run_qemu


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=Path("riscv_artifacts"))
    parser.add_argument("--run-host", action="store_true", help="Compile and execute through the native host adapter.")
    parser.add_argument("--run-qemu", action="store_true", help="Build and execute a freestanding RISC-V ELF.")
    parser.add_argument("--emit-mlir", action="store_true", help="Write the structured MLIR module.")
    parser.add_argument("--emit-llvm", action="store_true", help="Write translated LLVM IR.")
    parser.add_argument("--emit-asm", action="store_true", help="Write RISC-V assembly.")
    parser.add_argument("--emit-object", action="store_true", help="Write a RISC-V object file.")
    return parser


def lower_riscv(func):
    return tilelang.lower(func, target="riscv").rt_mod


def emit_requested_artifacts(name: str, func, args: argparse.Namespace) -> None:
    if not (args.emit_mlir or args.emit_llvm or args.emit_asm or args.emit_object):
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rt_mod = lower_riscv(func)
    if args.emit_mlir:
        emit_mlir(rt_mod, args.output_dir / f"{name}.mlir")
    if args.emit_llvm:
        emit_llvm_ir(rt_mod, args.output_dir / f"{name}.ll")
    if args.emit_asm:
        emit_asm(rt_mod, args.output_dir / f"{name}.s")
    if args.emit_object:
        emit_object(rt_mod, args.output_dir / f"{name}.o")


def run_host_check(name: str, func, out_idx: list[int], inputs: tuple[torch.Tensor, ...], expected: torch.Tensor) -> None:
    kernel = tilelang.compile(func, out_idx=out_idx, target="riscv")
    try:
        actual = kernel(*inputs)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    finally:
        kernel.close()
    print(f"{name} host check passed")


def run_qemu_check(
    name: str,
    func,
    all_args: tuple[np.ndarray, ...],
    expected: np.ndarray,
    output_arg_index: int,
    output_dir: Path,
) -> None:
    """Run small static-shaped examples with simple memref-style signatures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_qemu(lower_riscv(func), *all_args, path=output_dir / f"{name}.qemu.elf")
    np.testing.assert_allclose(all_args[output_arg_index], expected, atol=1e-5, rtol=1e-5)
    print(f"{name} qemu check passed")


def run_example(
    *,
    name: str,
    description: str,
    func,
    out_idx: list[int],
    make_torch_case: Callable[[], tuple[tuple[torch.Tensor, ...], torch.Tensor]],
    make_numpy_case: Callable[[], tuple[tuple[np.ndarray, ...], np.ndarray, int]],
) -> None:
    parser = build_arg_parser(description)
    args = parser.parse_args()
    emit_requested_artifacts(name, func, args)

    if args.run_host:
        inputs, expected = make_torch_case()
        run_host_check(name, func, out_idx, inputs, expected)

    if args.run_qemu:
        all_args, expected, output_arg_index = make_numpy_case()
        run_qemu_check(name, func, all_args, expected, output_arg_index, args.output_dir)

    if not any(
        (
            args.emit_mlir,
            args.emit_llvm,
            args.emit_asm,
            args.emit_object,
            args.run_host,
            args.run_qemu,
        )
    ):
        parser.print_help()
