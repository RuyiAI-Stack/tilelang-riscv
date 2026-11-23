#!/usr/bin/env python3
"""Build the TileLang INT8 GEMM firmware for BOSC AME."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(argv, *, stdin=None):
    return subprocess.run([str(arg) for arg in argv], input=stdin, text=True, capture_output=True, check=True).stdout


def adapt_layout(source, shape):
    """Express transposed B as a view and declare the launcher's buffer ABI."""
    m, n, k = (shape[key] for key in ("M", "N", "K"))
    a_type, b_type, c_type = f"memref<{m}x{k}xi8>", f"memref<{n}x{k}xi8>", f"memref<{m}x{n}xi32>"
    expected_maps = ["(d0,d1,d2)->(d0,d2)", "(d0,d1,d2)->(d1,d2)", "(d0,d1,d2)->(d0,d1)"]
    maps = {name: re.sub(r"\s+", "", body) for name, body in re.findall(r"^(#[\w]+) = affine_map<(.+)>$", source, re.MULTILINE)}
    pattern = (
        r"(?P<indent> +)linalg\.matmul indexing_maps = \[(?P<maps>[^\]]+)\] "
        rf"ins\((?P<a>%\w+), (?P<b>%\w+) : {re.escape(a_type)}, {re.escape(b_type)}\) "
        rf"outs\((?P<c>%\w+) : {re.escape(c_type)}\)"
    )
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1 or source.count("func.func ") != 1 or source.count("linalg.matmul") != 1:
        raise ValueError("Expected exactly one canonicalized, direct-buffer TileLang transposed-B INT8 matmul")
    match = matches[0]
    if [maps.get(name.strip()) for name in match["maps"].split(",")] != expected_maps:
        raise ValueError("Unexpected indexing maps: this adapter only supports A[m,k] * B[n,k] -> C[m,n]")
    a, b, c = (match[key] for key in ("a", "b", "c"))
    signature = f"func.func @bosc_gemm({a}: {a_type}, {b}: {b_type}, {c}: {c_type})"
    entry = signature + " {"
    if source.count(entry) != 1 or source.count("module {") != 1:
        raise ValueError("Unexpected entry signature/module attributes")
    body = source.split(entry, 1)[1].rsplit("}", 2)[0]
    if re.sub(r"\s+", "", body.replace(match[0], "", 1)) != "return":
        raise ValueError("Expected only matmul and return after canonicalization")
    if "%bosc_rhs_logical" in source:
        raise ValueError("Reserved adapter SSA name is already in use")
    indent = match["indent"]
    view_type = f"memref<{k}x{n}xi8, strided<[1, {k}], offset: 0>>"
    view = (
        f"{indent}%bosc_rhs_logical = memref.reinterpret_cast {b} to "
        f"offset: [0], sizes: [{k}, {n}], strides: [1, {k}] : {b_type} to {view_type}\n"
    )
    matmul = f"{indent}linalg.matmul ins({a}, %bosc_rhs_logical : {a_type}, {view_type}) outs({c} : {c_type})"
    result = source[: match.start()] + view + matmul + source[match.end() :]
    result = result.replace("module {", "module attributes {contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}} {", 1)
    return result.replace(entry, signature + " attributes {llvm.emit_c_interface, contract.grid = array<i32: 1, 1, 1>} {", 1)


def build(work, output, buddy, llvm_bin, buddy_bin):
    import tilelang
    from tilelang import tvm
    from kernel import K, M, N, bosc_gemm

    print("Exporting and lowering INT8 GEMM...", flush=True)
    with tvm.transform.PassContext(opt_level=3, config={"tl.riscv_structured_private_writes": True}):
        artifact = tilelang.lower(bosc_gemm, target="riscv")
    (work / "kernel.mlir").write_text(artifact.kernel_source)

    def optimize(source, result, *passes):
        run([buddy_bin / "buddy-opt", "--mlir-disable-threading", work / source, *passes, "-o", work / result])

    optimize("kernel.mlir", "canonical.mlir", "--canonicalize", "--cse")
    (work / "adapted.mlir").write_text(adapt_layout((work / "canonical.mlir").read_text(), {"M": M, "N": N, "K": K}))
    optimize("adapted.mlir", "contract.mlir", "--kernel-contract-annotate", "--kernel-contract-verify")
    optimize("contract.mlir", "planned.mlir", "--bosc-ame-plan", "--bosc-ame-plan-verify")
    optimize("planned.mlir", "ame.mlir", "--bosc-ame-emit")
    optimize("ame.mlir", "llvm.mlir", "--lower-bosc-ame")
    run([buddy_bin / "buddy-translate", work / "llvm.mlir", "--buddy-to-llvmir", "-o", work / "kernel.ll"])
    run(
        [
            buddy_bin / "buddy-llc",
            work / "kernel.ll",
            "-mtriple=riscv64-unknown-elf",
            "-mattr=+m,+a,+f,+d,+c,-v,+xboscame",
            "-target-abi=lp64d",
            "-relocation-model=static",
            "-code-model=medium",
            "-O=2",
            "-filetype=asm",
            "-o",
            work / "kernel.s",
        ]
    )
    assembly = (work / "kernel.s").read_text()
    for mnemonic in ("mlae8.m", "mlbe8.m", "mqma.b.mm", "msce32.m"):
        if mnemonic not in assembly:
            raise ValueError(f"Missing AME instruction: {mnemonic}")
    tools = buddy / "examples/BOSCAMEFPGA/tools"
    words = run([sys.executable, tools / "ame_to_word.py"], stdin=assembly)
    assembly = run(
        [sys.executable, tools / "restrict_fpga_assembly.py", "--reject-vector-ordered-reductions", "--reject-vector-max-reductions"],
        stdin=words,
    )
    (work / "kernel.nr.s").write_text(assembly)

    print("Compiling and linking the firmware...", flush=True)
    nr = buddy / "runtime/boscame/nr"
    (work / "kernel_config.h").write_text(f"#pragma once\n#define BOSC_M {M}\n#define BOSC_N {N}\n#define BOSC_K {K}\n")
    cflags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64gc_zicbom",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-pie",
        "-fno-stack-protector",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-ffp-contract=off",
        "-ffunction-sections",
        "-fdata-sections",
        "-O2",
        f"-I{nr}",
        f"-I{nr.parent / 'uart'}",
        f"-I{work}",
    ]
    inputs = [("kernel", work / "kernel.nr.s"), ("launch", HERE / "launch.c")]
    for filename in ("crt.S", "nr_runtime.c", "nr_memory.c", "nr_math.c", "ame_sync.c", "nr_copy.S"):
        inputs.append((Path(filename).stem, nr / filename))
    objects = []
    for name, source in inputs:
        obj = work / f"{name}.o"
        flags = cflags + ["-march=rv64gcv"] if name == "nr_copy" else cflags
        run([llvm_bin / "clang", *flags, "-c", source, "-o", obj])
        objects.append(obj)
    elf = work / "bosc_gemm.elf"
    run([llvm_bin / "ld.lld", "-m", "elf64lriscv", "-T", nr / "nr.ld", "--gc-sections", "--no-undefined", "-o", elf, *objects])
    run([llvm_bin / "llvm-objcopy", "-O", "binary", elf, work / "bosc_gemm.bin"])
    run(
        [
            sys.executable,
            tools / "check_nr_elf.py",
            elf,
            "--objdump",
            llvm_bin / "llvm-objdump",
            "--require-direct-store",
            "--output",
            work / "isa-check.json",
        ]
    )
    for name in ("bosc_gemm.elf", "bosc_gemm.bin"):
        shutil.copy2(work / name, output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New/empty output directory; default: build/<timestamp>")
    args = parser.parse_args()
    output = (args.output or HERE / "build" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be a new or empty directory: {output}")
    buddy = Path(os.environ.get("BOSC_BUDDY_SOURCE", HERE.parents[2].parent / "buddy-mlir")).expanduser().resolve()
    if not (buddy / "runtime/boscame/nr/nr_uart_protocol.h").is_file():
        parser.error("Buddy NR runtime must support UART v1; update BOSC_BUDDY_SOURCE")
    for name in ("TILELANG_RISCV_LLVM_ROOT", "TILELANG_RISCV_BUDDY_ROOT"):
        if not os.environ.get(name):
            parser.error(f"Set {name} or source examples/riscv/bosc/env.sh first")
    llvm_bin = Path(os.environ["TILELANG_RISCV_LLVM_ROOT"]).expanduser().resolve() / "bin"
    buddy_bin = Path(os.environ["TILELANG_RISCV_BUDDY_ROOT"]).expanduser().resolve() / "bin"
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="bosc-") as directory:
            build(Path(directory), output, buddy, llvm_bin, buddy_bin)
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Firmware: {output / 'bosc_gemm.bin'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
