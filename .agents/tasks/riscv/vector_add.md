# Implement and validate vector_add for TileLang-RISCV

## Goal

Complete or refine the TileLang-RISCV implementation for `vector_add` and validate it through task generation, x86/arm host execution, RISC-V artifact export, and QEMU execution when the toolchain is available.

## Operator Semantics

For each i in [0, 8), compute C[i] = A[i] + B[i].

## Inputs

- A: float32 tensor with shape (8,)
- B: float32 tensor with shape (8,)

## Outputs

- C: float32 tensor with shape (8,)

## Baseline Reference

- PyTorch: lhs + rhs
- NumPy: lhs + rhs

## Baseline Implementation

- PyTorch: expected = lhs + rhs
- NumPy: expected = lhs + rhs

## Context Files

- examples/riscv/example_vector_add.py
- testing/python/riscv/test_riscv_examples.py
- testing/python/riscv/test_riscv_operator_workflow_examples.py
- testing/python/riscv/test_riscv_qemu_smoke.py
- .agents/riscv_operator_workflow/README.md
- docs/get_started/BuildOnSG2044.md
- docs/get_started/targets.md
- testing/python/riscv/test_riscv_target_parse.py
- testing/python/riscv/test_riscv_mlir_codegen.py
- testing/python/riscv/test_riscv_toolchain.py
- testing/python/riscv/test_riscv_jit_runtime.py

## Scheduling Strategy

- Use a single serial loop over N.
- Use a T.block with one spatial axis.
- Avoid GPU-specific T.Kernel threading, shared memory, warp, or async-copy constructs for this first RISC-V path.

## Compilation Commands

- python examples/riscv/example_vector_add.py --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/vector_add
- python examples/riscv/example_vector_add.py --run-host --output-dir /tmp/vector_add
- python examples/riscv/example_vector_add.py --run-qemu --output-dir /tmp/vector_add

## Test Commands

- python -m pytest testing/python/riscv/test_riscv_operator_workflow_examples.py -q -k vector_add
- python -m pytest testing/python/riscv/test_riscv_qemu_smoke.py -q

## Success Criteria

- Host adapter output matches the PyTorch reference.
- Artifact export produces one non-empty .s and one non-empty .o file.
- QEMU smoke emits vector_add.qemu.elf and output matches NumPy reference when a runner is available.

## Failure Triage Buckets

- Generated TileLang/TIR source error
- Unsupported scheduling or TileLang intrinsic for linalg_riscv
- MLIR lowering or structured codegen limitation
- LLVM/RISC-V toolchain setup issue
- Runtime wrapper, host adapter, QEMU, or optional SG2044 environment issue
- Test configuration or tolerance mismatch

## Existing SG2044 Baseline

- Existing documented SG2044 native baseline from docs/get_started/BuildOnSG2044.md:
- testing/python/riscv/test_riscv_target_parse.py: 2 passed
- testing/python/riscv/test_riscv_mlir_codegen.py: 33 passed
- testing/python/riscv/test_riscv_toolchain.py: 3 passed
- testing/python/riscv/test_riscv_jit_runtime.py: 16 passed
- This baseline was not rerun during the no-hardware workflow validation pass.

## Reporting Template

- Operator:
- Implementation files changed:
- Host compile/run result:
- RISC-V artifact result:
- QEMU result:
- Optional SG2044 replay result:
- Failure bucket, if any:
- Follow-up needed:

## Target Flow

TileLang -> MLIR Linalg -> MLIR vector/RVV lowering -> LLVM/RISC-V artifact -> host check -> QEMU smoke check -> optional SG2044 native replay

The Codex operator workflow can be pre-validated without SG2044 hardware by using x86/arm host execution, RISC-V artifact export, and QEMU. Native SG2044 replay remains the final hardware validation step when SG2044 hardware is available.
