# Implement and validate row-wise reduce_sum for TileLang-RISCV

## Goal

Complete or refine the TileLang-RISCV implementation for `reduce_sum` and validate it through task generation, x86/arm host execution, RISC-V artifact export, and QEMU execution when the toolchain is available.

## Operator Semantics

For each row i in [0, 4), compute B[i] = sum(A[i, k] for k in [0, 8)).

## Inputs

- A: float32 tensor with shape (4, 8)

## Outputs

- B: float32 tensor with shape (4,)

## Baseline Reference

- PyTorch: data.sum(dim=1)
- NumPy: data.sum(axis=1)

## Baseline Implementation

- PyTorch: expected = data.sum(dim=1)
- NumPy: expected = data.sum(axis=1)

## Context Files

- examples/riscv/example_reduce_sum.py
- testing/python/riscv/test_riscv_examples.py
- testing/python/riscv/test_riscv_operator_workflow_examples.py
- .agents/riscv_operator_workflow/README.md
- docs/get_started/BuildOnSG2044.md
- docs/get_started/targets.md
- testing/python/riscv/test_riscv_target_parse.py
- testing/python/riscv/test_riscv_mlir_codegen.py
- testing/python/riscv/test_riscv_toolchain.py
- testing/python/riscv/test_riscv_jit_runtime.py
- testing/python/riscv/test_riscv_qemu_smoke.py

## Scheduling Strategy

- Use T.grid(ROWS, COLS) with one spatial row axis and one reduce axis.
- Use T.init() to zero the output row before accumulation.
- Keep the first validation shape small and static so qemu artifact generation has fixed memref sizes.

## Compilation Commands

- python examples/riscv/example_reduce_sum.py --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/reduce_sum
- python examples/riscv/example_reduce_sum.py --run-host --output-dir /tmp/reduce_sum
- python examples/riscv/example_reduce_sum.py --run-qemu --output-dir /tmp/reduce_sum

## Test Commands

- python -m pytest testing/python/riscv/test_riscv_operator_workflow_examples.py -q -k reduce_sum

## Success Criteria

- Lowered MLIR contains a reduction-friendly structured form such as linalg.reduce or equivalent generic lowering.
- Host adapter output matches the PyTorch row-sum reference.
- Artifact export produces one non-empty .s and one non-empty .o file.

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
