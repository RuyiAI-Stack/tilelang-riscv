# Implement and validate small matmul for TileLang-RISCV

## Goal

Complete or refine the TileLang-RISCV implementation for `matmul` and validate it through task generation, x86/arm host execution, RISC-V artifact export, and QEMU execution when the toolchain is available.

## Operator Semantics

For M=2, N=4, K=3, compute C[i, j] = sum(A[i, k] * B[k, j] for k in [0, 3)).

## Inputs

- A: float32 tensor with shape (2, 3)
- B: float32 tensor with shape (3, 4)

## Outputs

- C: float32 tensor with shape (2, 4)

## Baseline Reference

- PyTorch: lhs @ rhs
- NumPy: lhs @ rhs

## Baseline Implementation

- PyTorch: expected = lhs @ rhs
- NumPy: expected = lhs @ rhs

## Context Files

- examples/riscv/example_matmul.py
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

- Use T.grid(M, N, K) with two spatial axes and one reduce axis.
- Use T.init() to zero C[i, j] before the K accumulation.
- Start with plain TIR loops rather than GPU tile intrinsics; move to T.gemm only after the linalg_riscv path supports the needed pattern.

## Compilation Commands

- python examples/riscv/example_matmul.py --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/matmul
- python examples/riscv/example_matmul.py --run-host --output-dir /tmp/matmul
- python examples/riscv/example_matmul.py --run-qemu --output-dir /tmp/matmul

## Test Commands

- python -m pytest testing/python/riscv/test_riscv_operator_workflow_examples.py -q -k matmul

## Success Criteria

- Lowered MLIR keeps the matmul semantics visible in structured lowering or produces an equivalent loop form.
- Host adapter output matches the PyTorch matmul reference.
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
