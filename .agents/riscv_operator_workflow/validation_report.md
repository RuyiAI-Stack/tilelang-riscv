# TileLang-RISCV Operator Workflow Validation Report

This report records the initial Codex operator workflow validation for issue
`RuyiAI-Stack/tilelang-riscv#1`.

## Scope

- Host used for this report: x86_64 WSL Ubuntu with LLVM/MLIR 18 and QEMU user mode.
- Native SG2044 status: no new SG2044 run was performed in this validation pass because no SG2044 hardware is available.
- Existing SG2044 data: the documented SG2044 baseline from `docs/get_started/BuildOnSG2044.md` is included below for context.
- RISC-V execution coverage: QEMU user-mode execution of freestanding RISC-V ELFs for all initial operators.
- Workflow target: generate Codex task files, complete three representative operators, compile through the RISC-V lowering path, and run correctness checks.

The current QEMU validation path checks functional execution of small freestanding RISC-V ELFs, primarily targeting the `rv64gc` correctness path. It does not claim RVV-specific instruction coverage or performance validation.

## Environment

- Source checkout: `/home/devcontainers/codex-work/tilelang-riscv`
- Python environment: `/home/devcontainers/codex-work/tilelang-riscv/.venv-riscv`
- LLVM root: `/usr/lib/llvm-18`
- Required exports:

```bash
source .venv-riscv/bin/activate
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_RISCV_LLVM_ROOT=/usr/lib/llvm-18
export Z3_ROOT=/usr
```

## Workflow Artifacts

- Operator catalog: `.agents/riscv_operator_workflow/operators.json`
- Task generator: `maint/scripts/generate_riscv_operator_task.py`
- Generated tasks:
  - `.agents/tasks/riscv/vector_add.md`
  - `.agents/tasks/riscv/reduce_sum.md`
  - `.agents/tasks/riscv/matmul.md`
- Repository-local skill: `.agents/skills/riscv-operator-workflow/SKILL.md`
- Workflow guide: `docs/get_started/RiscvOperatorAgentWorkflow.md`
- One-command validation runner: `maint/scripts/run_riscv_operator_workflow_validation.sh`
- Validation runner controls: `TILELANG_RISCV_RUN_QEMU`, `TILELANG_RISCV_OPERATORS`, `TILELANG_RISCV_PYTEST_ARGS`, `TILELANG_RISCV_RUN_PYTEST`, `TILELANG_RISCV_VALIDATION_OUTPUT_ROOT`, `TILELANG_RISCV_RUN_HOST`, and `TILELANG_RISCV_EMIT_ARTIFACTS`

## Operator Results

| Operator | Category | Implementation | Host correctness | Artifact export | QEMU correctness | Failure bucket |
| --- | --- | --- | --- | --- | --- | --- |
| `vector_add` | elementwise | `examples/riscv/example_vector_add.py` | passed | `.mlir`, `.ll`, `.s`, `.o` passed | passed | none |
| `reduce_sum` | reduction | `examples/riscv/example_reduce_sum.py` | passed | `.mlir`, `.ll`, `.s`, `.o` passed | passed | none |
| `matmul` | matrix/tensor | `examples/riscv/example_matmul.py` | passed | `.mlir`, `.ll`, `.s`, `.o` passed | passed | none |

## Existing SG2044 Baseline Data

`docs/get_started/BuildOnSG2044.md` already records the following validated native SG2044 bring-up results:

| Test | Documented SG2044 result |
| --- | --- |
| `testing/python/riscv/test_riscv_target_parse.py` | `2 passed` |
| `testing/python/riscv/test_riscv_mlir_codegen.py` | `33 passed` |
| `testing/python/riscv/test_riscv_toolchain.py` | `3 passed` |
| `testing/python/riscv/test_riscv_jit_runtime.py` | `16 passed` |

These are existing repository baseline results. They were not rerun for this operator workflow validation because no SG2044 hardware is available in the current environment.

## SG2044 Native Validation Status

No new SG2044 native run was performed because SG2044 hardware was not available in this environment.

The existing SG2044 baseline from `docs/get_started/BuildOnSG2044.md` is kept as historical baseline data. It was not rerun as part of this submission.

The no-hardware validation in this submission covers:

- host checks
- RISC-V artifact export
- QEMU functional smoke checks

It does not replace SG2044 native RVV validation.

## Issue Requirement Audit

| Requirement | Evidence |
| --- | --- |
| Codex-compatible workflow | `.agents/skills/riscv-operator-workflow/SKILL.md`, `.agents/riscv_operator_workflow/operators.json`, and `docs/get_started/RiscvOperatorAgentWorkflow.md` |
| Task contexts include semantics, references, baseline implementation, files, schedule, compile/test commands, and success criteria | Generated task files in `.agents/tasks/riscv/`; enforced by `testing/python/riscv/test_riscv_operator_workflow.py` |
| Simple task generation flow for a given operator | `maint/scripts/generate_riscv_operator_task.py --operator <name>` and `--all` |
| 2 to 3 validation operators covering different compute patterns | `vector_add` for elementwise, `reduce_sum` for reduction, `matmul` for matrix/tensor compute |
| Compilation flow and basic correctness tests for each operator | `testing/python/riscv/test_riscv_examples.py`, `testing/python/riscv/test_riscv_operator_workflow_examples.py`, `testing/python/riscv/test_riscv_qemu_smoke.py`, and per-operator manual commands below |
| Results, failure logs, and failure reasons recorded | This report's operator table, existing SG2044 baseline, command results, and failure log |
| Reusable foundation for scaling to more operators | Catalog-driven generator, catalog-driven validation runner, and workflow test coverage |

## Commands And Results

Task generation:

```bash
bash maint/scripts/run_riscv_operator_workflow_validation.sh
# generates tasks, runs testing/python/riscv, and runs the three manual operator checks
# 85 passed, 26 skipped
# vector_add host check passed
# vector_add qemu check passed
# reduce_sum host check passed
# reduce_sum qemu check passed
# matmul host check passed
# matmul qemu check passed
```

Individual task generation:

```bash
python maint/scripts/generate_riscv_operator_task.py --list
# matmul
# reduce_sum
# vector_add

python maint/scripts/generate_riscv_operator_task.py --all
# .agents/tasks/riscv/matmul.md
# .agents/tasks/riscv/reduce_sum.md
# .agents/tasks/riscv/vector_add.md
```

RISC-V backend tests:

```bash
python -m pytest testing/python/riscv/test_riscv_operator_workflow.py -q
# 7 passed on Linux/WSL
# 5 passed, 2 skipped on Windows where bash validation is not available as a normal shell

python -m pytest testing/python/riscv/test_riscv_target_parse.py -q
# 2 passed

python -m pytest testing/python/riscv/test_riscv_toolchain.py -q
# 3 passed

python -m pytest testing/python/riscv/test_riscv_mlir_codegen.py -q
# 33 passed

python -m pytest testing/python/riscv/test_riscv_jit_runtime.py -q
# 16 passed

python -m pytest testing/python/riscv/test_riscv_artifact_export.py -q
# 4 passed

python -m pytest testing/python/riscv/test_riscv_examples.py -q
# 6 passed, 26 skipped

python -m pytest testing/python/riscv/test_riscv_operator_workflow_examples.py -q
# 6 passed

python -m pytest testing/python/riscv/test_riscv_qemu_smoke.py -q
# 6 passed

python -m pytest testing/python/riscv/test_riscv_tladapter_pipeline.py -q
# 1 passed

python -m pytest testing/python/riscv -q
# 85 passed, 26 skipped
```

Coverage note: `testing/python/riscv/test_riscv_examples.py` keeps the original 16-entry RISC-V example list. The current checkout contains the three initial workflow example scripts, so the missing legacy example filenames are reported as skips instead of being removed from the coverage entry point. The focused Issue #1 workflow checks live in `testing/python/riscv/test_riscv_operator_workflow_examples.py`.

Per-operator manual checks:

```bash
python examples/riscv/example_vector_add.py --run-host --run-qemu --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/tilelang-riscv-vector_add
# vector_add host check passed
# vector_add qemu check passed

python examples/riscv/example_reduce_sum.py --run-host --run-qemu --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/tilelang-riscv-reduce_sum
# reduce_sum host check passed
# reduce_sum qemu check passed

python examples/riscv/example_matmul.py --run-host --run-qemu --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/tilelang-riscv-matmul
# matmul host check passed
# matmul qemu check passed
```

## Failure Log

Current operator validation has no remaining failing operator.

Two environment and lowering issues were found while preparing the workflow:

- MLIR 18 API compatibility: Ubuntu MLIR 18 requires the newer `arith::ConstantIntOp`,
  `arith::ConstantFloatOp`, and `memref::SubViewOp` result type usage. This was fixed in
  `src/target/codegen_linalg_riscv.cc`. Failure bucket: MLIR lowering or structured codegen limitation.
- QEMU link failure from soft-float helpers such as `__addsf3`: the emitted RISC-V object did not
  consistently include hard-float features for the freestanding runner. This was fixed by defaulting
  `llc` to `-mattr=+m,+a,+f,+d,+c` and the QEMU linker path to `-march=rv64gc`, while preserving
  environment overrides. Failure bucket: LLVM/RISC-V toolchain setup issue.

## Effectiveness Summary

The workflow is effective for the initial operator loop:

- The catalog and generator create repeatable Codex task descriptions with operator semantics, shapes, dtypes, baseline references, scheduling guidance, compile commands, tests, success criteria, and failure buckets.
- The three initial operators cover elementwise, reduction, and matrix/tensor patterns.
- The examples can be used both as implementation references and as executable validation targets.
- Host execution and artifact export run on x86_64 with the RISC-V toolchain, which lets the agent loop be pre-validated without SG2044 hardware.
- QEMU checks provide a stronger non-native signal by executing freestanding RISC-V ELFs for all initial operators, but they do not replace SG2044 native RVV validation.
- Existing SG2044 backend baseline data is included from `docs/get_started/BuildOnSG2044.md` and kept separate from the new no-hardware operator validation results.

Optional follow-up:

- Replay the same commands on SG2044 for native RVV execution when hardware is available.
- RVV-specific checks can be added later by enabling vector-capable target attributes and checking emitted assembly for vector instructions such as `vsetvli`, `vle`, `vse`, `vfadd`, or `vfmacc`.
- Add more operators only after this initial workflow is reviewed, so new tasks can reuse the same catalog/report structure.

## PR Summary Draft

Summary:

- Add a Codex-compatible TileLang-RISC-V operator workflow catalog, generated tasks, repository-local skill, and workflow guide.
- Add three initial validation operators covering elementwise, reduction, and matrix/tensor patterns.
- Add executable examples, workflow tests, QEMU smoke coverage, and a catalog-driven validation runner.
- Fix LLVM/MLIR 18 compatibility and default RISC-V hard-float codegen flags needed by the QEMU path.
- Record no-hardware validation results and existing SG2044 baseline data separately.

Tests:

```bash
bash maint/scripts/run_riscv_operator_workflow_validation.sh
# 85 passed, 26 skipped
# vector_add host check passed
# vector_add qemu check passed
# reduce_sum host check passed
# reduce_sum qemu check passed
# matmul host check passed
# matmul qemu check passed
```
