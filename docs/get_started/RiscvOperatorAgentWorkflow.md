# RISC-V Operator Agent Workflow

This workflow turns a TileLang-RISCV operator request into a Codex-ready task, then validates the result through the same commands used by the RISC-V backend tests.

The Codex operator workflow can be pre-validated without SG2044 hardware by using x86 or arm host execution, RISC-V artifact export, and QEMU.
Native SG2044 replay remains the final hardware validation step when SG2044 hardware is available.

## Files

- `.agents/riscv_operator_workflow/operators.json`: operator catalog and shared validation context
- `.agents/riscv_operator_workflow/README.md`: short workflow entry point
- `maint/scripts/generate_riscv_operator_task.py`: task generator
- `.agents/tasks/riscv/*.md`: generated Codex task files
- `.agents/skills/riscv-operator-workflow/SKILL.md`: repository-local skill for the workflow
- `examples/riscv/*.py`: executable validation examples
- `testing/python/riscv/test_riscv_operator_workflow_examples.py`: focused tests for the initial workflow examples
- `.agents/riscv_operator_workflow/validation_report.md`: compilation, correctness, and failure-triage report for the initial operators

## Generate Tasks

List available operator tasks:

```bash
python maint/scripts/generate_riscv_operator_task.py --list
```

Generate all initial tasks:

```bash
python maint/scripts/generate_riscv_operator_task.py --all
```

Generate one task:

```bash
python maint/scripts/generate_riscv_operator_task.py --operator vector_add
```

## Initial Operators

| Operator | Pattern | Example |
| --- | --- | --- |
| `vector_add` | elementwise | `examples/riscv/example_vector_add.py` |
| `reduce_sum` | reduction | `examples/riscv/example_reduce_sum.py` |
| `matmul` | matrix/tensor compute | `examples/riscv/example_matmul.py` |

These three operators are intentionally small and static-shaped. They cover the initial workflow categories while keeping artifact and QEMU validation simple.

The current QEMU runner path is intended for small static-shaped examples with simple memref-style signatures. More complex dynamic-shape or runtime-heavy operators should be validated separately on SG2044 native hardware.

## Validation Levels

Use three separate validation levels when reporting results:

- x86/arm host check
- RISC-V artifact export plus QEMU smoke check
- SG2044 native hardware replay

QEMU smoke checks are useful no-hardware functional validation, but they do not replace SG2044 native RVV validation.

## x86/arm Validation

On an x86 or arm developer host, use the workflow to validate task generation and the non-native parts of the RISC-V flow:

```bash
bash maint/scripts/run_riscv_operator_workflow_validation.sh
```

The validation script reads `.agents/riscv_operator_workflow/operators.json`, so adding another catalog entry is enough to include that operator in the manual validation loop.
Useful environment variables:

- `TILELANG_RISCV_RUN_QEMU=0|1|auto`: control qemu execution; `auto` is the default.
- `TILELANG_RISCV_OPERATORS="vector_add matmul"`: validate only selected catalog operators.
- `TILELANG_RISCV_PYTEST_ARGS="testing/python/riscv/test_riscv_examples.py -q"`: override the pytest scope.
- `TILELANG_RISCV_RUN_PYTEST=0`: skip pytest and run only the manual operator loop.
- `TILELANG_RISCV_VALIDATION_OUTPUT_ROOT=/tmp`: choose where per-operator artifacts are written.
- `TILELANG_RISCV_RUN_HOST=0`: skip manual host execution.
- `TILELANG_RISCV_EMIT_ARTIFACTS=0`: skip manual artifact export.

Or run the checks individually:

```bash
python maint/scripts/generate_riscv_operator_task.py --all
python -m pytest testing/python/riscv/test_riscv_operator_workflow.py -q
python -m pytest testing/python/riscv/test_riscv_target_parse.py -q
python -m pytest testing/python/riscv/test_riscv_mlir_codegen.py -q
python -m pytest testing/python/riscv/test_riscv_examples.py -q
python -m pytest testing/python/riscv/test_riscv_operator_workflow_examples.py -q
```

When the Buddy/LLVM tools are available, also validate artifact export:

```bash
python examples/riscv/example_vector_add.py --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/vector_add
python examples/riscv/example_reduce_sum.py --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/reduce_sum
python examples/riscv/example_matmul.py --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/matmul
```

If the host adapter toolchain is configured, run host correctness:

```bash
python examples/riscv/example_vector_add.py --run-host --output-dir /tmp/vector_add
python examples/riscv/example_reduce_sum.py --run-host --output-dir /tmp/reduce_sum
python examples/riscv/example_matmul.py --run-host --output-dir /tmp/matmul
```

## Optional SG2044 Replay

SG2044 hardware is not required for no-hardware functional validation. When an SG2044 machine is available, first follow the native setup in `docs/get_started/BuildOnSG2044.md`, then run:

```bash
TILELANG_RISCV_RUN_QEMU=0 bash maint/scripts/run_riscv_operator_workflow_validation.sh
```

For an individual operator:

```bash
python examples/riscv/example_vector_add.py --run-host --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/vector_add
```

## Results Report

Record each operator result in `.agents/riscv_operator_workflow/validation_report.md`.
The report tracks host correctness, RISC-V artifact export, QEMU execution, optional SG2044 replay status, failure buckets, and follow-up.
It also includes the existing SG2044 baseline results already documented in `docs/get_started/BuildOnSG2044.md`, clearly separated from the new no-hardware operator validation results.

## Failure Buckets

Use these buckets in task reports:

- generated TileLang/TIR source error
- unsupported schedule or TileLang intrinsic for `linalg_riscv`
- MLIR lowering or structured codegen limitation
- LLVM/RISC-V toolchain setup issue
- runtime wrapper, host adapter, QEMU, or optional SG2044 environment issue
- test configuration or tolerance mismatch
