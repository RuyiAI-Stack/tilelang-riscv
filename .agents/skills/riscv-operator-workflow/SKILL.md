# RISC-V Operator Workflow

Use this skill when creating, completing, or validating TileLang operators for the `riscv` / `linalg_riscv` backend.

## Workflow

1. Read the generated operator task from `.agents/tasks/riscv/`.
2. If the task does not exist, generate it with:

```bash
python maint/scripts/generate_riscv_operator_task.py --operator <name>
```

3. Review the context files named in the task before editing code.
4. Implement the operator in `examples/riscv/` or the task's suggested files.
5. Validate on x86/arm first with host execution and artifact export when the Buddy/LLVM toolchain is available.
6. Record failures using one of the task's failure buckets.
7. Update `.agents/riscv_operator_workflow/validation_report.md` with the compile, host, artifact, QEMU, existing SG2044 baseline, and optional SG2044 replay result.
8. Run or hand off the same task on SG2044 only when native hardware is available.

## Commands

List available operator tasks:

```bash
python maint/scripts/generate_riscv_operator_task.py --list
```

Generate all initial tasks:

```bash
python maint/scripts/generate_riscv_operator_task.py --all
```

Run the initial validation examples:

```bash
python -m pytest testing/python/riscv/test_riscv_operator_workflow_examples.py -q
python -m pytest testing/python/riscv/test_riscv_qemu_smoke.py -q
```

QEMU smoke checks are no-hardware functional validation. They do not replace SG2044 native RVV validation.

Run an example manually:

```bash
python examples/riscv/example_vector_add.py --run-host --output-dir /tmp/vector_add
python examples/riscv/example_vector_add.py --emit-mlir --emit-llvm --emit-asm --emit-object --output-dir /tmp/vector_add
```

## Reporting

Each operator report should include:

- operator name and category
- files changed
- host execution result
- artifact export result
- QEMU result
- optional SG2044 replay result
- failure bucket if validation failed
- follow-up needed
