# TileLang-RISC-V Operator Workflow

This directory contains the Codex-compatible operator workflow for TileLang-RISCV.

## Entry Points

- `operators.json`: source of truth for initial operator task context
- `validation_report.md`: validation results, SG2044 baseline status, and PR summary draft
- `.agents/tasks/riscv/*.md`: generated Codex task prompts
- `.agents/skills/riscv-operator-workflow/SKILL.md`: Codex workflow instructions
- `docs/get_started/RiscvOperatorAgentWorkflow.md`: user-facing workflow guide

## Generate Tasks

```bash
python maint/scripts/generate_riscv_operator_task.py --list
python maint/scripts/generate_riscv_operator_task.py --all
python maint/scripts/generate_riscv_operator_task.py --operator vector_add
```

## Validate

```bash
bash maint/scripts/run_riscv_operator_workflow_validation.sh
```

The initial workflow validates `vector_add`, `reduce_sum`, and `matmul` through host checks, RISC-V artifact export, and QEMU functional smoke checks.

## SG2044 Boundary

No new SG2044 native run is claimed unless the workflow is actually rerun on SG2044 hardware.

Existing SG2044 baseline data from `docs/get_started/BuildOnSG2044.md` is kept as historical baseline data. QEMU functional smoke checks do not replace SG2044 native RVV validation.
