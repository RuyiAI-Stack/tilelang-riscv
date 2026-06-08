# AGENTS.md

## TileLang-RISC-V Operator Workflow

When creating, completing, or validating RISC-V operators, first read:

- `.agents/skills/riscv-operator-workflow/SKILL.md`
- `.agents/riscv_operator_workflow/operators.json`
- `docs/get_started/RiscvOperatorAgentWorkflow.md`
- `docs/get_started/BuildOnSG2044.md`
- `testing/python/riscv/test_riscv_operator_workflow.py`

Rules:

- Treat `target="riscv"` as the public spelling and `linalg_riscv` as the internal backend target.
- Use x86/arm for task generation, MLIR/codegen review, host adapter checks, artifact export, and QEMU checks.
- Do not claim new SG2044 native validation unless the workflow was actually rerun on SG2044 hardware.
- Initial operator workflow coverage uses `vector_add`, `reduce_sum`, and `matmul`.
- Record failures using the workflow failure buckets.
