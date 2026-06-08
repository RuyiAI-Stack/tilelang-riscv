#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON:-python}"
run_qemu="${TILELANG_RISCV_RUN_QEMU:-auto}"
run_host="${TILELANG_RISCV_RUN_HOST:-1}"
run_pytest="${TILELANG_RISCV_RUN_PYTEST:-1}"
emit_artifacts="${TILELANG_RISCV_EMIT_ARTIFACTS:-1}"
validation_output_root="${TILELANG_RISCV_VALIDATION_OUTPUT_ROOT:-/tmp}"

require_binary_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1." >&2
    exit 2
  fi
}

if [[ "${run_qemu}" == "auto" ]]; then
  if command -v qemu-riscv64 >/dev/null 2>&1 || [[ -n "${TILELANG_RISCV_RUNNER:-}" ]]; then
    run_qemu=1
  else
    run_qemu=0
  fi
fi

if [[ "${run_qemu}" != "0" && "${run_qemu}" != "1" ]]; then
  echo "TILELANG_RISCV_RUN_QEMU must be 0, 1, or auto." >&2
  exit 2
fi
require_binary_flag TILELANG_RISCV_RUN_HOST "${run_host}"
require_binary_flag TILELANG_RISCV_RUN_PYTEST "${run_pytest}"
require_binary_flag TILELANG_RISCV_EMIT_ARTIFACTS "${emit_artifacts}"

"${python_bin}" maint/scripts/generate_riscv_operator_task.py --all

if [[ "${run_pytest}" == "1" ]]; then
  read -r -a pytest_args <<< "${TILELANG_RISCV_PYTEST_ARGS:-testing/python/riscv -q}"
  "${python_bin}" -m pytest "${pytest_args[@]}"
fi

if [[ "${run_host}" != "1" && "${emit_artifacts}" != "1" && "${run_qemu}" != "1" ]]; then
  exit 0
fi

operator_lines="$(
  "${python_bin}" - <<'PY'
import json
import os
from pathlib import Path

catalog = json.loads(Path(".agents/riscv_operator_workflow/operators.json").read_text(encoding="utf-8"))
selected = set(os.environ.get("TILELANG_RISCV_OPERATORS", "").split())
known = {operator["name"] for operator in catalog["operators"]}
unknown = sorted(selected - known)
if unknown:
    raise SystemExit(f"Unknown TILELANG_RISCV_OPERATORS entries: {', '.join(unknown)}")
for operator in catalog["operators"]:
    if selected and operator["name"] not in selected:
        continue
    print(f"{operator['name']}\t{operator['current_example']}")
PY
)"

if [[ -z "${operator_lines}" ]]; then
  echo "No operators selected for manual validation." >&2
  exit 2
fi

while IFS=$'\t' read -r operator example_path; do
  args=()
  if [[ "${run_host}" == "1" ]]; then
    args+=(--run-host)
  fi
  if [[ "${emit_artifacts}" == "1" ]]; then
    args+=(--emit-mlir --emit-llvm --emit-asm --emit-object)
  fi
  args+=(--output-dir "${validation_output_root%/}/tilelang-riscv-${operator}")
  if [[ "${run_qemu}" == "1" ]]; then
    args+=(--run-qemu)
  fi
  "${python_bin}" "${example_path}" "${args[@]}"
done <<< "${operator_lines}"
