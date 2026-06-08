#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / ".agents" / "riscv_operator_workflow" / "operators.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".agents" / "tasks" / "riscv"


def _load_catalog(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _operator_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in catalog.get("operators", [])}


def _render_list(title: str, values: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.extend(f"- {value}" for value in values)
    lines.append("")
    return lines


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def render_task(catalog: dict[str, Any], operator: dict[str, Any]) -> str:
    common = catalog["common_context"]
    lines = [
        f"# {operator['task_title']}",
        "",
        "## Goal",
        "",
        (
            f"Complete or refine the TileLang-RISCV implementation for `{operator['name']}` and validate it through "
            "task generation, x86/arm host execution, RISC-V artifact export, and QEMU execution when the toolchain is available."
        ),
        "",
        "## Operator Semantics",
        "",
        operator["semantics"],
        "",
    ]
    lines.extend(_render_list("Inputs", operator["inputs"]))
    lines.extend(_render_list("Outputs", operator["outputs"]))
    lines.extend(_render_list("Baseline Reference", operator["baseline_reference"]))
    lines.extend(_render_list("Baseline Implementation", operator["baseline_implementation"]))
    lines.extend(
        _render_list(
            "Context Files",
            _unique([
                operator["current_example"],
                *operator["suggested_files"],
                *common["primary_docs"],
                *common["reference_tests"],
            ]),
        )
    )
    lines.extend(_render_list("Scheduling Strategy", operator["schedule_strategy"]))
    lines.extend(_render_list("Compilation Commands", operator["compile_commands"]))
    lines.extend(_render_list("Test Commands", operator["test_commands"]))
    lines.extend(_render_list("Success Criteria", operator["success_criteria"]))
    lines.extend(_render_list("Failure Triage Buckets", common["common_failure_buckets"]))
    if common.get("sg2044_baseline"):
        lines.extend(_render_list("Existing SG2044 Baseline", common["sg2044_baseline"]))
    lines.extend(
        [
            "## Reporting Template",
            "",
            "- Operator:",
            "- Implementation files changed:",
            "- Host compile/run result:",
            "- RISC-V artifact result:",
            "- QEMU result:",
            "- Optional SG2044 replay result:",
            "- Failure bucket, if any:",
            "- Follow-up needed:",
            "",
            "## Target Flow",
            "",
            common["riscv_flow"],
            "",
            catalog["validation_note"],
            "",
        ]
    )
    return "\n".join(lines)


def write_task(catalog: dict[str, Any], operator: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{operator['name']}.md"
    path.write_text(render_task(catalog, operator), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Codex task files for TileLang-RISCV operators.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--operator", help="Operator name to render.")
    parser.add_argument("--all", action="store_true", help="Render all operators in the catalog.")
    parser.add_argument("--list", action="store_true", help="List available operator names.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    catalog = _load_catalog(args.catalog)
    operators = _operator_map(catalog)

    if args.list:
        for name in sorted(operators):
            print(name)
        return

    if args.all:
        selected = [operators[name] for name in sorted(operators)]
    elif args.operator:
        if args.operator not in operators:
            choices = ", ".join(sorted(operators))
            raise SystemExit(f"Unknown operator `{args.operator}`. Available operators: {choices}")
        selected = [operators[args.operator]]
    else:
        raise SystemExit("Pass --operator NAME, --all, or --list.")

    for operator in selected:
        print(write_task(catalog, operator, args.output_dir))


if __name__ == "__main__":
    main()
