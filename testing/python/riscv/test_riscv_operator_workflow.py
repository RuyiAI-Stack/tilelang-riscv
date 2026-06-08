from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / ".agents" / "riscv_operator_workflow" / "operators.json"
TASK_ROOT = REPO_ROOT / ".agents" / "tasks" / "riscv"
GENERATOR_PATH = REPO_ROOT / "maint" / "scripts" / "generate_riscv_operator_task.py"
VALIDATION_SCRIPT = REPO_ROOT / "maint" / "scripts" / "run_riscv_operator_workflow_validation.sh"

REQUIRED_OPERATOR_FIELDS = {
    "name",
    "category",
    "task_title",
    "semantics",
    "inputs",
    "outputs",
    "baseline_reference",
    "baseline_implementation",
    "current_example",
    "suggested_files",
    "schedule_strategy",
    "compile_commands",
    "test_commands",
    "success_criteria",
}
REQUIRED_TASK_SECTIONS = (
    "## Goal",
    "## Operator Semantics",
    "## Inputs",
    "## Outputs",
    "## Baseline Reference",
    "## Baseline Implementation",
    "## Context Files",
    "## Scheduling Strategy",
    "## Compilation Commands",
    "## Test Commands",
    "## Success Criteria",
    "## Failure Triage Buckets",
    "## Existing SG2044 Baseline",
    "## Reporting Template",
    "## Target Flow",
)


def _load_catalog() -> dict:
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_riscv_operator_task", GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_riscv_operator_catalog_covers_initial_validation_patterns():
    catalog = _load_catalog()
    operators = catalog["operators"]

    assert catalog["target"] == "riscv"
    assert catalog["target_alias"] == "linalg_riscv"
    assert {operator["name"] for operator in operators} == {"vector_add", "reduce_sum", "matmul"}
    assert {operator["category"] for operator in operators} == {"elementwise", "reduction", "matrix"}

    for operator in operators:
        assert REQUIRED_OPERATOR_FIELDS <= set(operator)
        for field in REQUIRED_OPERATOR_FIELDS - {"name", "category", "task_title", "semantics", "current_example"}:
            assert operator[field], f"{operator['name']} has empty {field}"


def test_riscv_operator_context_files_exist():
    catalog = _load_catalog()
    common = catalog["common_context"]
    common_paths = [*common["primary_docs"], *common["reference_tests"]]

    for operator in catalog["operators"]:
        paths = [operator["current_example"], *operator["suggested_files"], *common_paths]
        for path in paths:
            assert (REPO_ROOT / path).is_file(), f"{operator['name']} references missing context file {path}"


def test_riscv_operator_tasks_are_up_to_date_with_generator():
    catalog = _load_catalog()
    generator = _load_generator()

    for operator in catalog["operators"]:
        expected = generator.render_task(catalog, operator)
        actual_path = TASK_ROOT / f"{operator['name']}.md"
        assert actual_path.is_file()
        assert actual_path.read_text(encoding="utf-8") == expected


def test_riscv_operator_generated_tasks_include_required_sections(tmp_path):
    result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--all", "--output-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    generated = sorted(path.name for path in tmp_path.glob("*.md"))
    assert generated == ["matmul.md", "reduce_sum.md", "vector_add.md"]
    for path in tmp_path.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        for section in REQUIRED_TASK_SECTIONS:
            assert section in content, f"{path.name} is missing {section}"


def test_riscv_operator_generator_rejects_unknown_operator():
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR_PATH),
            "--operator",
            "not_a_real_operator",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unknown operator" in result.stderr
    assert "vector_add" in result.stderr
    assert "reduce_sum" in result.stderr
    assert "matmul" in result.stderr


def test_riscv_operator_validation_script_is_bash_parseable():
    if os.name == "nt":
        pytest.skip("bash -n validation is covered on Linux/WSL")

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")

    result = subprocess.run(
        [bash, "-n", str(VALIDATION_SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_riscv_operator_validation_script_rejects_unknown_operator():
    if os.name == "nt":
        pytest.skip("bash validation is covered on Linux/WSL")

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")

    env = os.environ.copy()
    env.update({
        "PYTHON": sys.executable,
        "TILELANG_RISCV_RUN_PYTEST": "0",
        "TILELANG_RISCV_RUN_QEMU": "0",
        "TILELANG_RISCV_RUN_HOST": "1",
        "TILELANG_RISCV_EMIT_ARTIFACTS": "0",
        "TILELANG_RISCV_OPERATORS": "not_a_real_operator",
    })
    result = subprocess.run(
        [bash, str(VALIDATION_SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert result.returncode != 0
    assert "Unknown TILELANG_RISCV_OPERATORS entries: not_a_real_operator" in result.stderr
