from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("tilelang.tladapter._native")


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "riscv"
INITIAL_WORKFLOW_EXAMPLES = (
    "example_vector_add.py",
    "example_reduce_sum.py",
    "example_matmul.py",
)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    ld_parts = [
        str(REPO_ROOT / "build" / "lib"),
        str(REPO_ROOT / "3rdparty" / "llvm-project" / "install" / "lib"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


@pytest.mark.parametrize("example_name", INITIAL_WORKFLOW_EXAMPLES)
def test_riscv_operator_workflow_examples_run_on_host(example_name, tmp_path):
    example_path = EXAMPLE_ROOT / example_name
    assert example_path.is_file(), example_name

    result = subprocess.run(
        [sys.executable, str(example_path), "--run-host", "--output-dir", str(tmp_path / example_name)],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{example_name} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "host check passed" in result.stdout


@pytest.mark.parametrize("example_name", INITIAL_WORKFLOW_EXAMPLES)
def test_riscv_operator_workflow_examples_emit_full_artifacts(example_name, tmp_path):
    example_path = EXAMPLE_ROOT / example_name
    assert example_path.is_file(), example_name

    output_dir = tmp_path / example_name
    result = subprocess.run(
        [
            sys.executable,
            str(example_path),
            "--emit-mlir",
            "--emit-llvm",
            "--emit-asm",
            "--emit-object",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{example_name} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    mlir_candidates = list(output_dir.glob("*.mlir"))
    llvm_candidates = list(output_dir.glob("*.ll"))
    asm_candidates = list(output_dir.glob("*.s"))
    obj_candidates = list(output_dir.glob("*.o"))
    if len(mlir_candidates) != 1:
        raise AssertionError(f"{example_name} should emit exactly one MLIR file, found {mlir_candidates}")
    if len(llvm_candidates) != 1:
        raise AssertionError(f"{example_name} should emit exactly one LLVM IR file, found {llvm_candidates}")
    if len(asm_candidates) != 1:
        raise AssertionError(f"{example_name} should emit exactly one asm file, found {asm_candidates}")
    if len(obj_candidates) != 1:
        raise AssertionError(f"{example_name} should emit exactly one object file, found {obj_candidates}")
    assert "func.func" in mlir_candidates[0].read_text()
    assert "define void" in llvm_candidates[0].read_text()
    assert asm_candidates[0].stat().st_size > 0
    assert obj_candidates[0].stat().st_size > 0
