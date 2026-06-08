from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tilelang.jit.adapter.riscv import resolve_riscv_runner


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = (
    ("vector_add", REPO_ROOT / "examples" / "riscv" / "example_vector_add.py"),
    ("reduce_sum", REPO_ROOT / "examples" / "riscv" / "example_reduce_sum.py"),
    ("matmul", REPO_ROOT / "examples" / "riscv" / "example_matmul.py"),
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


@pytest.mark.parametrize("example_name,example_path", EXAMPLES)
def test_riscv_qemu_smoke_example_is_present(example_name, example_path):
    assert example_path.is_file(), example_name


@pytest.mark.parametrize("example_name,example_path", EXAMPLES)
def test_riscv_examples_run_on_qemu(example_name, example_path, tmp_path):
    pytest.importorskip("tilelang.tladapter._native")
    if resolve_riscv_runner(required=False) is None:
        pytest.skip("qemu/spike runner not available on this machine")

    output_dir = tmp_path / f"{example_name}_qemu"
    result = subprocess.run(
        [sys.executable, str(example_path), "--run-qemu", "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{example_name} qemu smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    assert "qemu check passed" in result.stdout
    assert (output_dir / f"{example_name}.qemu.elf").is_file()
