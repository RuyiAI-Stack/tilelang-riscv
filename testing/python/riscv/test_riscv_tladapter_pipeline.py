from __future__ import annotations

import json
import sys

import pytest

import tilelang.tladapter as tladapter
from tilelang.tladapter.transforms import mlir as mlir_passes


pytest.importorskip("tilelang.tladapter._native")


def test_tladapter_pipeline_runs_canonicalize():
    source = """module {
  func.func @kernel() {
    return
  }
}
"""

    pipeline = tladapter.Pipeline()
    pipeline.add(mlir_passes.canonicalize)
    pipeline.add(mlir_passes.cse)

    lowered = pipeline.run(source)

    assert "func.func @kernel()" in lowered
    assert str(pipeline).startswith("builtin.module(")


@pytest.mark.parametrize("tool_name", ["buddy-opt", "mlir-opt"])
@pytest.mark.parametrize("fails", [False, True])
def test_tladapter_pipeline_executes_selected_optimizer(monkeypatch, tmp_path, fails, tool_name):
    optimizer = tmp_path / ("buddy-mlir" if tool_name == "buddy-opt" else "llvm") / "bin" / tool_name
    optimizer.parent.mkdir(parents=True)
    record = tmp_path / "invocation.json"
    optimizer.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "source = sys.stdin.read()\n"
        f"Path({str(record)!r}).write_text(json.dumps({{'args': sys.argv[1:], 'source': source}}))\n"
        + (
            "sys.stderr.write('selected optimizer rejected the pass')\nsys.exit(1)\n"
            if fails
            else "sys.stdout.write(source.replace('@kernel', '@optimized_kernel'))\n"
        )
    )
    optimizer.chmod(0o755)
    if tool_name == "buddy-opt":
        monkeypatch.setenv("TILELANG_RISCV_BUDDY_ROOT", str(optimizer.parents[1]))
        monkeypatch.delenv("TILELANG_RISCV_OPT", raising=False)
    else:
        monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(optimizer.parents[1]))
        monkeypatch.setenv("TILELANG_RISCV_OPT", "mlir-opt")
    source = "module { func.func @kernel() { return } }"
    pipeline = tladapter.Pipeline().add(mlir_passes.canonicalize).enable_ir_printing()

    if fails:
        with pytest.raises(RuntimeError, match=f"{tool_name} failed: selected optimizer rejected the pass"):
            pipeline.run(source)
    else:
        assert "@optimized_kernel" in pipeline.run(source)
    invocation = json.loads(record.read_text())
    assert invocation["source"] == source
    assert invocation["args"] == [
        "--pass-pipeline=builtin.module(canonicalize)",
        "--mlir-print-ir-after-all",
        "--mlir-print-ir-module-scope",
    ]
