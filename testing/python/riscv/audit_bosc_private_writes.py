"""Run the upstream RISC-V tests and compare Buddy diagnostics on each actual codegen call.

From an activated SG2044 environment:
    python testing/python/riscv/audit_bosc_private_writes.py --report /tmp/bosc.json
Pass pytest paths or node IDs after -- to select a subset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch


def verify(buddy: str, source: str) -> dict:
    result = subprocess.run(
        [
            buddy,
            "--mlir-disable-threading",
            "--canonicalize",
            "--cse",
            "--kernel-contract-annotate",
            "--kernel-contract-verify",
            "-o",
            "/dev/null",
        ],
        input=source,
        capture_output=True,
        text=True,
        timeout=120,
    )
    diagnostic = next((line.partition("error:")[2].strip() for line in result.stderr.splitlines() if "error:" in line), None)
    return {"exit_code": result.returncode, "diagnostic": diagnostic or result.stderr.strip() or None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--buddy-opt", default="buddy-opt")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    buddy = shutil.which(args.buddy_opt)
    if not buddy:
        parser.error(f"Cannot find {args.buddy_opt}")
    tests = args.pytest_args
    if tests[:1] == ["--"]:
        tests = tests[1:]
    if not tests:
        tests = [str(Path(__file__).resolve().parent / name) for name in ("TileOps", "TileKernels")]

    # Set before importing TileLang. A persistent JIT cache hit would hide the
    # codegen call and could execute a binary built with different options.
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    import pytest
    from tilelang import tvm

    lower = importlib.import_module("tilelang.engine.lower")
    modules = {}
    cases = {}

    class ObserveCases:
        def pytest_runtest_setup(self, item):
            self.current = item.nodeid
            cases[self.current] = []

    observer = ObserveCases()
    observer.current = "<collection>"

    def compare(codegen):
        def emit(device_mod, target):
            if target.kind.name != "riscv":
                return codegen(device_mod, target)
            context = tvm.transform.PassContext.current()
            config = dict(context.config)
            sources = []
            for enabled in (False, True):
                config["tl.riscv_structured_private_writes"] = enabled
                with tvm.transform.PassContext(opt_level=context.opt_level, config=config):
                    compiled = codegen(device_mod, target)
                    sources.append(compiled.inspect_source())
            digests = [hashlib.sha256(s.encode()).hexdigest() for s in sources]
            key = "/".join(digests)
            cases.setdefault(observer.current, []).append(key)
            if key not in modules:
                modules[key] = {label: verify(buddy, src) for label, src in zip(("before", "after"), sources)}
            return compiled  # Execute the enabled variant through the normal test driver.

        return emit

    with (
        tempfile.TemporaryDirectory(prefix="tilelang-bosc-audit-") as cache,
        patch.dict(os.environ, TILELANG_CACHE_DIR=cache),
        patch.object(lower, "device_codegen", compare(lower.device_codegen)),
        patch.object(lower, "device_codegen_without_compile", compare(lower.device_codegen_without_compile)),
    ):
        status = pytest.main(tests, plugins=[observer])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"pytest_exit": int(status), "cases": cases, "modules": modules}, indent=2) + "\n")
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
