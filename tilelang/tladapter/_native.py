"""Tool-backed compatibility layer for MLIR pass execution."""

from __future__ import annotations

import subprocess

from .toolchain import resolve_optimizer


class PassPipeline:
    def __init__(self):
        self._passes: list[str] = []
        self._enable_ir_printing = False

    def add(self, pipeline_text: str) -> None:
        self._passes.append(str(pipeline_text))

    def enable_ir_printing(self) -> None:
        self._enable_ir_printing = True

    def _pipeline_text(self) -> str:
        if not self._passes:
            return ""
        if len(self._passes) == 1 and self._passes[0].lstrip().startswith("builtin.module("):
            return self._passes[0]
        return f"builtin.module({','.join(self._passes)})"

    def run(self, mlir_str: str) -> str:
        if not self._passes:
            return mlir_str

        optimizer = resolve_optimizer()
        cmd = [
            str(optimizer),
            f"--pass-pipeline={self._pipeline_text()}",
        ]
        if self._enable_ir_printing:
            cmd.extend(["--mlir-print-ir-after-all", "--mlir-print-ir-module-scope"])

        proc = subprocess.run(
            cmd,
            input=mlir_str,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip() or "unknown optimizer failure"
            raise RuntimeError(f"{optimizer} failed: {message}")
        return proc.stdout

    def __str__(self) -> str:
        return self._pipeline_text() or "builtin.module()"

    def __repr__(self) -> str:
        return f"PassPipeline({self})"
