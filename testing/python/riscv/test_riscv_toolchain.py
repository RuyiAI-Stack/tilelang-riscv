from __future__ import annotations

import os
import importlib
from pathlib import Path

import tilelang.tladapter.toolchain as toolchain
from tilelang.tladapter.toolchain import (
    ToolchainNotFoundError,
    resolve_llvm_dir,
    resolve_llvm_root,
    resolve_mlir_dir,
    resolve_mlir_python_root,
    resolve_tool,
)

riscv_wrapper = importlib.import_module("tilelang.jit.adapter.riscv.wrapper")


def _make_fake_toolchain(root: Path) -> None:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "cmake" / "mlir").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "cmake" / "llvm").mkdir(parents=True, exist_ok=True)
    (root / "python_packages" / "mlir_core" / "mlir").mkdir(parents=True, exist_ok=True)
    for tool_name in ("mlir-opt", "mlir-translate", "llc", "clang"):
        tool_path = root / "bin" / tool_name
        tool_path.write_text("#!/usr/bin/env bash\n")
        tool_path.chmod(0o755)


def test_toolchain_env_root_wins(monkeypatch, tmp_path):
    fake_root = tmp_path / "llvm-install"
    _make_fake_toolchain(fake_root)
    monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(fake_root))
    monkeypatch.delenv("TILELANG_LLVM_INSTALL_DIR", raising=False)

    assert resolve_llvm_root() == fake_root.resolve()
    assert resolve_mlir_dir() == (fake_root / "lib" / "cmake" / "mlir").resolve()
    assert resolve_llvm_dir() == (fake_root / "lib" / "cmake" / "llvm").resolve()
    assert resolve_mlir_python_root() == (fake_root / "python_packages" / "mlir_core").resolve()
    assert resolve_tool("mlir-opt") == (fake_root / "bin" / "mlir-opt").resolve()


def test_tool_lookup_falls_back_to_path(monkeypatch, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_tool = fake_bin / "mlir-opt"
    fake_tool.write_text("#!/usr/bin/env bash\n")
    fake_tool.chmod(0o755)

    monkeypatch.setattr(toolchain, "_LLVM_SUBMODULE_ROOT", tmp_path / "missing-llvm-project")
    monkeypatch.setattr(toolchain, "_BUDDY_MLIR_ROOT", tmp_path / "missing-buddy-mlir")
    monkeypatch.delenv("TILELANG_RISCV_LLVM_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_LLVM_INSTALL_DIR", raising=False)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    assert resolve_tool("mlir-opt") == fake_tool.resolve()


def test_missing_toolchain_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(toolchain, "_LLVM_SUBMODULE_ROOT", tmp_path / "missing-toolchain")
    monkeypatch.setattr(toolchain, "_BUDDY_MLIR_ROOT", tmp_path / "missing-buddy")
    monkeypatch.delenv("TILELANG_RISCV_LLVM_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_LLVM_INSTALL_DIR", raising=False)

    try:
        resolve_llvm_root(required=True)
    except ToolchainNotFoundError:
        return

    raise AssertionError("resolve_llvm_root should have raised ToolchainNotFoundError")


def test_riscv_clang_flags_adds_gcc_runtime_multilib_dirs(monkeypatch, tmp_path):
    fake_gcc_root = tmp_path / "gcc-native"
    runtime_dir = fake_gcc_root / "lib64" / "lp64d"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "libgcc_s.so").write_text("")
    (fake_gcc_root / "sysroot").mkdir()

    monkeypatch.setenv("TILELANG_RISCV_GCC_ROOT", str(fake_gcc_root))
    monkeypatch.delenv("GCC_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_RISCV_ABI", raising=False)
    monkeypatch.delenv("TILELANG_RISCV_CLANG_FLAGS", raising=False)

    flags = riscv_wrapper._riscv_clang_flags(include_runtime_library_dirs=True)

    assert f"--gcc-toolchain={fake_gcc_root.resolve()}" in flags
    assert f"--sysroot={fake_gcc_root.resolve() / 'sysroot'}" in flags
    assert f"-L{runtime_dir}" in flags
    assert f"-Wl,-rpath,{runtime_dir}" in flags
