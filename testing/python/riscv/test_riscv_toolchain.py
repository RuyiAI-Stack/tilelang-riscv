from __future__ import annotations

import os
import importlib
from pathlib import Path

import pytest

import tilelang.tladapter.toolchain as toolchain
from tilelang.tladapter.toolchain import (
    ToolchainNotFoundError,
    resolve_buddy_root,
    resolve_llvm_dir,
    resolve_llvm_root,
    resolve_mlir_dir,
    resolve_mlir_python_root,
    resolve_optimizer,
    resolve_tool,
    toolchain_summary,
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


def _make_fake_buddy_toolchain(root: Path) -> None:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    for tool_name in ("buddy-opt", "buddy-translate"):
        executable = root / "bin" / tool_name
        executable.write_text("#!/usr/bin/env bash\n")
        executable.chmod(0o755)


def test_optimizer_defaults_to_buddy_root_and_keeps_llvm_tools_separate(monkeypatch, tmp_path):
    llvm_root = tmp_path / "llvm"
    buddy_root = tmp_path / "buddy-mlir"
    _make_fake_toolchain(llvm_root)
    _make_fake_buddy_toolchain(buddy_root)
    # Other tools on PATH must not override the configured toolchain roots.
    monkeypatch.setenv("PATH", str(llvm_root / "bin"))
    monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(llvm_root))
    monkeypatch.setenv("TILELANG_RISCV_BUDDY_ROOT", str(buddy_root))
    monkeypatch.delenv("TILELANG_RISCV_OPT", raising=False)

    assert resolve_buddy_root() == buddy_root.resolve()
    assert resolve_optimizer() == (buddy_root / "bin" / "buddy-opt").resolve()
    assert resolve_tool("buddy-translate") == (buddy_root / "bin" / "buddy-translate").resolve()
    assert resolve_tool("mlir-translate") == (llvm_root / "bin" / "mlir-translate").resolve()
    assert resolve_tool("clang") == (llvm_root / "bin" / "clang").resolve()
    summary = toolchain_summary()
    assert summary["buddy_root"] == str(buddy_root.resolve())
    assert summary["llvm_root"] == str(llvm_root.resolve())
    assert summary["optimizer"] == summary["buddy_opt"]

    monkeypatch.setenv("TILELANG_RISCV_OPT", "mlir-opt")
    assert resolve_optimizer() == (llvm_root / "bin" / "mlir-opt").resolve()
    assert toolchain_summary()["optimizer"] == str(resolve_optimizer())


@pytest.mark.parametrize("layout", ["release", "build", "install"])
def test_buddy_root_discovers_sibling_release_or_source_build(monkeypatch, tmp_path, layout):
    sibling = tmp_path / "buddy-mlir"
    root = sibling if layout == "release" else sibling / layout
    _make_fake_buddy_toolchain(root)
    monkeypatch.setattr(toolchain, "_BUDDY_MLIR_ROOT", sibling)
    monkeypatch.delenv("TILELANG_RISCV_BUDDY_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_RISCV_OPT", raising=False)

    assert resolve_buddy_root() == root.resolve()
    assert resolve_optimizer() == (root / "bin" / "buddy-opt").resolve()


def test_buddy_optimizer_falls_back_to_path(monkeypatch, tmp_path):
    buddy_root = tmp_path / "buddy-mlir"
    _make_fake_buddy_toolchain(buddy_root)
    monkeypatch.setattr(toolchain, "_BUDDY_MLIR_ROOT", tmp_path / "missing")
    monkeypatch.delenv("TILELANG_RISCV_BUDDY_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_RISCV_OPT", raising=False)
    monkeypatch.setenv("PATH", str(buddy_root / "bin"))

    assert resolve_optimizer() == (buddy_root / "bin" / "buddy-opt").resolve()


def test_missing_buddy_does_not_silently_select_mlir_opt(monkeypatch, tmp_path):
    llvm_root = tmp_path / "llvm"
    _make_fake_toolchain(llvm_root)
    monkeypatch.setattr(toolchain, "_BUDDY_MLIR_ROOT", tmp_path / "missing")
    monkeypatch.delenv("TILELANG_RISCV_BUDDY_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_RISCV_OPT", raising=False)
    monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(llvm_root))
    monkeypatch.setenv("PATH", str(llvm_root / "bin"))

    assert resolve_buddy_root(required=False) is None
    assert resolve_optimizer(required=False) is None
    with pytest.raises(ToolchainNotFoundError, match="TILELANG_RISCV_BUDDY_ROOT"):
        resolve_buddy_root()
    with pytest.raises(ToolchainNotFoundError, match="TILELANG_RISCV_OPT=mlir-opt"):
        resolve_optimizer()
    monkeypatch.setenv("TILELANG_RISCV_OPT", "mlir-opt")
    assert resolve_optimizer() == (llvm_root / "bin" / "mlir-opt").resolve()


@pytest.mark.parametrize("selection", ["absolute", "relative", "name"])
def test_optimizer_override_selects_buddy_opt(monkeypatch, tmp_path, selection):
    llvm_root = tmp_path / "llvm"
    buddy_root = tmp_path / "buddy-mlir with spaces"
    _make_fake_toolchain(llvm_root)
    _make_fake_buddy_toolchain(buddy_root)
    buddy_opt = buddy_root / "bin" / "buddy-opt"
    monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(llvm_root))
    monkeypatch.setenv("TILELANG_RISCV_BUDDY_ROOT", str(buddy_root))
    if selection == "absolute":
        override = str(buddy_opt)
    elif selection == "relative":
        monkeypatch.chdir(tmp_path)
        override = str(buddy_opt.relative_to(tmp_path))
    else:
        override = "buddy-opt"
    monkeypatch.setenv("TILELANG_RISCV_OPT", override)

    assert resolve_optimizer() == buddy_opt.resolve()
    assert toolchain_summary()["optimizer"] == str(buddy_opt.resolve())
    assert resolve_tool("mlir-opt") == (llvm_root / "bin" / "mlir-opt").resolve()


@pytest.mark.parametrize("invalid_tool", ["missing", "not-executable", "directory"])
def test_optimizer_override_fails_without_falling_back(monkeypatch, tmp_path, invalid_tool):
    llvm_root = tmp_path / "llvm"
    buddy_root = tmp_path / "buddy-mlir"
    _make_fake_toolchain(llvm_root)
    _make_fake_buddy_toolchain(buddy_root)
    buddy_opt = tmp_path / invalid_tool
    if invalid_tool == "not-executable":
        buddy_opt.write_text("#!/usr/bin/env bash\n")
        buddy_opt.chmod(0o644)
    elif invalid_tool == "directory":
        buddy_opt.mkdir()
    monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(llvm_root))
    monkeypatch.setenv("TILELANG_RISCV_BUDDY_ROOT", str(buddy_root))
    monkeypatch.setenv("TILELANG_RISCV_OPT", str(buddy_opt))

    assert resolve_optimizer(required=False) is None
    with pytest.raises(ToolchainNotFoundError, match="TILELANG_RISCV_OPT"):
        resolve_optimizer()
