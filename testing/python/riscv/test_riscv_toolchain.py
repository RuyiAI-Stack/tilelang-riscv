from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tilelang.tladapter.toolchain as toolchain
from tilelang.cache.kernel_cache import KernelCache
from tvm.target import Target
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
def test_find_compiler_rt_builtins_per_target_and_vendor_normalized(monkeypatch, tmp_path):
    clang_path = tmp_path / "llvm" / "bin" / "clang"
    clang_path.parent.mkdir(parents=True)
    clang_path.write_text("#!/usr/bin/env bash\n")
    clang_path.chmod(0o755)

    resource_dir = tmp_path / "llvm" / "lib" / "clang" / "24"
    builtins = resource_dir / "lib" / "riscv64-unknown-linux-gnu" / "libclang_rt.builtins.a"
    builtins.parent.mkdir(parents=True)
    builtins.write_text("")

    monkeypatch.setattr(
        riscv_wrapper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=f"{resource_dir}\n", stderr=""),
    )

    assert riscv_wrapper._find_compiler_rt_builtins(clang_path, "riscv64-unknown-linux-gnu") == builtins
    assert riscv_wrapper._find_compiler_rt_builtins(clang_path, "riscv64-openEuler-linux-gnu") == builtins
    assert riscv_wrapper._find_compiler_rt_builtins(clang_path, "riscv64-openEuler-linux") == builtins
    assert riscv_wrapper._find_compiler_rt_builtins(clang_path, "x86_64-unknown-linux-gnu") is None
    assert riscv_wrapper._compiler_rt_builtins_flags(clang_path, "riscv64-openEuler-linux") == [
        f"-L{builtins.parent}",
        "-lclang_rt.builtins",
    ]


def test_find_compiler_rt_builtins_legacy_linux_layout(monkeypatch, tmp_path):
    clang_path = tmp_path / "llvm" / "bin" / "clang"
    clang_path.parent.mkdir(parents=True)
    clang_path.write_text("#!/usr/bin/env bash\n")
    clang_path.chmod(0o755)

    resource_dir = tmp_path / "llvm" / "lib" / "clang" / "24"
    builtins = resource_dir / "lib" / "linux" / "libclang_rt.builtins-riscv64.a"
    builtins.parent.mkdir(parents=True)
    builtins.write_text("")

    monkeypatch.setattr(
        riscv_wrapper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=f"{resource_dir}\n", stderr=""),
    )

    assert riscv_wrapper._find_compiler_rt_builtins(clang_path, "riscv64-unknown-linux-gnu") == builtins
    assert riscv_wrapper._compiler_rt_builtins_flags(clang_path, "riscv64-unknown-linux-gnu") == [
        f"-L{builtins.parent}",
        "-lclang_rt.builtins-riscv64",
    ]
    assert riscv_wrapper._find_compiler_rt_builtins(clang_path, "x86_64-unknown-linux-gnu") is None


def test_build_host_shared_library_links_compiler_rt_and_no_undefined(monkeypatch, tmp_path):
    llvm_root = tmp_path / "llvm"
    clang_path = llvm_root / "bin" / "clang"
    clang_path.parent.mkdir(parents=True)
    clang_path.write_text("#!/usr/bin/env bash\n")
    clang_path.chmod(0o755)

    resource_dir = llvm_root / "lib" / "clang" / "24"
    builtins = resource_dir / "lib" / "riscv64-unknown-linux-gnu" / "libclang_rt.builtins.a"
    builtins.parent.mkdir(parents=True)
    builtins.write_text("")

    for env_name in (
        "TILELANG_RISCV_GCC_ROOT",
        "GCC_ROOT",
        "CC",
        "CXX",
        "TILELANG_RISCV_CLANG_FLAGS",
        "TILELANG_RISCV_ABI",
        "TILELANG_RISCV_MARCH",
        "TILELANG_RISCV_CPU",
        "TILELANG_RISCV_CLANG_OPT_LEVEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setattr(riscv_wrapper, "emit_llvm_ir", lambda value, pipeline=None: "define void @k() { ret void }\n")
    monkeypatch.setattr(riscv_wrapper, "resolve_llvm_root", lambda: llvm_root)
    monkeypatch.setattr(riscv_wrapper, "resolve_tool", lambda name: clang_path)
    monkeypatch.setattr(
        riscv_wrapper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=f"{resource_dir}\n", stderr=""),
    )

    captured_cmds: list[list[str]] = []
    monkeypatch.setattr(
        riscv_wrapper,
        "_run_checked",
        lambda cmd: captured_cmds.append(list(cmd)) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    riscv_wrapper.build_host_shared_library("dummy", tmp_path / "out_linux.so", triple="riscv64-unknown-linux-gnu")
    linux_cmd = captured_cmds[-1]
    assert "-Wl,--no-undefined" in linux_cmd
    assert f"-L{builtins.parent}" in linux_cmd
    assert "-lclang_rt.builtins" in linux_cmd
    assert linux_cmd.index("-lm") < linux_cmd.index(f"-L{builtins.parent}") < linux_cmd.index("-lclang_rt.builtins")

    riscv_wrapper.build_host_shared_library("dummy", tmp_path / "out_darwin.so", triple="arm64-apple-darwin23.6.0")
    darwin_cmd = captured_cmds[-1]
    assert "-Wl,--no-undefined" not in darwin_cmd

    monkeypatch.setenv("TILELANG_RISCV_CLANG_FLAGS", "-L/custom/lib -lclang_rt.builtins")
    riscv_wrapper.build_host_shared_library("dummy", tmp_path / "out_custom.so", triple="riscv64-unknown-linux-gnu")
    custom_cmd = captured_cmds[-1]
    assert custom_cmd.count("-lclang_rt.builtins") == 1
    assert f"-L{builtins.parent}" not in custom_cmd

    monkeypatch.setenv("TILELANG_RISCV_CLANG_FLAGS", "-rtlib=compiler-rt")
    riscv_wrapper.build_host_shared_library("dummy", tmp_path / "out_rtlib.so", triple="riscv64-unknown-linux-gnu")
    rtlib_cmd = captured_cmds[-1]
    assert "-rtlib=compiler-rt" in rtlib_cmd
    assert "-lclang_rt.builtins" not in rtlib_cmd


def test_riscv_cache_key_tracks_toolchain_env_and_builtins_in_process(monkeypatch, tmp_path):
    llvm_root = tmp_path / "llvm"
    clang_path = llvm_root / "bin" / "clang"
    clang_path.parent.mkdir(parents=True)
    clang_path.write_text("#!/usr/bin/env bash\n")
    clang_path.chmod(0o755)

    resource_dir = llvm_root / "lib" / "clang" / "24"
    builtins_dir = resource_dir / "lib" / "riscv64-unknown-linux-gnu"
    builtins_dir.mkdir(parents=True)

    for env_name in (
        "TILELANG_RISCV_CLANG_FLAGS",
        "TILELANG_RISCV_GCC_ROOT",
        "GCC_ROOT",
        "TILELANG_HOST_TRIPLE",
        "TILELANG_RISCV_ABI",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setattr(riscv_wrapper, "resolve_tool", lambda name, required=True: clang_path if name == "clang" else None)

    def _fake_run(cmd, *args, **kwargs):
        if "-print-resource-dir" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"{resource_dir}\n", stderr="")
        if "-dumpmachine" in cmd:
            return SimpleNamespace(returncode=0, stdout="riscv64-unknown-linux-gnu\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(riscv_wrapper.subprocess, "run", _fake_run)

    fake_func = SimpleNamespace(script=lambda show_meta=True: "@T.prim_func\ndef stub():\n    pass\n")
    cache = object.__new__(KernelCache)

    key_initial = cache._generate_key(fake_func, [1], args=(), target="riscv")

    monkeypatch.setenv("TILELANG_RISCV_CLANG_FLAGS", "-lclang_rt.builtins")
    key_flags = cache._generate_key(fake_func, [1], args=(), target="riscv")
    assert key_flags != key_initial

    gcc_root = tmp_path / "gcc-native"
    gcc_root.mkdir()
    monkeypatch.setenv("TILELANG_RISCV_GCC_ROOT", str(gcc_root))
    key_gcc = cache._generate_key(fake_func, [1], args=(), target="riscv")
    assert key_gcc != key_flags

    (builtins_dir / "libclang_rt.builtins.a").write_bytes(b"archive")
    key_builtins = cache._generate_key(fake_func, [1], args=(), target="riscv")
    assert key_builtins != key_gcc

    riscv_target_key_1 = cache._generate_key(fake_func, [1], args=(), target=Target("riscv"))
    monkeypatch.setenv("TILELANG_RISCV_CLANG_FLAGS", "-O0 -DRISCV_OBJ")
    riscv_target_key_2 = cache._generate_key(fake_func, [1], args=(), target=Target("riscv"))
    assert riscv_target_key_1 != riscv_target_key_2

    cuda_key_1 = cache._generate_key(fake_func, [1], args=(), target="cuda")
    llvm_key_1 = cache._generate_key(fake_func, [1], args=(), target=Target("llvm"))
    monkeypatch.setenv("TILELANG_RISCV_CLANG_FLAGS", "-O0 -DMUTATED")
    cuda_key_2 = cache._generate_key(fake_func, [1], args=(), target="cuda")
    llvm_key_2 = cache._generate_key(fake_func, [1], args=(), target=Target("llvm"))
    assert cuda_key_1 == cuda_key_2
    assert llvm_key_1 == llvm_key_2
