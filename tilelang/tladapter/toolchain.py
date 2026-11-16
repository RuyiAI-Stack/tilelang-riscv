"""Helpers for locating the external LLVM/MLIR toolchain."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class ToolchainNotFoundError(RuntimeError):
    """Raised when the expected LLVM/MLIR toolchain is unavailable."""


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LLVM_SUBMODULE_ROOT = _REPO_ROOT / "3rdparty" / "llvm-project"
_BUDDY_MLIR_ROOT = _REPO_ROOT.parent / "buddy-mlir"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("TILELANG_RISCV_LLVM_ROOT", "TILELANG_LLVM_INSTALL_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            roots.append(Path(env_value).expanduser())

    roots.extend(
        [
            _BUDDY_MLIR_ROOT / "llvm" / "build",
            _BUDDY_MLIR_ROOT / "llvm" / "install",
            _LLVM_SUBMODULE_ROOT / "install",
            _LLVM_SUBMODULE_ROOT / "build-host" / "install",
            _LLVM_SUBMODULE_ROOT / "build" / "install",
        ]
    )
    return roots


def _candidate_buddy_roots() -> list[Path]:
    roots: list[Path] = []
    env_value = os.environ.get("TILELANG_RISCV_BUDDY_ROOT")
    if env_value:
        roots.append(Path(env_value).expanduser())
    roots.extend([_BUDDY_MLIR_ROOT, _BUDDY_MLIR_ROOT / "build", _BUDDY_MLIR_ROOT / "install"])
    return roots


def _existing_dirs(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_dir() and resolved not in result:
            result.append(resolved)
    return result


def iter_llvm_roots() -> list[Path]:
    """Return known LLVM/MLIR install roots in priority order."""

    return _existing_dirs(_candidate_roots())


def iter_buddy_roots() -> list[Path]:
    """Return known buddy-mlir install and build roots in priority order."""

    return _existing_dirs(_candidate_buddy_roots())


def resolve_buddy_root(*, required: bool = True) -> Path | None:
    """Resolve the buddy-mlir toolchain root, separate from the LLVM root."""

    for root in iter_buddy_roots():
        if (root / "bin").is_dir():
            return root
    if required:
        raise ToolchainNotFoundError(
            "buddy-mlir toolchain not found. Set TILELANG_RISCV_BUDDY_ROOT "
            "to the extracted buddy-mlir directory or its build/install prefix."
        )
    return None


def resolve_llvm_root(*, required: bool = True) -> Path | None:
    """Resolve the preferred LLVM/MLIR install root."""

    for root in iter_llvm_roots():
        if (root / "bin").is_dir():
            return root
    if required:
        raise ToolchainNotFoundError(
            "LLVM/MLIR toolchain not found. Build Buddy's LLVM/MLIR as described in "
            "docs/get_started/BuildOnSG2044.md and set TILELANG_RISCV_LLVM_ROOT "
            "to its build or install prefix."
        )
    return None


def _resolve_cmake_dir(package_name: str, *, required: bool) -> Path | None:
    for root in iter_llvm_roots():
        for lib_dir_name in ("lib", "lib64"):
            cmake_dir = root / lib_dir_name / "cmake" / package_name
            if cmake_dir.is_dir():
                return cmake_dir
    if required:
        raise ToolchainNotFoundError(f"{package_name} CMake package directory not found under the configured LLVM root.")
    return None


def resolve_mlir_dir(*, required: bool = True) -> Path | None:
    """Resolve the MLIR CMake package directory."""

    return _resolve_cmake_dir("mlir", required=required)


def resolve_llvm_dir(*, required: bool = True) -> Path | None:
    """Resolve the LLVM CMake package directory."""

    return _resolve_cmake_dir("llvm", required=required)


def resolve_tool(tool_name: str, *, required: bool = True) -> Path | None:
    """Resolve Buddy tools from buddy-mlir roots, LLVM tools from LLVM roots, then PATH."""

    roots = iter_buddy_roots() if tool_name.startswith("buddy-") else iter_llvm_roots()
    for root in roots:
        candidate = root / "bin" / tool_name
        if candidate.is_file():
            return candidate

    path_hit = shutil.which(tool_name)
    if path_hit:
        return Path(path_hit).resolve()

    if required:
        root_variable = "TILELANG_RISCV_BUDDY_ROOT" if tool_name.startswith("buddy-") else "TILELANG_RISCV_LLVM_ROOT"
        raise ToolchainNotFoundError(f"Tool `{tool_name}` not found. Set {root_variable} or add the tool to PATH.")
    return None


def resolve_optimizer(*, required: bool = True) -> Path | None:
    """Select buddy-opt by default, or the tool named by TILELANG_RISCV_OPT."""

    selection = os.environ.get("TILELANG_RISCV_OPT") or "buddy-opt"
    tool_name = os.path.expanduser(selection)
    if os.path.dirname(tool_name):
        tool_path = shutil.which(tool_name)
    else:
        tool_path = resolve_tool(tool_name, required=False)
    if tool_path is not None and os.access(tool_path, os.X_OK):
        return Path(tool_path).resolve()

    if required:
        raise ToolchainNotFoundError(
            f"Optimizer {selection!r} selected by TILELANG_RISCV_OPT is not executable or was not found. "
            "Set TILELANG_RISCV_BUDDY_ROOT for buddy-opt, TILELANG_RISCV_LLVM_ROOT for mlir-opt, "
            "or add the selected tool to PATH. Set TILELANG_RISCV_OPT=mlir-opt to use the LLVM optimizer."
        )
    return None


def resolve_mlir_python_root(*, required: bool = True) -> Path | None:
    """Resolve the LLVM/MLIR Python package root."""

    for root in iter_llvm_roots():
        candidate = root / "python_packages" / "mlir_core"
        if (candidate / "mlir").is_dir():
            return candidate.resolve()

    if required:
        raise ToolchainNotFoundError(
            "MLIR Python bindings not found. Build Buddy's LLVM/MLIR with "
            "MLIR_ENABLE_BINDINGS_PYTHON=ON as described in "
            "docs/get_started/BuildOnSG2044.md and set TILELANG_RISCV_LLVM_ROOT."
        )
    return None


def toolchain_summary() -> dict[str, str]:
    """Return a small summary of the discovered toolchain paths."""

    summary: dict[str, str] = {}
    buddy_root = resolve_buddy_root(required=False)
    if buddy_root is not None:
        summary["buddy_root"] = str(buddy_root)
    llvm_root = resolve_llvm_root(required=False)
    if llvm_root is not None:
        summary["llvm_root"] = str(llvm_root)
    mlir_dir = resolve_mlir_dir(required=False)
    if mlir_dir is not None:
        summary["mlir_dir"] = str(mlir_dir)
    llvm_dir = resolve_llvm_dir(required=False)
    if llvm_dir is not None:
        summary["llvm_dir"] = str(llvm_dir)
    mlir_python_root = resolve_mlir_python_root(required=False)
    if mlir_python_root is not None:
        summary["mlir_python_root"] = str(mlir_python_root)
    optimizer = resolve_optimizer(required=False)
    if optimizer is not None:
        summary["optimizer"] = str(optimizer)
    for tool in ("buddy-opt", "buddy-translate", "mlir-opt", "mlir-translate", "llc", "clang", "ld.lld"):
        tool_path = resolve_tool(tool, required=False)
        if tool_path is not None:
            summary[tool.replace("-", "_")] = str(tool_path)
    return summary
