from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
TILEOPS_ROOT = Path(os.environ.get("TILEOPS_ROOT", REPO_ROOT / "3rdparty" / "TileOPs"))
TILEOPS_SOURCE_ROOT = TILEOPS_ROOT / "src" if (TILEOPS_ROOT / "src" / "tileops").is_dir() else TILEOPS_ROOT

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
os.environ.setdefault("TILELANG_TARGET", "riscv")
os.environ.setdefault("TILELANG_RISCV_CLANG_OPT_LEVEL", "0")

for path in (REPO_ROOT, TILEOPS_SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _maybe_set_llvm_root() -> Path | None:
    candidates = [
        os.environ.get("TILELANG_RISCV_LLVM_ROOT"),
        "/root/llvm/build",
        "/home/liquanyi/tool/llvm",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "bin" / "mlir-translate").is_file():
            os.environ.setdefault("TILELANG_RISCV_LLVM_ROOT", str(root))
            os.environ.setdefault("TILELANG_RISCV_MLIR_MODE", "ON")
            return root
    return None


def _apply_tileops_test_overrides() -> None:
    from ._harness import (
        _ensure_minimal_tileops_kernel_modules,
        _ensure_tvm_tirx_shim,
        _ensure_torch_library_custom_op_shim,
    )

    _ensure_tvm_tirx_shim()
    _ensure_torch_library_custom_op_shim()
    # TileLang 0.1.12 moved the CUDA macro package. Reuse this checkout's
    # implementation for imports; this does not provide a different kernel.
    import importlib
    import types

    for name in ("tilelang.cuda", "tilelang.cuda.intrinsics", "tilelang.cuda.intrinsics.macro"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
    sys.modules.setdefault(
        "tilelang.cuda.intrinsics.macro.wgmma_macro_generator",
        importlib.import_module("tilelang.intrinsics.wgmma_macro_generator"),
    )
    # Install schedule metadata before any test imports upstream modules by
    # their direct paths. Otherwise their from-imports capture real CUDA probes
    # and a focused run behaves differently from collection of the full suite.
    _ensure_minimal_tileops_kernel_modules()


if not TILEOPS_ROOT.exists():
    pytest.skip("TileOPs checkout is not available", allow_module_level=True)

if _maybe_set_llvm_root() is None:
    pytest.skip("LLVM/MLIR runtime toolchain not available", allow_module_level=True)

pytest.importorskip("tilelang.tladapter._native")
_apply_tileops_test_overrides()
