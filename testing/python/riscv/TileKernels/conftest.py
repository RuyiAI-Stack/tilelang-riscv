from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
TILEKERNELS_ROOT = REPO_ROOT / "3rdparty" / "TileKernels"

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

for path in (REPO_ROOT, TILEKERNELS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _maybe_set_llvm_root() -> Path | None:
    buddy_llvm = REPO_ROOT.parent / "buddy-mlir" / "llvm"
    vendored_llvm = REPO_ROOT / "3rdparty" / "llvm-project"
    candidates = [
        os.environ.get("TILELANG_RISCV_LLVM_ROOT"),
        os.environ.get("TILELANG_LLVM_INSTALL_DIR"),
        buddy_llvm / "build",
        buddy_llvm / "install",
        vendored_llvm / "install",
        vendored_llvm / "build-host" / "install",
        vendored_llvm / "build" / "install",
    ]
    path_tool = shutil.which("mlir-translate")
    if path_tool:
        candidates.append(Path(path_tool).resolve().parent.parent)

    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if (root / "bin" / "mlir-translate").is_file():
            os.environ.setdefault("TILELANG_RISCV_LLVM_ROOT", str(root))
            os.environ.setdefault("TILELANG_RISCV_MLIR_MODE", "ON")
            return root
    return None


if not TILEKERNELS_ROOT.exists():
    pytest.skip("TileKernels checkout is not available", allow_module_level=True)

if _maybe_set_llvm_root() is None:
    pytest.skip("LLVM/MLIR runtime toolchain not available", allow_module_level=True)

pytest.importorskip("tilelang.tladapter._native")
