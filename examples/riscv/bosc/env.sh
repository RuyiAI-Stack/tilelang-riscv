#!/usr/bin/env bash
# Source after activating a TileLang environment with MLIR support.
_bosc_example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_bosc_repo_root="$(cd -- "${_bosc_example_dir}/../../.." && pwd)"
export BOSC_BUDDY_SOURCE="${BOSC_BUDDY_SOURCE:-$(dirname -- "${_bosc_repo_root}")/buddy-mlir}"
export TILELANG_RISCV_LLVM_ROOT="${TILELANG_RISCV_LLVM_ROOT:-${BOSC_BUDDY_SOURCE}/llvm/build}"
if [[ -z "${TILELANG_RISCV_BUDDY_ROOT:-}" ]]; then
  if [[ -x "${BOSC_BUDDY_SOURCE}/build-24/bin/buddy-opt" ]]; then
    export TILELANG_RISCV_BUDDY_ROOT="${BOSC_BUDDY_SOURCE}/build-24"
  else
    export TILELANG_RISCV_BUDDY_ROOT="${BOSC_BUDDY_SOURCE}/build"
  fi
fi
export TILELANG_RISCV_OPT="${TILELANG_RISCV_BUDDY_ROOT}/bin/buddy-opt"
export PATH="${TILELANG_RISCV_BUDDY_ROOT}/bin:${TILELANG_RISCV_LLVM_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${TILELANG_RISCV_LLVM_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export TILELANG_DISABLE_CACHE=1
unset _bosc_example_dir _bosc_repo_root
