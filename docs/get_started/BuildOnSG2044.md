# Build and Run on SG2044

This document describes the validated native bring-up flow for the structured RISC-V backend on an SG2044 machine.
It is intentionally separate from the generic installation guide because this path uses a real RISC-V host, a Buddy
/ LLVM toolchain, and a native GCC sysroot instead of the usual x86 or GPU development environment.

## Scope

The validated flow is:

- `TileLang -> MLIR Linalg -> MLIR vector / RVV lowering`
- native host shared library build with the RISC-V GCC toolchain
- execution on SG2044

This document is about native execution on SG2044.

## Buddy MLIR Version

The RISC-V backend is validated against the current stable Buddy MLIR release
as of October 16, 2026,
`release/v0.1.0` (commit `2860a0e135966387e22c48e61105c99a824ef7fe`). It uses
LLVM/MLIR 24.0.0git from the `llvm` submodule at
`3845a20590afedfbece4d000f97a6cfbbea3bf43`. The backend links to LLVM/MLIR
directly and does not use Buddy-specific libraries. CMake requires LLVM/MLIR
24.0 to catch unsupported toolchains during configuration.

Clone the pinned Buddy release and its LLVM source before building the
toolchain:

```bash
git clone --depth 1 --branch release/v0.1.0 \
  https://github.com/buddy-compiler/buddy-mlir.git
cd buddy-mlir
git submodule update --init llvm
test "$(git rev-parse HEAD)" = "2860a0e135966387e22c48e61105c99a824ef7fe"
test "$(git -C llvm rev-parse HEAD)" = "3845a20590afedfbece4d000f97a6cfbbea3bf43"
```

Use a checkout of the matching LLVM submodule for the build steps in Buddy's
[RVV environment guide](https://github.com/buddy-compiler/buddy-mlir/blob/release/v0.1.0/docs/RVVEnvironment.md).
The generic installation steps below apply to this pinned release.

For x86_64 compiler and codegen validation, the release also provides the
`llvm-24.0.0git-cp312-abi3-manylinux_2_28_x86_64.tar.gz` asset. Extract it and
set `TILELANG_RISCV_LLVM_ROOT` to the extracted directory. Use the matching
RISC-V asset or a source build on SG2044 when native execution is required.

## Get the Sources

Clone the repository with its bundled dependencies, including the `TileKernels`
submodule used by the RISC-V operator-library tests:

```bash
git clone --recursive https://github.com/RuyiAI-Stack/tilelang-riscv.git tilelang-riscv
cd tilelang-riscv
```

If you already cloned the repository without submodules, run:

```bash
git submodule update --init --recursive
```

## Environment Setup

Activate the Python environment and export the toolchain variables:

```bash
source ~/.venv-buddy/bin/activate

export TILELANG_RISCV_LLVM_ROOT=/path/to/buddy-mlir/llvm/build
export Z3_ROOT=/path/to/.local/z3
export CC=/path/to/gcc
export CXX=/path/to/g++
export PATH=/path/to/gcc-bin:$TILELANG_RISCV_LLVM_ROOT/bin:$Z3_ROOT/bin:$PATH
export LIBRARY_PATH=/path/to/gcc-lib64/lp64d:/path/to/gcc-libgcc/lp64d${LIBRARY_PATH:+:$LIBRARY_PATH}
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DTILELANG_RISCV_MLIR_MODE=ON"
```

Notes:

- `Z3_ROOT` is required when `z3-solver` wheels are unavailable on `riscv64`
- if the native toolchain uses a split `lib64/lp64d` layout, include it in
  `LIBRARY_PATH` or CMake's compiler sanity check can fail on `-lgcc_s`
- `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` avoids first-import JIT building of the optional Torch DLPack extension
- `torch-c-dlpack-ext` is skipped on `riscv64`; the validated SG2044 flow uses
  TVM FFI without that optional prebuilt extension

## Install Dependencies

Install the Python-side build dependencies first:

```bash
pip install scikit-build-core cython patchelf setuptools_scm cloudpickle pytest
```

Install the vendored TVM FFI package from the source tree:

```bash
cd tilelang-riscv
pip install ./3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps
```

Then install TileLang itself in editable mode:

```bash
cd tilelang-riscv
pip install -e . --no-build-isolation -v
```

## Build Notes

The SG2044 path depends on a few implementation details that are already wired into this tree:

- use `target="riscv"` as the backend name
- toolchain discovery covers sibling Buddy builds such as `../buddy-mlir/llvm/build`
- the RISC-V host wrapper passes `--gcc-toolchain` and `--sysroot` when it detects `/opt/gcc-native`
- RISC-V builds disable TVM's alternative linker selection because `ld.lld` failed on the validated SG2044 toolchain

If you need to rebuild after source changes, the same editable install command is sufficient:

```bash
cd tilelang-riscv
pip install -e . --no-build-isolation -v
```

## Validation

The validated native test set is:

```bash
cd tilelang-riscv
pytest testing/python/riscv/test_riscv_target_parse.py -q
pytest testing/python/riscv/test_riscv_toolchain.py -q
pytest testing/python/riscv/test_riscv_tladapter_pipeline.py -q
pytest testing/python/riscv/test_riscv_cache.py -q
pytest testing/python/riscv/codegen_ops -q
pytest testing/python/riscv/TileKernels -q
```

Useful extended runtime checks:

```bash
pytest testing/python/riscv/test_riscv_jit_runtime.py -q
pytest testing/python/riscv/test_riscv_artifact_export.py -q
```

`test_riscv_jit_runtime.py` and `test_riscv_artifact_export.py` exercise broader
runtime and artifact-export paths and may expose additional toolchain or runtime
issues beyond the smoke set above.

## Troubleshooting

- If configuration fails with a Z3 lookup error, check that `Z3_ROOT` points to a prefix containing `include` and `lib`
- If the first CMake configure fails on `-lgcc_s`, verify that the toolchain's
  `lib64/lp64d` directory is present in `LIBRARY_PATH`
- If runtime shared library linking fails, verify that `CC`, `CXX`, and `/opt/gcc-native/sysroot` are all present
- If `tvm_ffi` import fails, reinstall the vendored package from `3rdparty/tvm/3rdparty/tvm-ffi`
- If first import becomes slow or tries to build extra Torch extensions, keep `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`
