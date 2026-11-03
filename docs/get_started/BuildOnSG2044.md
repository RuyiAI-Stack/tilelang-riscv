# Build and Run on SG2044

This document describes a reproducible native bring-up flow for the structured RISC-V backend on an SG2044 machine.
It is intentionally separate from the generic installation guide because this path uses a real RISC-V host, Buddy
MLIR, and a native GCC sysroot instead of the usual x86 or GPU development environment.

## Scope

The intended flow is:

- `TileLang -> MLIR Linalg -> MLIR vector / RVV lowering`
- native host shared library build with the RISC-V GCC toolchain
- execution on SG2044

This document is about native execution on SG2044.

## Buddy MLIR Version

The October 28 native operator tests used Buddy MLIR
[`release/v0.1.2`](https://github.com/buddy-compiler/buddy-mlir/releases/tag/release/v0.1.2).
The release tag resolves to `60f354a7b659a1ce36fdf9ba7bd97fdbf6bb0e5f` and pins
its LLVM submodule to `3845a20590afedfbece4d000f97a6cfbbea3bf43`. The separate compiler CI uses the Buddy `release/v0.1.0` x86_64 LLVM/MLIR
archive, which pins the same LLVM revision. Both builds require LLVM/MLIR 24.0;
this native guide does not change the CI toolchain.

Clone the release and verify that Git checked out its pinned LLVM revision:

```bash
git clone https://github.com/buddy-compiler/buddy-mlir.git
cd buddy-mlir
git checkout 60f354a7b659a1ce36fdf9ba7bd97fdbf6bb0e5f
git submodule update --init llvm
test "$(git -C llvm rev-parse HEAD)" = 3845a20590afedfbece4d000f97a6cfbbea3bf43
```

Create and activate a Python 3.12+ environment before configuring either build.
Install the compiler's Python build dependencies and the release's supported
RISC-V Torch version:

```bash
python3.13 -m venv ~/.venv-buddy
source ~/.venv-buddy/bin/activate
pip install --extra-index-url https://ruyirepo.ruyicommunity.cn/pypi/simple/ \
  torch==2.10.0 'numpy>=2,<2.5' 'nanobind>=2.9,<3' \
  scikit-build-core cython setuptools_scm cloudpickle pytest z3-solver ml-dtypes psutil
```

Set `CC` and `CXX` to the native GCC executables. If GCC uses a separate
`lib64/lp64d` layout, export its directories in both `LIBRARY_PATH` (link time)
and `LD_LIBRARY_PATH` (execution time) before running CMake. See the environment
example below. A source build of `z3-solver` can take several minutes on RISC-V.

Build LLVM/MLIR and Buddy MLIR:

```bash
cmake -G Ninja -S llvm/llvm -B llvm/build \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_ENABLE_RUNTIMES="openmp" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"
ninja -C llvm/build

cmake -G Ninja -S . -B build \
  -DMLIR_DIR="$PWD/llvm/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm/build/lib/cmake/llvm" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_PACKAGE_VERSION=0.1.2 \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"
ninja -C build buddy-opt buddy-translate
```

`LLVM_ENABLE_ZSTD=OFF` avoids a mismatch between system zstd headers and a
separate GCC sysroot. Compression support is optional for this pipeline.
The Buddy compiler tools above do not require its optional model dependencies;
install the release's `requirements.txt` when using its model importer.

Set `TILELANG_RISCV_LLVM_ROOT` to `buddy-mlir/llvm/build` for this build.
Record `git rev-parse HEAD`, `git -C llvm rev-parse HEAD`, and
`llvm/build/bin/llvm-config --version` with each SG2044 test report.

## Get the Sources

Use this RISC-V checkout on the TileLang 0.1.8 development line. When cloning
another copy, set `TILELANG_RISCV_REVISION` to the source revision being validated.
Then initialize its pinned dependencies:

```bash
git clone --recursive https://github.com/RuyiAI-Stack/tilelang-riscv.git tilelang-riscv
cd tilelang-riscv
git checkout "$TILELANG_RISCV_REVISION"
git submodule update --init --recursive
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
export TILELANG_RISCV_MLIR_MODE=ON
export TILELANG_RISCV_ABI=lp64d
export Z3_ROOT="$(python -c 'import pathlib, z3; print(pathlib.Path(z3.__file__).parent)')"
export CC=/path/to/gcc
export CXX=/path/to/g++
export PATH=/path/to/gcc-bin:$TILELANG_RISCV_LLVM_ROOT/bin:$Z3_ROOT/bin:$PATH
export LIBRARY_PATH=/path/to/gcc-lib64/lp64d:/path/to/gcc-libgcc/lp64d${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF"
```

Notes:

- Buddy MLIR `release/v0.1.2` requires Python 3.12 or newer for its Python packages.
  The release's supported RISC-V environment uses Torch 2.10.0 from the Ruyi package index.
- `Z3_ROOT` points to the installed `z3-solver` package, which contains `include` and `lib`
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

Use the vendored TVM FFI revision with this TVM pin; an unrelated newer FFI
installation can fail during TVM type registration. The wheel uses the Python
3.9 stable ABI, matching the project's minimum supported Python and Cython 3.3.

## Build Notes

The SG2044 path depends on a few implementation details wired into this tree:

- use `target="riscv"` as the backend name
- toolchain discovery covers sibling Buddy builds such as `../buddy-mlir/llvm/build`
- the RISC-V host wrapper passes `--gcc-toolchain` and `--sysroot` when it detects `/opt/gcc-native`
- RISC-V builds disable TVM's alternative linker selection because `ld.lld` failed on an SG2044 toolchain

If you need to rebuild after source changes, the same editable install command is sufficient:

```bash
cd tilelang-riscv
pip install -e . --no-build-isolation -v
```

## Validation

Run the native test set against the exact source and toolchain revisions recorded above:

```bash
cd tilelang-riscv
pytest testing/python/riscv/test_riscv_target_parse.py -q
pytest testing/python/riscv/test_riscv_toolchain.py -q
pytest testing/python/riscv/test_riscv_tladapter_pipeline.py -q
pytest testing/python/riscv/test_riscv_cache.py -q
pytest testing/python/riscv/codegen_ops -q
pytest testing/python/riscv/TileKernels -q
pytest testing/python/riscv/TileOps -q
```

Useful extended runtime checks:

```bash
pytest testing/python/riscv/test_riscv_jit_runtime.py -q
pytest testing/python/riscv/test_riscv_artifact_export.py -q
```

`test_riscv_jit_runtime.py` and `test_riscv_artifact_export.py` exercise broader
runtime and artifact-export paths. A test is considered SG2044-validated only
when its result is recorded from this native host with the source and Buddy/LLVM
revisions above; x86 results do not establish native RISC-V correctness.

## Troubleshooting

- If configuration fails with a Z3 lookup error, check that `Z3_ROOT` points to a prefix containing `include` and `lib`
- If the first CMake configure fails on `-lgcc_s`, verify that the toolchain's
  `lib64/lp64d` directory is present in `LIBRARY_PATH`
- If runtime shared library linking fails, verify that `CC`, `CXX`, and `/opt/gcc-native/sysroot` are all present
- If `tvm_ffi` import fails, reinstall the vendored package from `3rdparty/tvm/3rdparty/tvm-ffi`
- If first import becomes slow or tries to build extra Torch extensions, keep `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`
