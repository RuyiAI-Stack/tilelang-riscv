# Build and Run on SG2044

Build TileLang natively on SG2044 with buddy-mlir and LLVM/MLIR.

## Prerequisites

- Python 3.12+, native GCC/G++, CMake 3.26.1+ and Ninja.
- buddy-mlir and its matching LLVM/MLIR 24.0 development toolchain.

Download the matching `riscv64` buddy-mlir and LLVM/MLIR archives from the
[buddy-mlir releases](https://github.com/buddy-compiler/buddy-mlir/releases).
Choose packages compatible with your Python and system; `manylinux_2_39`
packages require glibc 2.39 or newer. To build the toolchain from source, follow
[buddy-mlir's build instructions](https://github.com/buddy-compiler/buddy-mlir#readme)
using the selected release and its pinned LLVM submodule.

## Environment

Replace `/path/to/...` with your LLVM, Buddy and native GCC installations:

```bash
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm
export TILELANG_RISCV_BUDDY_ROOT=/path/to/buddy-mlir
export TILELANG_RISCV_GCC_ROOT=/path/to/native-gcc
export CC="$TILELANG_RISCV_GCC_ROOT/bin/gcc"
export CXX="$TILELANG_RISCV_GCC_ROOT/bin/g++"

export PATH="$TILELANG_RISCV_BUDDY_ROOT/bin:$TILELANG_RISCV_LLVM_ROOT/bin:$TILELANG_RISCV_GCC_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TILELANG_RISCV_LLVM_ROOT/lib:$TILELANG_RISCV_BUDDY_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DTILELANG_RISCV_MLIR_MODE=ON"
export TILELANG_RISCV_ABI=lp64d
```

The optimizer defaults to `buddy-opt`. To use `mlir-opt` from the LLVM installation:

```bash
export TILELANG_RISCV_OPT=mlir-opt
```

For a GCC installation with separate runtime libraries, add their directories
before building:

```bash
export LIBRARY_PATH="/path/to/gcc-runtime-libs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/path/to/gcc-runtime-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`TILELANG_RISCV_MLIR_MODE` is a CMake build option: `ON` requires LLVM/MLIR,
`AUTO` enables it when available, and `OFF` disables the backend integration.

## Build

Run from the TileLang repository root:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --extra-index-url https://ruyirepo.ruyicommunity.cn/pypi/simple/ \
  torch scikit-build-core 'cython>=3.1' patchelf setuptools setuptools_scm \
  pytest 'z3-solver>=4.13,<4.15.5'

python -m pip install ./3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps
python -m pip install -e . --no-build-isolation -v
```

Install the vendored TVM FFI revision before TileLang. To rebuild after source
changes, rerun the final editable-install command.

## Verify

Check the selected optimizer and run a compile-and-execute test:

```bash
python -c 'from tilelang.tladapter import resolve_optimizer; print(resolve_optimizer())'
TILELANG_DISABLE_CACHE=1 python -m pytest -q \
  testing/python/riscv/test_riscv_jit_runtime.py::test_tilelang_compile_runs_riscv_host_adapter
```

Use `target="riscv"` when compiling kernels:

```python
kernel = tilelang.compile(func, target="riscv")
```

See [RISC-V tests](../../testing/python/riscv/README.md) for the full test suite.

For runnable examples, see [GEMM](../../examples/riscv/gemm/README.md) and
[BOSC GEMM](../../examples/riscv/bosc/README.md). GEMM uses the current host;
BOSC builds FPGA firmware and requires Buddy AME tools and an NR runtime with
UART v1 support.
