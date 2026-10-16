# tilelang-riscv

TileLang compiler backend for RISC-V platforms.

This repository extends [TileLang](https://github.com/tile-ai/tilelang) with a
structured MLIR-backed RISC-V target. It is intended for native execution on
RISC-V systems such as SG2044 and for developing portable TileLang kernels
against the LLVM/MLIR toolchain.

The RISC-V path does not require a CUDA or AMD GPU toolchain. It uses MLIR for
structured lowering and a native RISC-V C/C++ toolchain for host execution.

**Compilation pipeline:**

```text
TileLang Python DSL
  └─► TVM TIR
        └─► MLIR Linalg / SCF / MemRef / Vector
              └─► LLVM MLIR
                    └─► LLVM IR
                          └─► native RISC-V shared library
                                └─► execution on SG2044
```

## Scope

The main focus of this repository is:

- a `target="riscv"` backend for TileLang
- TIR to structured MLIR lowering
- native RISC-V host-library generation and execution
- compatibility with TileLang operator implementations, TileOps and TileKernels
- correctness validation on real RISC-V hardware

The backend is experimental and is being developed incrementally. GPU
backends remain available through the upstream TileLang codebase.

## Clone

Clone the repository together with its required submodules:

```sh
git clone --recursive https://github.com/RuyiAI-Stack/tilelang-riscv.git
cd tilelang-riscv
```

If the repository was cloned without submodules:

```sh
git submodule update --init --recursive
```

## Prerequisites

The native RISC-V workflow requires:

- Python 3.9 or newer
- CMake 3.26 or newer
- an LLVM/MLIR installation, such as the toolchain built by Buddy MLIR
- a native RISC-V GCC toolchain and sysroot
- a RISC-V machine for native execution, such as SG2044

The compiler can be built on another host for inspection or artifact
generation, but RVV runtime validation must be performed on a RISC-V machine.

## Environment

Set the toolchain variables for a native RISC-V build. The paths below are
examples and should be adapted to the local installation:

```sh
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm-mlir
export CC=/path/to/riscv64-gcc
export CXX=/path/to/riscv64-g++
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF -DTILELANG_RISCV_MLIR_MODE=ON"
```

`TILELANG_RISCV_MLIR_MODE` controls the optional LLVM/MLIR integration:

- `ON` requires LLVM/MLIR and fails during configuration if it is missing
- `AUTO` enables it when a compatible installation is found
- `OFF` disables the RISC-V MLIR backend

For the complete SG2044 environment and toolchain setup, see
[Build and Run on SG2044](./docs/get_started/BuildOnSG2044.md).

## Install and Build

Install the vendored TVM FFI package first, then install TileLang in editable
mode:

```sh
python -m venv .venv
source .venv/bin/activate

pip install scikit-build-core cython patchelf setuptools_scm cloudpickle pytest
pip install ./3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps
pip install -e . --no-build-isolation -v
```

After changing C++ or Python sources, rebuild the editable installation:

```sh
pip install -e . --no-build-isolation -v
```

## Verify the Build

Check that the Python package and native extension can be imported:

```sh
python -c "import tilelang; print(tilelang.__file__)"
python -c "import tilelang.tladapter._native; print('native adapter: ok')"
```

The RISC-V target is selected explicitly in user code:

```python
kernel = tilelang.compile(tir_func, target="riscv")
```

For target syntax and backend limitations, see
[Understanding Targets](./docs/get_started/targets.md).

## Run Tests

Run the core RISC-V backend tests:

```sh
python -m pytest -q testing/python/riscv/test_riscv_*.py
```

Run the synthetic MLIR/code-generation tests:

```sh
python -m pytest -q testing/python/riscv/codegen_ops
```

Run the TileKernels runtime correctness tests:

```sh
python -m pytest -q testing/python/riscv/TileKernels
```

Run the TileOps runtime correctness tests:

```sh
TILELANG_DISABLE_CACHE=1 python -m pytest -q testing/python/riscv/TileOps
```

Both suites compile original upstream programs with `target="riscv"` and compare
selected executions against independent CPU/PyTorch references. TileOps is pinned
to `67250e81`; its suite has 415 passing checks (412 numerical cases and three
unsupported-feature rejection checks). TileKernels is pinned to `36d9e45d`, with
48 numerical checks and references to all 48 top-level JIT factories, including
nine mHC backward factories. Test counts do not certify all operator variants.

These October 16 results were obtained on an x86 host through the RISC-V MLIR
adapter. The current TileOps pin and compiler changes still need SG2044 native
validation. See the [TileOps coverage and limitations](./testing/python/riscv/TileOps/README.md),
[TileKernels tests](./testing/python/riscv/TileKernels/README.md) and
[RISC-V test suite](./testing/python/riscv/README.md) for reproduction instructions.

## Development

The main implementation areas are:

- `src/target/codegen_riscv.cc`: TIR to MLIR lowering
- `tilelang/jit/adapter/riscv/`: host compilation, ABI handling, and runtime
  execution
- `cmake/TileLangRISCVMLIR.cmake`: LLVM/MLIR discovery and build configuration
- `testing/python/riscv/`: backend and runtime validation

When adding support for a new TileLang operator, prefer this workflow:

1. keep the original operator implementation unchanged
2. adapt the RISC-V lowering or runtime path in this repository
3. add a focused runtime correctness test
4. validate it on SG2044

## Documentation

- [Understanding Targets](./docs/get_started/targets.md)
- [Build and Run on SG2044](./docs/get_started/BuildOnSG2044.md)
- [RISC-V test suite layout](./testing/python/riscv/README.md)

## Contributing

Changes to the RISC-V backend should include a focused test or a documented
reason why runtime validation is not applicable. Please keep third-party
submodules unchanged; backend adaptations belong in this repository.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the general contribution
guidelines.

## License

This project is released under the [MIT License](./LICENSE).
