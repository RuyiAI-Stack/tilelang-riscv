# tilelang-riscv

TileLang compiler backend for RISC-V platforms.

This repository extends [TileLang](https://github.com/tile-ai/tilelang) with a
structured MLIR-backed RISC-V target. It is intended for native execution on
RISC-V systems such as SG2044, host-side kernel development, and BOSC AME
FPGA firmware generation through Buddy MLIR.

The RISC-V path does not require a CUDA or AMD GPU toolchain. It uses MLIR for
structured lowering and a native RISC-V C/C++ toolchain for host execution.

**Native host compilation pipeline:**

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
- BOSC AME INT8 GEMM firmware generation
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

## Build and Install

Follow [Build and Run on SG2044](./docs/get_started/BuildOnSG2044.md) for toolchain
requirements, environment configuration and installation.

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

## Examples

Run the examples from the repository root after configuring the toolchain:

| Example | Usage | Output |
| --- | --- | --- |
| [GEMM](./examples/riscv/gemm/README.md) | Compile and execute FP16 GEMM on the current host | Printed result |
| [BOSC GEMM](./examples/riscv/bosc/README.md) | Build INT8 GEMM firmware with Buddy AME tools and an NR runtime supporting UART v1 | FPGA ELF/bin |

`target="riscv"` selects the MLIR backend. The host adapter uses the current
host's compiler target; the BOSC example uses a separate firmware build.

## Run Tests

Run the core RISC-V backend tests:

```sh
python -m pytest -q testing/python/riscv/test_riscv_*.py
```

Pull requests that change the RISC-V backend also run the
[Buddy MLIR CI workflow](./.github/workflows/ci-buddy-mlir.yml), which builds
against the pinned LLVM/MLIR 24.0 toolchain from Buddy MLIR `release/v0.1.0`.
That compiler-only job explicitly selects `TILELANG_RISCV_OPT=mlir-opt`.

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
to `67250e81`; its suite has 451 passing checks (448 numerical cases and three
unsupported-feature rejection checks). TileKernels is pinned to `36d9e45d`, with
48 numerical checks and references to all 48 top-level JIT factories, including
nine mHC backward factories. Test counts do not certify all operator variants.

These October 28–29 results were obtained natively on SG2044 through the RISC-V
MLIR adapter, using Buddy `release/v0.1.2` with the same LLVM revision as the
separate Buddy `release/v0.1.0` compiler CI. Selected paths now include aligned
FP8 quantization, INT8 row/block quantization, SmoothQuant, per-tensor INT8
dequantization and FP16/BF16 sigmoid. Public backend registration remains scoped
to tests. See the [TileOps coverage and limitations](./testing/python/riscv/TileOps/README.md),
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
- [GEMM example](./examples/riscv/gemm/README.md)
- [BOSC GEMM example](./examples/riscv/bosc/README.md)

## Contributing

Changes to the RISC-V backend should include a focused test or a documented
reason why runtime validation is not applicable. Please keep third-party
submodules unchanged; backend adaptations belong in this repository.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the general contribution
guidelines.

## License

This project is released under the [MIT License](./LICENSE).
