# RISC-V Tests

This directory contains TileLang RISC-V backend tests.

## Test Groups

- `test_riscv_*.py`: core backend tests for target parsing, toolchain lookup,
  TLAdapter pass execution, artifact export, cache reload, and JIT runtime
  correctness.
- `codegen_ops/`: synthetic RISC-V MLIR/codegen tests.  See
  `codegen_ops/README.md` for the per-file breakdown and dump options.
- `TileKernels/`: runtime correctness tests covering the 48 JIT factories in
  the pinned TileKernels revision (9 mHC backward factories and 39 others).
  Factory counts and pytest case counts are separate measurements.
  Each test compiles with `target="riscv"`, executes through the RISC-V host
  adapter, and compares against a CPU/PyTorch reference.
- `TileOps/`: runtime correctness tests for selected TileOps kernels.  See
  [TileOps/README.md](TileOps/README.md) for the current coverage.

## Run

Run from the repository root with a TileLang build configured using
`-DTILELANG_RISCV_MLIR_MODE=ON`. Replace the toolchain paths below with your
installations:

```bash
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm
export TILELANG_RISCV_BUDDY_ROOT=/path/to/buddy-mlir
export PATH="$TILELANG_RISCV_BUDDY_ROOT/bin:$TILELANG_RISCV_LLVM_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TILELANG_RISCV_LLVM_ROOT/lib:$TILELANG_RISCV_BUDDY_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

The optimizer defaults to `buddy-opt`. Set `TILELANG_RISCV_OPT=mlir-opt` to use
the LLVM optimizer. See the [SG2044 build guide](../../../docs/get_started/BuildOnSG2044.md)
for installation and native GCC configuration.

Core backend tests:

```bash
python -m pytest -q testing/python/riscv/test_riscv_*.py
```

Synthetic codegen tests:

```bash
python -m pytest -q testing/python/riscv/codegen_ops
```

TileKernels runtime tests:

```bash
python -m pytest -q testing/python/riscv/TileKernels
```

TileOps runtime tests:

```bash
python -m pytest -q testing/python/riscv/TileOps
```

## Operator inventory and acceptance

Generate an inventory from the pinned libraries, optionally attaching test-file
results from a local or SG2044 JUnit report:

```bash
python testing/python/riscv/operator_inventory.py \
  --junit results.xml --output operator-inventory.json --check
```

The inventory includes every TileOps manifest entry and TileKernels JIT factory.
It records upstream kernel/component references and scoped public-API tests,
and keeps upstream manifest status separate from local execution evidence.
Test references and passing test files
do **not** certify every public Op, shape, dtype, or optimized kernel variant.
`--check` validates the manifest schema and stale file/function mappings;
unmapped entries remain explicit. `--require-tileops-numerical` requires observed
numerical evidence for every TileOps entry and rejects known or observed gaps.
It currently fails as expected. Rejection, compilation-only and expected-failure
results are reported separately from numerical passes.
See [TileOps coverage and limitations](TileOps/README.md) and
[TileKernels coverage](TileKernels/README.md).

When validating compiler changes, set `TILELANG_DISABLE_CACHE=1` to ensure the
tests compile fresh artifacts. The SG2044 environment also needs its native
LLVM/MLIR and GCC library paths; see the
[SG2044 build guide](../../../docs/get_started/BuildOnSG2044.md).

## Engram Hash IR and PyTorch comparison

From the repository root, with the RISC-V test toolchain configured as above,
run the existing Engram Hash correctness test with artifact export enabled:

```bash
TILELANG_DISABLE_CACHE=1 TILELANG_ENGRAM_HASH_ARTIFACTS=artifacts/engram-hash \
  python3 -m pytest -qs testing/python/riscv/TileKernels/test_engram_hash_kernel.py
```

The test prints a new run directory containing `input.tir.py`, TIR pass dumps
under `tir-passes/`, `kernel.mlir`, and `comparison.json`. The JSON records
the execution architecture, seed, pass/fail verdict and both output tensors.
A numeric mismatch writes the comparison and fails pytest.

## Validation scope

The dedicated [Buddy MLIR CI workflow](../../../.github/workflows/ci-buddy-mlir.yml)
uses the official LLVM/MLIR 24.0 toolchain asset from Buddy MLIR
`release/v0.1.0`, builds TileLang against it, and runs the core lowering and
codegen suites on an x86_64 runner. It explicitly selects
`TILELANG_RISCV_OPT=mlir-opt` and validates compiler integration, not native RVV
execution on SG2044.
See the [SG2044 build guide](../../../docs/get_started/BuildOnSG2044.md)
for toolchain setup and installation.

The October 28–29 SG2044 native run at TileOps `67250e81` passed 451 TileOps
checks (436 kernel/component numerical, 12 scoped public-API numerical and three
rejection checks) and 48 TileKernels numerical checks. The 499 checks include
496 numerical executions and three rejection checks, with no compile-only cases,
skips or expected failures. The toolchain was Buddy `release/v0.1.2` with LLVM
`3845a20590afedfbece4d000f97a6cfbbea3bf43`. These results cover the declared
test configurations; they do not certify every operator contract or performance.

The related compiler regressions cover thread-local state across serialized
phases, safe memory tails, IEEE arithmetic, bit operations, FP8/FP4 conversion,
dynamic while loops, affine buffer extents, shared-memory allocation policy,
streaming-load bit preservation and the FP32 `__fdividef` signature and exceptional
values. The operator inventory has 180 TileOps kernel-test references, nine
component-test references and 16 unmapped declarations; all 48 TileKernels
factories have references. References alone are not numerical acceptance.
To run the tracked backend and operator tests with fresh compilation, then check
cache reload separately:

```bash
TILELANG_DISABLE_CACHE=1 python -m pytest -q \
  testing/python/riscv/test_riscv_*.py testing/python/riscv/codegen_ops \
  testing/python/riscv/TileKernels testing/python/riscv/TileOps \
  testing/python/riscv/test_operator_inventory.py \
  testing/python/transform/test_tilelang_transform_shared_memory_reuse_config.py \
  --deselect=testing/python/riscv/test_riscv_cache.py::test_riscv_disk_cache_reloads_without_recompile

TILELANG_DISABLE_CACHE=0 python -m pytest -q testing/python/riscv/test_riscv_cache.py
```

For manual example usage, see [GEMM](../../../examples/riscv/gemm/README.md)
and [BOSC GEMM](../../../examples/riscv/bosc/README.md). FPGA execution is a
separate validation step from the host and codegen tests above.
