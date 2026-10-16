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

## Validation scope

The October 16 x86 host runs at TileOps `67250e81` passed 415 TileOps checks
(400 kernel/component numerical, 12 scoped public-API numerical and three
rejection checks) and 48 TileKernels numerical checks. There are no compile-only
cases in either operator-library suite. Current sources have not been rerun on
SG2044; host execution does not establish native RISC-V correctness or performance.

The related compiler regressions cover thread-local state across serialized
phases, safe memory tails, IEEE arithmetic, bit operations, FP8/FP4 conversion,
dynamic while loops, affine buffer extents and shared-memory allocation policy.
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

Set `TILELANG_RISCV_LLVM_ROOT` to the LLVM/MLIR toolchain and
`TILELANG_RISCV_MLIR_MODE=ON` for these runs.
