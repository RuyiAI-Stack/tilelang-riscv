# RISC-V TileKernels tests

These tests compile original TileKernels programs through the RISC-V MLIR
adapter and compare selected executions against independent CPU/PyTorch
references. The submodule is pinned to `36d9e45d` with no local source changes.

## Coverage

The pinned source has **48 top-level JIT factories: nine mHC backward factories
and 39 others**. The inventory references all 48. The suite currently has
**48 numerical test cases**; factory and test counts measure different things,
and a test may execute several factories or reuse an existing forward helper.
They do not establish correctness for every shape, dtype or schedule.

Coverage includes routing and indexing, per-block/channel/token quantization,
fused SwiGLU, Engram, and mHC forward/backward components.
`test_mhc_backward.py` checks backward factories against autograd references,
including accumulation into existing gradient buffers. The Engram fused-routing
case checks preservation of shared values across serialized thread phases.

The October 16 x86 host run passed all 48 cases through the RISC-V adapter.
These results do not establish SG2044 native performance; the current compiler
changes still need native validation.

## Run

Initialize the pinned submodules and build the RISC-V adapter first. Set
`TILELANG_RISCV_LLVM_ROOT` to the LLVM/MLIR toolchain, then run:

```bash
TILELANG_DISABLE_CACHE=1 TILELANG_RISCV_MLIR_MODE=ON \
python -m pytest -q testing/python/riscv/TileKernels --junitxml=tilekernels.xml

python testing/python/riscv/operator_inventory.py \
  --check --junit tilekernels.xml --output tilekernels-inventory.json
```

See the [RISC-V test suite](../README.md) for compiler regressions and cache
verification, and the [TileOps suite](../TileOps/README.md) for its separate
operator contracts and coverage.
