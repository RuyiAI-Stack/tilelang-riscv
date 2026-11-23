# RISC-V Troubleshooting and Known Limitations

This guide documents environment issues, root causes, toolchain integration behavior, and current architectural limitations observed when building and running `RuyiAI-Stack/tilelang-riscv` (`target="riscv"`) on native `riscv64` hardware.

## 1. Validated Native Baseline and Scope

Full end-to-end runtime validation was completed on native `riscv64` hardware with the following environment:

- **Host**: `openeuler-riscv64` (64-core RV64GCV with `v`/`zfh`/`zvfh`; **no `zfbfmin`/`zvfbfmin`**, so `bf16` conversions lower to software libcalls)
- **OS / Toolchain**: openEuler 24.03, `glibc` 2.38, GCC 12.2, Python 3.11.6
- **TileLang baseline**: `RuyiAI-Stack/tilelang-riscv` @ `32fce5ed`
- **LLVM / MLIR**: self-built `24.0.0git` (`RuyiAI-Stack/llvm-project` @ `3845a205`) with `compiler-rt` builtins (`libclang_rt.builtins.a`) built incrementally

When `compiler-rt` builtins and GCC library paths are properly supplied during host shared-library linking, the native test suite passes **823 / 823** cases (0 failures, 0 skipped, 1896.05 s):

- `testing/python/riscv/test_riscv_*.py`: 108 passed
- `testing/python/riscv/codegen_ops`: 252 passed
- `testing/python/riscv/TileKernels`: 48 passed (48 JIT factories)
- `testing/python/riscv/TileOps`: 415 passed (with a test-harness workaround for 6 FP8 `torch.full` cases; see Section 4)
- `python3 testing/python/riscv/operator_inventory.py --junit ... --output ... --check`: exited 0 across 205 TileOps manifest declarations and 48 TileKernels JIT factories

> [!NOTE]
> `.github/workflows/ci-buddy-mlir.yml` runs on `ubuntu-latest` (`x86_64`) and executes the 5 compile/codegen suites (`test_riscv_target_parse.py`, `test_riscv_toolchain.py`, `test_riscv_tladapter_pipeline.py`, `test_riscv_artifact_export.py`, `codegen_ops`). It does not run `TileKernels`, `TileOps`, `test_riscv_jit_runtime.py`, or `test_riscv_cache.py` on native `riscv64` hardware. Also note the existing doc discrepancy between `docs/get_started/BuildOnSG2044.md` (which describes a validated SG2044 flow) and `README.md` (which notes that October 16 results came from an `x86` host and still require SG2044 native validation).

---

## 2. Low-Precision (`fp16` / `bf16`) Link Failures and Silent Numerical Corruption

### Symptoms

- **Runtime symbol lookup failure**:
  ```text
  OSError: .../kernel.so: undefined symbol: __truncsfhf2
  ```
  (or `__extendhfsf2`, `__truncsfbf2`, `__extendbfsf2`) raised when `HostKernelLibrary.__init__` loads the compiled shared library via `ctypes.CDLL` (`tilelang/jit/adapter/riscv/wrapper.py`).
- **Silent `bf16` numerical corruption**: tensor outputs contain `nan` or `1e23`/`1e38`-scale garbage across elements when low-precision libcalls are missing or mismatched.

### Root Cause

1. **Codegen emits standard MLIR float casts**: `CastValueToType` in `src/target/codegen_riscv.cc` (`:6204-6218`, `:6288-6301`) lowers floating-point width conversions directly to `arith.extf` / `arith.truncf`.
2. **MLIR lowering relies on LLVM RTLIB libcalls**: `tilelang/jit/adapter/riscv/libgen.py` (`_POST_LOW_PRECISION_LEGALIZATION_PASSES`, `:306-318`) passes `arith.extf` / `arith.truncf` for `f16`/`bf16` through `convert-arith-to-llvm`, and `_legalize_low_precision_conversions_for_riscv` (`:98`, `:115`) also emits `arith.truncf` / `arith.extf` when converting between FP8 and `f16`/`bf16`. On `riscv64` targets without hardware `zfbfmin` (and for libcall paths), LLVM lowers these to `__truncsfhf2`, `__extendhfsf2`, `__truncsfbf2`, and `__extendbfsf2`.
3. **GCC `libgcc` lacks these builtins on `riscv64`**: System GCC 12 `libgcc.a` / `libgcc_s.so` on openEuler 24.03 does not export `__truncsfbf2` / `__truncsfhf2` (`nm libgcc.a | grep truncsfbf2` is empty). On `x86_64`, default toolchain runtimes supply these symbols, hiding the gap in x86 CI.
4. **Deferred error under `-shared`**: `build_host_shared_library` previously linked with `clang -shared` without `-Wl,--no-undefined` and without probing `compiler-rt`, allowing broken `.so` binaries to link cleanly, fail later at `ctypes.CDLL`, and get persisted to disk cache.

### Resolution and Manual Override

`build_host_shared_library` in `tilelang/jit/adapter/riscv/wrapper.py` now:

- Probes `clang -print-resource-dir` for a target-matching `libclang_rt.builtins*.a` archive (checking `<resource>/lib/<triple>/libclang_rt.builtins.a`, vendor-normalized `<arch>-unknown-...`, `<resource>/lib/linux/libclang_rt.builtins-<arch>.a`, and `<arch>-*/libclang_rt.builtins.a`) and links it via `-L<dir> -l<lib>` unless `TILELANG_RISCV_CLANG_FLAGS` / `clang_flags` already specifies `-rtlib=` or `clang_rt.builtins`.
- Passes `-Wl,--no-undefined` on ELF/GNU-style targets so missing runtime symbols fail immediately at link time rather than being cached and crashing in `ctypes.CDLL`.

If your self-built LLVM/MLIR toolchain was built with an empty `LLVM_ENABLE_RUNTIMES`, build `compiler-rt` builtins incrementally inside the LLVM build directory:

```bash
cmake -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
      -DCOMPILER_RT_BUILD_BUILTINS=ON \
      -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
      -DCOMPILER_RT_BUILD_XRAY=OFF \
      -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
      -DCOMPILER_RT_BUILD_PROFILE=OFF .
ninja -j 32 builtins
```

Verify where the archive landed (do not guess `lib/linux/` vs triple-named directories):

```bash
find "$TILELANG_RISCV_LLVM_ROOT" -name 'libclang_rt.builtins*.a'
```

To override manually, pass the exact directory and library via `TILELANG_RISCV_CLANG_FLAGS`:

```bash
export TILELANG_RISCV_CLANG_FLAGS="... -L/path/to/lib/clang/24/lib/riscv64-unknown-linux-gnu -lclang_rt.builtins"
```

---

## 3. Kernel Cache Stale Hits After Toolchain Changes

### Symptom

After fixing `TILELANG_RISCV_CLANG_FLAGS`, `TILELANG_RISCV_GCC_ROOT`, `TILELANG_RISCV_LLVM_ROOT`, or building `compiler-rt` builtins, rerunning pytest keeps failing with the exact same `OSError` or bad numeric output unless `TILELANG_DISABLE_CACHE=1` is set.

### Root Cause and Fix

Previously, `KernelCache._generate_key` (`tilelang/cache/kernel_cache.py:217-262`) only hashed TIR source, `args_repr`, `target`, `pass_configs`, `compile_flags`, and `_get_base_key()` (version + `platform.machine()` + `libtilelang.so` hash, cached via `@functools.cache` at `:104-115`). Any previously cached broken `.so` continued matching the same cache key after environment fixes.

`KernelCache._generate_key` now embeds an uncached `riscv_toolchain_fingerprint()` entry for `riscv` targets capturing:

- RISC-V toolchain environment variables (`TILELANG_RISCV_LLVM_ROOT`, `TILELANG_RISCV_GCC_ROOT`, `GCC_ROOT`, `CC`, `CXX`, `TILELANG_RISCV_CLANG_FLAGS`, `TILELANG_RISCV_ABI`, `TILELANG_RISCV_MARCH`, `TILELANG_RISCV_CPU`, `TILELANG_HOST_TRIPLE`, `TILELANG_RISCV_CLANG_OPT_LEVEL`)
- Resolved GCC root path, host triple, `clang` binary identity (`path`, `size`, `mtime_ns`), and discovered `libclang_rt.builtins*.a` identity (`path`, `size`, `mtime_ns`)

If you modify C++ passes or rebuild LLVM libraries without changing `libtilelang.so` / `clang` timestamps, or want clean compiler timing runs, you can still force fresh compilation with:

```bash
export TILELANG_DISABLE_CACHE=1
```

---

## 4. Native `riscv64` Distro and Build Troubleshooting

### Non-SG2044 GCC Layout & `crtbeginS.o` / `-lgcc_s` Not Found

- **Cause**: `_resolve_riscv_gcc_root()` (`tilelang/jit/adapter/riscv/wrapper.py:661-682`) falls back to `/opt/gcc-native` (SG2044 SDK layout) when `TILELANG_RISCV_GCC_ROOT`, `GCC_ROOT`, `CC`, and `CXX` are unset. On distros such as openEuler 24.03, passing `--gcc-toolchain=/usr` to a self-built Clang can also fail to resolve `crtbeginS.o` and `-lgcc_s`.
- **Workaround**: Locate the active GCC directory and pass `--gcc-install-dir`:

```bash
dirname "$(gcc -print-file-name=crtbeginS.o)"
# e.g. /usr/lib/gcc/riscv64-openEuler-linux/12
export TILELANG_RISCV_CLANG_FLAGS="--gcc-install-dir=/usr/lib/gcc/riscv64-openEuler-linux/12"
```

### Python & Build Dependency Constraints

- **Cython 3.3 vs `cp38` Limited API**: `pyproject.toml:66,76` specifies `cython>=3.1.0` without an upper bound alongside `wheel.py-api = "cp38"`. Cython 3.3 errors with `#error "Cython 3.3 requires the Python Limited API version to be 3.9 or greater."`. Pin `cython==3.1.8`.
- **NumPy 2.x vs PyTorch 2.4.1 ABI**: Prebuilt/community `riscv64` PyTorch 2.4.1 wheels (built against NumPy 1.x) fail with `_ARRAY_API not found` if `pip install -e .` upgrades NumPy to 2.x. Re-pin `numpy<2` (e.g. `1.26.4`) after installing `tilelang`.
- **`requirements.txt` vs `pyproject.toml`**: `pyproject.toml:38` skips `torch-c-dlpack-ext` on `riscv64`, whereas `requirements.txt:4` does not have `platform_machine != 'riscv64'`. Install via `pyproject.toml` (`pip install -e . --no-build-isolation`) rather than `pip install -r requirements.txt`, and keep `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`.
- **`torch` / `z3-solver` / LLVM 24 availability**: PyPI does not ship official `riscv64` wheels for `torch` or `z3-solver`; `z3-solver` must be compiled from source (or provided via `Z3_ROOT`), and LLVM/MLIR 24 may need to be built from source if `glibc < 2.39` prevents using prebuilt tarballs.
- **LLVM 24 `R_RISCV_JAL` relocation overflow**: Building `TosaProfileCompliance.cpp` at `-O3` on `riscv64` can exceed the +/-1 MB `JAL` branch range (`relocation truncated to fit: R_RISCV_JAL`). Add `set_source_files_properties(TosaProfileCompliance.cpp PROPERTIES COMPILE_OPTIONS "-Os")` in `mlir/lib/Dialect/Tosa/Transforms/CMakeLists.txt`.
- **Stale CMake cache -> Placeholder MLIR backend**: If `TILELANG_RISCV_LLVM_ROOT` was missing on the first configure, `build/CMakeCache.txt` caches an empty path and compiles the placeholder backend, causing tests to skip silently. Clean before reconfiguring:

```bash
rm -rf build/CMakeCache.txt build/CMakeFiles build/tilelang_cython_wrapper.cpp
```

### Test Execution Caveats

- **Explicit `TILELANG_RISCV_LLVM_ROOT`**: `testing/python/riscv/TileKernels/conftest.py:22-24` and `testing/python/riscv/TileOps/conftest.py:26-27` fall back to author-specific directories (`/root/llvm/build`, `/home/liquanyi/tool/llvm`) and silently skip the entire suite if not found. Always export `TILELANG_RISCV_LLVM_ROOT` and `TILELANG_RISCV_MLIR_MODE=ON`.
- **Run pytest from the repository root**: `testing/python/riscv/operator_inventory.py` matches `<testcase classname="...TileOps.test_...">` entries in JUnit XML, which pytest only generates when invoked from the repository root.
- **PyTorch CPU FP8 `fill_cpu` limitation**: PyTorch 2.4.1 CPU does not implement `fill_cpu` for `Float8_e4m3fn` (`RuntimeError: "fill_cpu" not implemented for 'Float8_e4m3fn'`), affecting 6 TileOps tests during test-input construction (`torch.full(..., dtype=fp8)` -> create in `float32` and `.to(fp8)`).

---

## 5. Known Limitations and Follow-up Roadmap

### 5.1 Scalarized Lowering & Downstream BOSCAME/AME Incompatibility (Not Changed in This PR)

**Location**: `tilelang/engine/phase.py:138-152` (`LowerAndLegalize` RISC-V branch)

The RISC-V branch currently runs an 11-pass pipeline (`BindTarget`, `AddWrapperForSingleBufStore`, `LegalizeNegativeIndex`, optional `VerifyParallelLoop`, `InjectAssumes`, `Simplify`, `LowerAccessPtr`, `LegalizeSafeMemoryAccess`, `Simplify`, `HoistNonRestrictParams`) and intentionally skips GPU/structured passes such as `LowerTileOp`, `LayoutInference`, `PipelinePlanning`, `InjectSoftwarePipeline`, `DecoupleTypeCast`, and loop vectorization. Afterward, `LegalizeSafeMemoryAccess` inserts per-element `scf.if` bounds checks around loads and stores.

**Observed behavior against Buddy-MLIR `v0.1.5` (`buddy-opt` / LLVM 24 on `openeuler-riscv64`)**:

- **Buddy official contract sample** (`196x768x768` matmul): passes `--kernel-contract-verify --bosc-ame-plan --bosc-ame-plan-verify` (exit 0) and `--bosc-ame-emit` (exit 0), producing `bosc_ame.ddr.frame`, `bosc_ame.dma.abi`, `bosc_ame.tile.load_a/b/acc`, and `bosc_ame.tile.mma` (x8).
- **TileLang RISC-V MLIR outputs** (tested via `--linalg-to-boscame` and `--kernel-contract-annotate --kernel-contract-verify`):
  - `engram_hash_kernel` (`0` `linalg` ops, `14` `scf.for`, `12` `scf.if`, `11` `memref.load`, `9` `memref.store`) -> **rejected** (`exit 1`)
  - `engram_grad_w_reduce_kernel` (`6` `linalg.fill`, `13` `scf.for`, `14` `scf.if`, `15` `memref.load`, `11` `memref.store`, despite using `T.alloc_fragment`, `T.alloc_shared`, `T.copy`, and `T.Pipelined(..., num_stages=2)`) -> **rejected** (`exit 1`)
  - Minimal `T.gemm` + `T.clear(C_local)` control kernel -> emits `linalg.matmul` (`src/target/codegen_riscv.cc:9333-9349`), **but still rejected** (`exit 1`) because `T.clear` also emits a scalar double-loop `memref.store` alongside `linalg.fill`
- **Exact contract failure**:
  ```text
  error: 'memref.store' op kernel contract V22/V25: v1 private scalar stores require rank-zero storage; a single-element write cannot initialize the whole root
  ```

**Why this PR leaves `phase.py` unchanged**: Re-enabling `LowerTileOp` and structured passes in `phase.py` interacts directly with `src/target/codegen_riscv.cc` TIR lowering assumptions and carries high regression risk across the 823-test runtime suite. It should be addressed in a dedicated follow-up PR with full native regression coverage.

**Recommended follow-up directions**:

1. Introduce RISC-V structured `LowerTileOp` / Linalg lowering so `T.copy` and `T.reduce` preserve `linalg.copy` / `linalg.generic` / `linalg.reduce` instead of scalar `scf.for` + `memref.load/store`.
2. Preserve `T.Pipelined` staging semantics (`num_stages`) for async DMA / double-buffering lowering.
3. Eliminate redundant scalar `scf.for` zero-initialization loops that duplicate `linalg.fill` on fragments/buffers (the direct V22/V25 trigger in `T.gemm` and `engram_grad_w_reduce`).
4. Hoist `LegalizeSafeMemoryAccess` bounds checks to tile/block granularity where static shapes or divisibility permit.
5. Annotate on-chip scratch buffers (`memref.alloca`) vs DDR workspaces so downstream planners can distinguish SPM allocations, and keep pure integer address-hashing kernels (such as `engram_hash`) off matrix-engine data paths.

### 5.2 Cross-Path ABI Divergence (`llc` Soft-Float vs `clang -x ir` Hard-Float)

- `tilelang/jit/adapter/riscv/libgen.py:363-396` (`_emit_llc_artifact`) invokes `llc -mtriple=... -filetype=asm|obj` without `-mattr` or `-mabi`, defaulting to soft-float (`lp64`, passing/returning via `a0`).
- `tilelang/jit/adapter/riscv/wrapper.py` (`build_host_shared_library`) invokes `clang -x ir -target <triple>`, defaulting to hard-float (`lp64d`).
- Additionally, `build_qemu_executable` (`tilelang/jit/adapter/riscv/wrapper.py:990-1039`) uses `-nostdlib -static -ffreestanding` without host conversion helpers or `compiler-rt` builtins. Unifying `llc` target attributes/ABI with `clang` requires `.s` disassembly verification on native RISC-V hardware and is left for a dedicated follow-up PR.

### 5.3 Native `riscv64` CI Coverage

Adding a periodic or self-hosted native `riscv64` CI job covering `testing/python/riscv/test_riscv_jit_runtime.py`, `testing/python/riscv/test_riscv_cache.py`, `testing/python/riscv/TileKernels`, and `testing/python/riscv/TileOps` will prevent toolchain/ABI regressions from slipping past x86-only compile checks.

---

## 6. Reference Reports and Raw Evidence

Full investigation logs, native test artifacts, and acceptance scripts are archived on [`BACKt2049/AME-CPU` (`env-reports`)](https://github.com/BACKt2049/AME-CPU/tree/env-reports/docs-riscv):

- [Full environment issue inventory (22 items)](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/reports/TileLang_RISCV_环境问题全量汇总.md)
- [Root-cause summary (5-layer attribution)](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/reports/TileLang_RISCV_环境问题根源摘要.md)
- [Deep compatibility & disassembly RCA](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/reports/TileLang_RISCV_兼容性缺陷深度调查报告.md)
- [Full 823-test execution trace](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/reports/TileLang_RISCV_全量测试Trace.md)
- [Native bring-up & troubleshooting log](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/reports/TileLang_RISCV_踩坑修复记录.md)
- [Reproducible clean-checkout verification report](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/reports/TileLang_RISCV_绕过清单可复现验证报告.md)
- [Raw JUnit XML evidence (823/823 passed)](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/evidence/riscv_results.xml)
- [Official operator inventory output](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/evidence/operator_inventory.json)
- [Idempotent test runner script](https://github.com/BACKt2049/AME-CPU/blob/env-reports/docs-riscv/scripts/run_all_riscv_tests.sh)
