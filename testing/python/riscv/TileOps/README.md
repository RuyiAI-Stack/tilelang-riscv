# RISC-V TileOps tests

This suite compiles original upstream TileOps programs through the RISC-V MLIR
backend and compares selected executions with independent PyTorch references.
The pinned revision is `67250e8172fa4a97c3ab7cb5b6c269169f12299b` (2026-10-16).
Upstream source is unmodified.

## Adaptation boundary

TileOps owns operator contracts, algorithms, compositions and kernel selection.
TileLang owns language semantics, lowering and runtime execution. Tests use the
upstream algorithms and rely on general compiler/runtime support. Current GDN
decode and prefill tests replace the removed upstream GatedDeltaNet interfaces.

The harness fixes schedules and handles import/API differences: a `tvm.tirx`
alias, a moved CUDA macro import alias, an eager-only custom-op decorator on
older PyTorch, and fixed GPU schedule-selection metadata. It bypasses CUDA
architecture classification when testing original programs on RISC-V. These
are test accommodations, not a shipped backend, device detector, or replacement
operator math. Unsupported constructs must fail in the compiler.

Upstream declares PyTorch `>=2.13,<2.14` and TileLang `0.1.12`; the development
host uses PyTorch 2.12 and SG2044 uses 2.1. These tests do not certify a standard
installation, `torch.compile`, or every public Op interface in those environments.

## New upstream execution coverage

| Test | Scope |
| --- | --- |
| `test_gqa_decode.py` | Original dense GQA decode with D=256, FP16/BF16 and KV lengths 63/65; independent reference and input-preservation checks |
| `test_upstream_attention.py` | Original MLA MMA (FP16/BF16, split 1/2, sequence tails); varlen and paged GQA (FP16/BF16, causal/noncausal, uneven lengths and nonmonotonic pages) |
| `test_gqa_paged_features.py` | 16 original general paged GQA cases: FP16/BF16, one/three splits, NeoX/interleaved RoPE, windows, softcap and raw sinks; dynamic metadata reuse, empty requests, poisoned cache padding and bytewise input-preservation checks |
| `test_upstream_gdn.py` | Original GDN decode and prefill, both state layouts and precisions, output/state/no-mutation checks; prefill lengths 5/67 cover short input and a chunk tail |
| `test_moe_grouped_gemm.py` | Original MMA grouped GEMM, 5 layouts × 2 precisions × 3 activations; empty experts, caller-owned output and repeated calls |
| `test_upstream_public_api.py` | Real `GQAVarlenFwdOp(target="riscv")` with a scoped upstream builder, 8 varlen cases with optional sinks plus 4 original gather orchestration cases with strided destination storage; scoped registration and cache reuse |
| `test_gqa_sliding_window_varlen_fwd.py` | Original varlen sliding-window attention executes with uneven Q/KV lengths, window masking and input-preservation checks |
| `test_gemm.py` | Original W4A16 MMA, FP16/BF16, independently packed INT4 weights and numerical dequantization/GEMM reference |
| `test_paged_cache_gather.py` | 9 original gather cases: FP16/BF16/FP8, scalar and multidimensional rows, dynamic packed ranges/page pools, row/column tails, empty requests, all FP8 encodings and bit-exact outputs |
| `test_paged_kv_cache_write.py` | Original paged KV cache scatter, 10 cases: FP16/BF16 inputs, 16-bit/FP8 cache, dynamic tokens, ragged schedules, repeated writes, skipped slots, empty launch and byte-exact FP8 boundary checks |
| `test_ada_layer_norm_zero.py`, `test_int8_dequant.py` | AdaLayerNormZero and INT8 per-tensor/per-channel/per-block dequantization, FP16/BF16/FP32; per-tensor tests cover staged and direct loads |
| `test_fp8_quant_per_block.py` | Original aligned and unaligned FP8 block quantization, FP16/BF16/FP32, both eviction hints, ragged tiles and random/zero/tiny values; exact output and scale checks |
| `test_int8_quant.py` | Original INT8 per-channel/per-block quantization and SmoothQuant, FP16/BF16/FP32, aligned and ragged rows, zero/tiny inputs, exact codes/scales and input-preservation checks |
| `test_elementwise_activations.py` | Sigmoid FP16/BF16/FP32 at lengths 256/257, streaming and generic loads, signed zero, infinities and large negative inputs |
| `test_sampling_*.py`, `test_chain_speculative_sampling.py` | Min-P, Top-K, Top-P, Top-K+Top-P, probability and speculative sampling; ragged vocabulary, 32/64-thread schedules, independent Philox/acceptance/CDF reference |
| `test_delta_decode.py` | DeltaNet and KDA decode, FP16/BF16, optional Q/K normalization, recurrence output/state and input-preservation checks |

The paged GQA tests exercise RoPE, windows and sinks against independent FP64
references. The cache-write and gather tests preserve upstream flattening,
optional-argument handling and in-place output semantics; FP8 conversion tests
include every encoding and precision boundaries. Gather public-Op tests retain
upstream staging/copy-back for strided destination views. Only compilation and
host kernel selection are rebound within the tests.

The public varlen builder connects tensor metadata and the original kernel factory.
Registration is isolated to tests; general public backend registration is not
shipped. The tested attention contract includes optional sinks and 16-bit data
without RoPE, optional scales, custom score parameters or sliding windows.

Existing tests were migrated to current module/class names, factory signatures
and tensor layouts, including DeltaNet/GLA chunks, dense/paged attention,
normalization, reductions, RoPE, MoE, pooling and convolution. Mamba2 and composite
MoE retain component coverage, not full current-public-Op acceptance.

## Acceptance and gaps

The October 28–29 SG2044 native run at `67250e81` passed **451 checks**:
**436 kernel/component numerical cases, 12 scoped public-API numerical cases,
and 3 rejection checks**. That is **448 numerical executions**, with zero
compile-only cases, skips or expected failures. The run used Buddy
`release/v0.1.2` and LLVM `3845a20590afedfbece4d000f97a6cfbbea3bf43`.
These are parameterized test counts, not supported-operator counts. The
TileKernels suite separately passed 48 numerical cases. See the
[RISC-V test suite](../README.md#validation-scope) for reproduction.

The backend now lowers `tileops::load16`, both 16-byte eviction-hint helpers,
scalar eviction-hint loads and `tileops::load_f32_evict_last` with ordinary
synchronous memory semantics. The 16-byte helpers preserve bits across storage
types and overlapping copies; GPU cache hints do not select a CPU cache policy.
`__fdividef` accepts its FP32 signature and preserves the documented large-divisor
exceptional behavior, using ordinary division elsewhere. This is not a claim of
bitwise agreement with every CUDA approximate-division result.

| Gap | Evidence retained |
| --- | --- |
| W4A16 WGMMA variant | 2 tests require rejection of `tileops::w4a16_dequant_word`; original MMA FP16/BF16 has numerical coverage |
| MLA warp-specialized variant | Rejects `tileops::uniform_wgmma_bases`; numerical MLA coverage uses original MMA |
| Paged GQA | General FP16/BF16 RoPE/window/sink and split numerics are covered; WS RoPE preparation, FP8 and public dispatch remain unvalidated |
| Remaining quantization variants | INT8 per-tensor quantization and INT4 per-group quantization remain unvalidated; multi-CTA synchronization and packed reinterpret support are still gaps |
| Aligned norm/reduction fast paths | The load helpers are supported; aligned RMSNorm/Softmax schedules still need kernel-level numerical validation |
| Sampling and decode breadth | Selected scalar-load sampling schedules and decode paths are validated; packed/vector/cluster variants and full prefill compositions remain gaps |
| KDA prefill | Decode numerics pass; default 256-thread chunk-prepare overflows the host stack, while a 32-thread prepare/scan run yields incorrect output and state, so no prefill numerical support is claimed |

No expected-failure markers remain in this suite. Rejection checks are not
numerical support. There are **16 unmapped declarations**:
**3 implemented upstream and 13 spec-only**. The inventory now has 180 kernel
references and 9 component references; these do not certify full Op contracts.

## Inventory and reproduction

The manifest now lives in `src/tileops/manifest/spec`: **205 operators, 191
implemented and 14 spec-only**. `types.yaml` defines types, not another operator.
Upstream defines spec-only as no fully conforming implementation yet; partial
code may exist. It says nothing about this RISC-V suite's execution status.
For example, paged GQA retains that metadata while its 16-bit kernels have local
numerical evidence.

`operator_inventory.py` reads current declarations, inheritance and delegates
without importing CUDA. It rejects empty/changed schemas and stale references,
and preserves upstream status independently of observed test outcomes. Per-entry
results can overlap and mixed test files can contain other operators; references
do not certify all shapes, dtypes or variants.

```bash
TVM_FFI_DISABLE_TORCH_C_DLPACK=1 TILELANG_DISABLE_CACHE=1 \
TILELANG_RISCV_MLIR_MODE=ON \
python -m pytest -q -rx testing/python/riscv/TileOps --junitxml=tileops.xml

python testing/python/riscv/operator_inventory.py \
  --check --junit tileops.xml --output tileops-inventory.json
```

Use a development Python with the built RISC-V adapter and set
`TILELANG_RISCV_LLVM_ROOT` to the LLVM/MLIR toolchain. The default source checkout
is `3rdparty/TileOPs`; `TILEOPS_ROOT` can select another checkout but then counts
and mappings must be revalidated. `--check` validates inventory structure and
references; `--require-tileops-numerical` intentionally fails while gaps remain.
