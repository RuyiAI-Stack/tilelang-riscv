"""Public dispatch wired to an original upstream program, without replacement math.

Registration is scoped to each test. This is an integration proof, not a shipped
RISC-V backend or a claim that every parameter combination is supported.
"""

import pytest
import torch

from ._harness import compile_tileops_jit
from .test_upstream_attention import attention_reference


@pytest.fixture
def isolated_registry(monkeypatch):
    import torch.fx.experimental.symbolic_shapes as symbolic_shapes

    def unsupported_symbolic_boolean(*_args):
        raise NotImplementedError("Legacy PyTorch fixture supports concrete eager shapes only")

    # PyTorch 2.1 lacks these imports. Concrete eager checks never call them;
    # fail explicitly if an upstream change starts using symbolic execution.
    for name in ("sym_and", "sym_or"):
        if not hasattr(symbolic_shapes, name):
            monkeypatch.setattr(symbolic_shapes, name, unsupported_symbolic_boolean, raising=False)
    if not hasattr(torch.compiler, "is_compiling"):
        from torch._dynamo import is_compiling

        monkeypatch.setattr(torch.compiler, "is_compiling", is_compiling, raising=False)
    from tileops.backend import registry

    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.REGISTERED_KERNEL_TYPES.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    try:
        yield registry
    finally:
        registry.restore(state)


@pytest.mark.parametrize("has_sinks", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_varlen_dispatch_to_upstream(isolated_registry, causal, dtype, has_sinks):
    from tileops.attention import GQAVarlenFwdOp
    from tileops.kernels.attention.gqa.prefill_varlen import _gqa_prefill_varlen_fwd_kernel

    compiled = []

    def build(q, k, v, cq, ck, *optional, **params):
        assert all(spec is None for spec in optional[:-1])
        assert (optional[-1] is not None) == has_sinks
        assert params["pos_encoding_mode"] == "none"
        assert params["window_size_left"] == params["window_size_right"] == -1
        factory = _gqa_prefill_varlen_fwd_kernel(
            cq.shape[0] - 1, q.shape[1], k.shape[1], q.shape[2], params["is_causal"], dtype=str(q.dtype).split(".")[-1], has_sinks=has_sinks
        )
        kernel = compile_tileops_jit(factory, dict(block_m=16, block_n=16, num_stages=1, threads=128))
        compiled.append(kernel)

        def run(q, k, v, cq, ck, *optional):
            return kernel(q, k, v, cq, ck, optional[-1] if has_sinks else torch.empty(q.shape[1], dtype=torch.float32))

        return run

    isolated_registry.register_kernel_builder("GQAVarlenFwdOp", "riscv", build)
    op = GQAVarlenFwdOp(target="riscv", is_causal=causal)
    torch.manual_seed(97)
    q = (torch.randn(20, 4, 32) * 0.2).to(dtype)
    k = (torch.randn(55, 2, 32) * 0.2).to(dtype)
    v = torch.randn_like(k)
    cq, ck = torch.tensor([0, 17, 20], dtype=torch.int32), torch.tensor([0, 35, 55], dtype=torch.int32)
    try:
        for iteration, values in enumerate((v, v * 0.5)):
            sinks = torch.tensor([-2.0 + iteration, float("inf"), -float("inf"), 3.0 - iteration]) if has_sinks else None
            inputs = [t for t in (q, k, values, cq, ck, sinks) if t is not None]
            originals = [t.view(torch.uint8).clone() for t in inputs]
            actual = op(q, k, values, cq, ck, sinks=sinks)
            torch.testing.assert_close(actual, attention_reference(q, k, values, cq, ck, causal, sinks), atol=4e-3, rtol=2e-2)
            if has_sinks:
                assert torch.count_nonzero(actual[:, 1]) == 0
            for tensor, original in zip(inputs, originals, strict=True):
                assert torch.equal(tensor.view(torch.uint8), original)
        assert op.serving_target == "riscv"
        assert len(compiled) == 1
    finally:
        for kernel in compiled:
            kernel.close()


@pytest.mark.parametrize(
    "cache_dtype,out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
)
def test_public_gather_preserves_strided_destination(isolated_registry, cache_dtype, out_dtype):
    from tileops.attention import PagedKVCacheGatherFwdOp
    from tileops.backend import BUILTIN
    from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherKernel
    from .test_paged_cache_gather import bind_gather_compilation, gather_reference, assert_gather_equal

    compiled = []

    class HostGather(PagedKVCacheGatherKernel):
        # Test-scoped compilation binding; all math and forward methods inherited.
        devices = frozenset({"cpu"})

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            compiled.append(bind_gather_compilation(self))

    isolated_registry.register_kernel_type("PagedKVCacheGatherFwdOp", "riscv_test_gather", HostGather)
    op = PagedKVCacheGatherFwdOp(out_dtype=out_dtype, target=BUILTIN)
    generator = torch.Generator().manual_seed(262)
    cache = torch.randn((11, 7, 2, 17), generator=generator).to(cache_dtype)
    table = torch.tensor([[9, 2, 7, 0], [1, 8, 3, 6], [5, 4, 10, 2]], dtype=torch.int32)
    offsets = torch.tensor([0, 17, 17, 26], dtype=torch.int32)
    starts = torch.tensor([3, 2, 6], dtype=torch.int32)
    scale = torch.tensor([0.625]) if cache_dtype == torch.float8_e4m3fn else None
    inputs = [t for t in (cache, table, offsets, starts, scale) if t is not None]
    originals = [t.view(torch.uint8).clone() for t in inputs]
    backing = torch.full((26, 2, 34), -123, dtype=out_dtype)
    dst = backing[..., ::2]
    pointer = dst.data_ptr()
    try:
        for _iteration in range(2):
            dst.fill_(float("nan"))
            assert op(dst, cache, table, offsets, starts, scale) is None
            assert_gather_equal(dst, gather_reference(cache, table, offsets, starts, scale, dst))
            assert dst.data_ptr() == pointer
            assert torch.all(backing[..., 1::2] == -123)
        assert len(compiled) == 1
        assert op.serving_target is BUILTIN
        for tensor, original in zip(inputs, originals, strict=True):
            assert torch.equal(tensor.view(torch.uint8), original)
    finally:
        for kernel in compiled:
            kernel.close()
