"""Original paged-cache gather: dynamic packed ranges, row views and FP8 bits."""

import math

import pytest
import torch

from ._harness import compile_tileops_jit


CASES = [
    pytest.param(torch.float16, torch.float16, (3, 5), 7, False, id="fp16-odd-page"),
    pytest.param(torch.bfloat16, torch.bfloat16, (8, 129), 16, True, id="bf16-wide-starts"),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, (576,), 64, True, id="fp8-bf16-mla-row"),
    pytest.param(torch.float8_e4m3fn, torch.float16, (320,), 7, False, id="fp8-fp16-odd-page"),
    pytest.param(torch.float16, torch.float16, (), 4, True, id="scalar-row"),
    pytest.param(torch.bfloat16, torch.bfloat16, (129,), 7, False, id="bf16-column-tail"),
    pytest.param(torch.float16, torch.float16, (257,), 64, True, id="fp16-large-row-tail"),
]


def gather_reference(cache, table, offsets, starts, scale, dst):
    result = torch.empty_like(dst)
    flat_cache = cache.float().flatten(0, 1)
    for request, (begin, end) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
        start = starts[request].item() if starts is not None else 0
        positions = torch.arange(start, start + end - begin)
        slots = table[request, positions // cache.shape[1]].long() * cache.shape[1]
        slots += positions % cache.shape[1]
        values = flat_cache[slots]
        if scale is not None:
            values = values * scale[0]
        result[begin:end] = values.to(dst.dtype)
    return result


def assert_gather_equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = ~torch.isnan(expected)
    # In particular, preserve the sign of zero during copy/dequantization.
    torch.testing.assert_close(actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], rtol=0, atol=0)


def bind_gather_compilation(gather):
    """Retain upstream forward's flattening and optional-argument handling."""
    compiled = compile_tileops_jit(gather.kernel, gather.config)

    def select_compiled(**schedule):
        assert schedule == gather.config
        return compiled

    gather.kernel = select_compiled
    return compiled


@pytest.mark.parametrize("cache_dtype,out_dtype,row_shape,page_size,has_starts", CASES)
def test_paged_cache_gather_runtime_compare(cache_dtype, out_dtype, row_shape, page_size, has_starts):
    from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherKernel

    row_width = math.prod(row_shape)
    config = None if row_width >= 576 else dict(block_rows=4, block_cols=32, threads=128)
    gather = PagedKVCacheGatherKernel(page_size, row_width, cache_dtype, out_dtype, has_starts, config=config)
    kernel = bind_gather_compilation(gather)
    generator = torch.Generator().manual_seed(261)
    scale = torch.tensor([0.7], dtype=torch.float32) if cache_dtype == torch.float8_e4m3fn else None
    try:
        # One compilation, varying total rows, batch, table width and cache pool.
        # Multiple full row tiles, page crossings, and repeated/empty requests.
        calls = [
            ([0, page_size + 3, 1, 0, 2 * page_size + 5], [1, page_size - 1, 0, 2, 3], 5),
            ([page_size + 1, 0, 2, 1], [2, 0, 3, 0], 4),
            ([1, 2], [0, 1], 2),
            ([0, 0], [0, 1], 2),
        ]
        for lengths, starts_values, width in calls:
            batch = len(lengths)
            starts = torch.tensor(starts_values, dtype=torch.int32) if has_starts else None
            effective_starts = starts_values if has_starts else [0] * batch
            assert all(start + length <= width * page_size for start, length in zip(effective_starts, lengths, strict=True))
            offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
            pool_pages = batch * width + 3
            table = torch.stack([torch.randperm(pool_pages, generator=generator)[:width] for _ in range(batch)]).to(torch.int32)
            values = torch.randn((pool_pages, page_size, *row_shape), generator=generator)
            # Every selected page contains zeros with both signs and a NaN.
            flat = values.reshape(pool_pages, -1)
            flat[:, :3] = torch.tensor([-0.0, 0.0, float("nan")])
            cache = values.to(cache_dtype)
            dst = torch.full((sum(lengths), *row_shape), float("nan"), dtype=out_dtype)
            inputs = [t for t in (cache, table, offsets, starts, scale) if t is not None]
            input_before = [tensor.view(torch.uint8).clone() for tensor in inputs]
            expected = gather_reference(cache, table, offsets, starts, scale, dst)
            assert gather(dst, cache, table, offsets, starts, scale) is None
            assert_gather_equal(dst, expected)
            for tensor, original in zip(inputs, input_before, strict=True):
                assert torch.equal(tensor.view(torch.uint8), original)
    finally:
        kernel.close()


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_paged_cache_gather_every_fp8_encoding(out_dtype):
    from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherKernel

    gather = PagedKVCacheGatherKernel(1, 256, torch.float8_e4m3fn, out_dtype)
    kernel = bind_gather_compilation(gather)
    encoded = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(1, 1, 256)
    dst = torch.empty((1, 256), dtype=out_dtype)
    table = torch.tensor([[0]], dtype=torch.int32)
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    scale = torch.tensor([0.625], dtype=torch.float32)
    try:
        assert gather(dst, encoded, table, offsets, scale=scale) is None
    finally:
        kernel.close()
    expected = (encoded.float() * scale[0]).to(out_dtype).reshape_as(dst)
    assert_gather_equal(dst, expected)
