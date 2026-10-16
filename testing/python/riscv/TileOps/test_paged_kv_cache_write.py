"""Original paged KV cache writes, including dynamic extents and FP8 semantics."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class


CACHE_DTYPES = [
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float8_e4m3fn),
    (torch.bfloat16, torch.float8_e4m3fn),
]


def cache_write_reference(k, v, k_pages, v_pages, slots, k_scale, v_scale):
    """Independent scatter; FP8 saturates overflow/infinities but preserves NaN."""
    expected = [k_pages.clone(), v_pages.clone()]
    for source, pool, scale in zip((k, v), expected, (k_scale, v_scale)):
        flat_pool = pool.reshape(-1, *source.shape[1:])
        for token, slot in enumerate(slots.tolist()):
            if slot >= 0:
                row = source[token].float()
                if pool.dtype == torch.float8_e4m3fn:
                    row = (row / scale[0]).clamp(-448, 448)
                flat_pool[slot] = row.to(pool.dtype)
    return expected


def assert_cache_equal(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0, equal_nan=True)
    # Float comparisons alone miss signed zero and changes to untouched FP8 bytes.
    finite = ~torch.isnan(expected.float())
    storage_dtype = torch.uint8 if actual.dtype == torch.float8_e4m3fn else torch.int16
    torch.testing.assert_close(actual.view(storage_dtype)[finite], expected.view(storage_dtype)[finite], rtol=0, atol=0)


def compile_cache_writer(heads, dim, pages, page_size, dtype, cache_dtype, config):
    """Bind only compilation to RISC-V; retain upstream forward/empty-call behavior."""
    cls = get_kernel_class("attention.paged_cache_write", "PagedKVCacheWriteKernel")
    writer = cls(heads, dim, pages, page_size, dtype, cache_dtype, config=config)
    compiled = compile_tileops_jit(writer.kernel, writer.config)
    launches = []

    def select_compiled(**schedule):
        assert schedule == writer.config

        def run(*args):
            launches.append(args[0].shape[0])
            compiled(*args)

        return run

    writer.kernel = select_compiled
    return writer, compiled, launches


@pytest.mark.parametrize("dtype,cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize(
    "heads,dim,config",
    [
        (2, 64, {"block_rows": 16, "block_cols": 128, "threads": 128}),
        (3, 17, {"block_rows": 4, "block_cols": 32, "threads": 32}),
    ],
    ids=["default-schedule", "row-and-column-tails"],
)
def test_paged_kv_cache_write_runtime_compare(dtype, cache_dtype, heads, dim, config):
    """Reuse one program across token counts, overwrite rows, skip slots, and empty input."""
    pages, page_size = 3, 7
    torch.manual_seed(210)
    k_pages = torch.full((pages, page_size, heads, dim), 0.25, dtype=cache_dtype)
    v_pages = torch.full_like(k_pages, -0.5)
    pointers = (k_pages.data_ptr(), v_pages.data_ptr())
    writer, compiled, launches = compile_cache_writer(heads, dim, pages, page_size, dtype, cache_dtype, config)
    try:
        # The third call crosses a row tile and reuses previously written slots.
        for slot_values in ([20, -1, 0, 6, 13], [6], list(range(16, -1, -1)), [-1] * 3, []):
            tokens = len(slot_values)
            k = (torch.randn(tokens, heads, dim) * 0.2).to(dtype)
            v = (torch.randn(tokens, heads, dim) * 0.3).to(dtype)
            slots = torch.tensor(slot_values, dtype=torch.int64)
            fp8 = cache_dtype == torch.float8_e4m3fn
            ks = torch.tensor([0.3]) if fp8 else None
            vs = torch.tensor([0.7]) if fp8 else None
            readonly = [t for t in (k, v, slots, ks, vs) if t is not None]
            before = [t.clone() for t in readonly]
            expected = cache_write_reference(k, v, k_pages, v_pages, slots, ks, vs)
            assert writer(k, v, k_pages, v_pages, slots, ks, vs) is None
            assert_cache_equal(k_pages, expected[0])
            assert_cache_equal(v_pages, expected[1])
            assert pointers == (k_pages.data_ptr(), v_pages.data_ptr())
            for actual, original in zip(readonly, before):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert launches == [5, 1, 17, 3]  # The original wrapper skips the empty call.
    finally:
        compiled.close()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_kv_cache_write_fp8_special_values(dtype):
    values = torch.tensor(
        [
            1000,
            -1000,
            448,
            -448,
            1.0625,
            1.1875,
            float("nan"),
            float("inf"),
            -float("inf"),
            0.0,
            -0.0,
            2**-10,
            3 * 2**-10,
            -(2**-10),
            -3 * 2**-10,
            0.5,
        ],
        dtype=dtype,
    )
    k = values.reshape(1, 1, 16).repeat(3, 1, 1)
    v = -k
    slots = torch.tensor([3, -1, 0], dtype=torch.int64)
    k_pages = torch.full((2, 2, 1, 16), 0.25, dtype=torch.float8_e4m3fn)
    v_pages = torch.full_like(k_pages, -0.5)
    ks, vs = torch.ones(1), torch.ones(1)
    before = [t.clone() for t in (k, v, slots, ks, vs)]
    expected = cache_write_reference(k, v, k_pages, v_pages, slots, ks, vs)
    writer, compiled, _ = compile_cache_writer(
        1,
        16,
        2,
        2,
        dtype,
        torch.float8_e4m3fn,
        {"block_rows": 4, "block_cols": 32, "threads": 32},
    )
    try:
        assert writer(k, v, k_pages, v_pages, slots, ks, vs) is None
        assert_cache_equal(k_pages, expected[0])
        assert_cache_equal(v_pages, expected[1])
        for actual, original in zip((k, v, slots, ks, vs), before):
            torch.testing.assert_close(actual, original, rtol=0, atol=0, equal_nan=True)
    finally:
        compiled.close()
