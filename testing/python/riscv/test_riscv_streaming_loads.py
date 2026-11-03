"""Streaming-load cache hints retain their ordinary memory semantics on RISC-V."""

import pytest
import torch
import tilelang
import tilelang.language as T


@pytest.mark.parametrize("helper", ["load16", "load16_evict_first", "load16_evict_last"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float16, torch.int32),
        (torch.bfloat16, torch.int32),
        (torch.int32, torch.float16),
        (torch.int8, torch.int64),
        (torch.float8_e4m3fn, torch.int32),
    ],
)
def test_streaming_vector_load_preserves_bytes(helper, src_dtype, dst_dtype):
    src_bytes = torch.empty((), dtype=src_dtype).element_size()
    dst_bytes = torch.empty((), dtype=dst_dtype).element_size()
    src_cols, dst_cols = 32 // src_bytes, 32 // dst_bytes
    src_name, dst_name = str(src_dtype).split(".")[-1], str(dst_dtype).split(".")[-1]
    extern_name = "tileops::" + helper

    @T.prim_func
    def copy(src: T.Tensor((2, src_cols), src_name), dst: T.Tensor((2, dst_cols), dst_name)):
        with T.Kernel(1, threads=1):
            T.call_extern("handle", extern_name, T.address_of(dst[0, 24 // dst_bytes]), T.address_of(src[0, 16 // src_bytes]))

    # Include sign bits and NaN encodings; every copied bit must survive packing
    # into another storage type and a destination span crossing a row boundary.
    data = torch.tensor(([0, 128, 255, 127, 1, 129, 254, 126] * 8), dtype=torch.uint8)
    before = data.clone()
    destination = torch.full((64,), 0x5A, dtype=torch.uint8)
    expected = destination.clone()
    expected[24:40] = data[16:32]
    kernel = tilelang.compile(copy, target="riscv")
    try:
        kernel(data.view(src_dtype).reshape(2, src_cols), destination.view(dst_dtype).reshape(2, dst_cols))
        torch.testing.assert_close(destination, expected, atol=0, rtol=0)
        torch.testing.assert_close(data, before, atol=0, rtol=0)
    finally:
        kernel.close()


@pytest.mark.parametrize("helper", ["load_evict_first", "load_evict_last", "load_f32_evict_last"])
def test_streaming_scalar_load_preserves_values(helper):
    extern_name = "tileops::" + helper

    @T.prim_func
    def load(src: T.Tensor((2, 4), "float32"), dst: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=1):
            for i in T.serial(8):
                dst[i] = T.call_extern("float32", extern_name, T.address_of(src[i // 4, i % 4]))

    data = torch.tensor([0.0, -0.0, 1.0, -3.5, float("inf"), -float("inf"), float("nan"), 1e-38])
    kernel = tilelang.compile(load, target="riscv", out_idx=[1])
    try:
        actual = kernel(data.reshape(2, 4))
        torch.testing.assert_close(actual.view(torch.int32), data.view(torch.int32), atol=0, rtol=0)
    finally:
        kernel.close()


def test_streaming_vector_load_reads_before_overlapping_stores():
    @T.prim_func
    def copy(data: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=1):
            T.call_extern("handle", "tileops::load16", T.address_of(data[2]), T.address_of(data[0]))

    data = torch.arange(8, dtype=torch.int32)
    expected = torch.tensor([0, 1, 0, 1, 2, 3, 6, 7], dtype=torch.int32)
    kernel = tilelang.compile(copy, target="riscv")
    try:
        kernel(data)
        torch.testing.assert_close(data, expected)
    finally:
        kernel.close()
