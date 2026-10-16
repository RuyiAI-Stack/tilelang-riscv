"""Runtime FP8 casts, independent of any operator library."""

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.tladapter.toolchain import resolve_llvm_root

pytest.importorskip("tilelang.tladapter._native")
if resolve_llvm_root(required=False) is None:
    pytest.skip("LLVM/MLIR toolchain unavailable", allow_module_level=True)


FORMATS = ["float8_e4m3fn", "float8_e4m3fnuz"]


@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("output_dtype", ["float32", "float16", "bfloat16"])
def test_fp8_decode_all_encodings(dtype, output_dtype):
    @T.prim_func
    def decode(src: T.Tensor((256,), dtype), dst: T.Tensor((256,), output_dtype)):
        with T.Kernel(2, threads=32) as block:
            for i in T.Parallel(128):
                dst[block * 128 + i] = T.Cast(output_dtype, src[block * 128 + i])

    source = torch.arange(256, dtype=torch.uint8).view(getattr(torch, dtype))
    expected = source.to(getattr(torch, output_dtype))
    kernel = tilelang.compile(decode, target="riscv", out_idx=[1])
    try:
        actual = kernel(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(expected)
        storage_dtype = torch.int32 if output_dtype == "float32" else torch.int16
        torch.testing.assert_close(actual.view(storage_dtype)[finite], expected.view(storage_dtype)[finite], rtol=0, atol=0)
    finally:
        kernel.close()


def conversion_inputs(dtype, source_dtype):
    if source_dtype != "float32":
        # All 65,536 encodings include every finite value, both zeros, infinities
        # and all NaN payloads of the two supported 16-bit source formats.
        return torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(getattr(torch, source_dtype))
    values = torch.arange(128, dtype=torch.uint8).view(getattr(torch, dtype)).float()
    values = values[torch.isfinite(values)]
    midpoints = (values[:-1] + values[1:]) / 2
    neighbors = torch.cat(
        [
            values,
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
            torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf"))),
            torch.tensor([0.0, 2**-149, 2**-126, 1000.0, torch.finfo(torch.float32).max, float("inf"), float("nan")]),
        ]
    )
    return torch.cat([neighbors, -neighbors])


@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("source_dtype", ["float32", "float16", "bfloat16"])
def test_fp8_encode_rounding_saturation_and_special_values(dtype, source_dtype):
    source = conversion_inputs(dtype, source_dtype)
    count = source.numel()

    @T.prim_func
    def encode(src: T.Tensor((count,), source_dtype), dst: T.Tensor((count,), dtype)):
        with T.Kernel(T.ceildiv(count, 128), threads=32) as block:
            for i in T.Parallel(128):
                index = block * 128 + i
                if index < count:
                    dst[index] = T.Cast(dtype, src[index])

    dt = getattr(torch, dtype)
    limit = torch.finfo(dt).max
    # TileLang's casts saturate finite overflow/infinity before nearest-even
    # rounding. PyTorch's non-saturating cast supplies the rounding reference.
    expected = source.float().clamp(-limit, limit).to(dt)
    kernel = tilelang.compile(encode, target="riscv", out_idx=[1])
    try:
        actual = kernel(source)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(expected.float())
        torch.testing.assert_close(actual.view(torch.uint8)[finite], expected.view(torch.uint8)[finite], rtol=0, atol=0)
    finally:
        kernel.close()
