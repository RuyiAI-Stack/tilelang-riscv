"""IEEE arithmetic semantics shared by all libraries using the RISC-V backend."""

import pytest
import torch
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.tladapter.toolchain import resolve_llvm_root

pytest.importorskip("tilelang.tladapter._native")
if resolve_llvm_root(required=False) is None:
    pytest.skip("LLVM/MLIR toolchain unavailable", allow_module_level=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ieee_division_special_values(dtype):
    @T.prim_func
    def divide(a: T.Tensor((10,), dtype), b: T.Tensor((10,), dtype), out: T.Tensor((10,), dtype)):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(10):
                out[i] = T.ieee_fdiv(a[i], b[i], "rn")

    dt = getattr(torch, dtype)
    tiny = torch.finfo(dt).tiny
    a = torch.tensor([1.0, -1.0, 0.0, -0.0, float("inf"), float("nan"), tiny, -tiny, 1.0, -1.0], dtype=dt)
    b = torch.tensor([3.0, 3.0, 2.0, 2.0, float("inf"), 1.0, 2.0, 2.0, 0.0, 0.0], dtype=dt)
    kernel = tilelang.compile(divide, target="riscv", out_idx=[2])
    try:
        actual = kernel(a, b)
        expected = a / b
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(actual[2:4]), torch.signbit(expected[2:4]))
        assert "arith.divf" in kernel.get_kernel_source()
    finally:
        kernel.close()


@pytest.mark.parametrize("dtype,half_precision", [("float32", 13), ("float64", 27)])
def test_ieee_fma_has_one_rounding(dtype, half_precision):
    @T.prim_func
    def fused(a: T.Tensor((1,), dtype), b: T.Tensor((1,), dtype), c: T.Tensor((1,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=1):
            out[0] = T.ieee_fmaf(a[0], b[0], c[0], "rn")

    dt = getattr(torch, dtype)
    epsilon = 2.0**-half_precision
    a, b, c = [torch.tensor([v], dtype=dt) for v in (1 + epsilon, 1 - epsilon, -1)]
    # Exact product is 1 - epsilon**2; an intermediate rounding loses that term.
    expected = torch.tensor([-(epsilon**2)], dtype=dt)
    assert (a * b + c).item() == 0
    kernel = tilelang.compile(fused, target="riscv", out_idx=[3])
    try:
        torch.testing.assert_close(kernel(a, b, c), expected, atol=0, rtol=0)
        assert "math.fma" in kernel.get_kernel_source()
    finally:
        kernel.close()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ieee_reciprocal_special_values(dtype):
    @T.prim_func
    def reciprocal(a: T.Tensor((7,), dtype), out: T.Tensor((7,), dtype)):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(7):
                out[i] = T.ieee_frcp(a[i], "rn")

    dt = getattr(torch, dtype)
    data = torch.tensor([3.0, -3.0, 0.0, -0.0, float("inf"), -float("inf"), float("nan")], dtype=dt)
    kernel = tilelang.compile(reciprocal, target="riscv", out_idx=[1])
    try:
        actual = kernel(data)
        torch.testing.assert_close(actual, 1.0 / data, atol=0, rtol=0, equal_nan=True)
        assert "arith.divf" in kernel.get_kernel_source()
    finally:
        kernel.close()


@pytest.mark.parametrize("operation", ["div", "fma", "rcp"])
@pytest.mark.parametrize("rounding", ["rz", "ru", "rd"])
def test_ieee_directed_rounding_rejected(operation, rounding):
    @T.prim_func
    def directed(a: T.Tensor((1,), "float32"), b: T.Tensor((1,), "float32"), out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            if operation == "div":
                out[0] = T.ieee_fdiv(a[0], b[0], rounding)
            elif operation == "fma":
                out[0] = T.ieee_fmaf(a[0], b[0], T.float32(1), rounding)
            else:
                out[0] = T.ieee_frcp(a[0], rounding)

    with pytest.raises(tvm.error.InternalError, match="Unsupported.*(ieee_fdiv|ieee_fmaf|ieee_frcp)"):
        tilelang.compile(directed, target="riscv", out_idx=[2])
