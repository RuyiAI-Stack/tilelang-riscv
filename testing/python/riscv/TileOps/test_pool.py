from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_pool_kernel_class


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-5, atol=1e-5)


def test_avg_pool1d_float32_runtime_compare():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    tileops_kernel = get_pool_kernel_class("avg_pool1d", "AvgPool1dKernel")(
        n=1,
        c_in=2,
        l_in=8,
        kernel_l=2,
        stride_l=2,
        pad_l=0,
        ceil_mode=False,
        count_include_pad=True,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
    _assert_close(kernel(x.flatten(0, 1).contiguous()), expected)


def test_avg_pool2d_float32_runtime_compare():
    x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    tileops_kernel = get_pool_kernel_class("avg_pool2d", "AvgPool2dKernel")(
        n=1,
        c_in=2,
        h_in=4,
        w_in=4,
        kernel_h=2,
        kernel_w=2,
        stride_h=2,
        stride_w=2,
        pad_h=0,
        pad_w=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    _assert_close(kernel(x.flatten(0, 1).contiguous()), expected)


def test_avg_pool3d_float32_runtime_compare():
    x = torch.arange(64, dtype=torch.float32).reshape(1, 2, 4, 4, 2)
    tileops_kernel = get_pool_kernel_class("avg_pool3d", "AvgPool3dKernel")(
        n=1,
        c_in=2,
        d_in=4,
        h_in=4,
        w_in=2,
        kernel_d=2,
        kernel_h=2,
        kernel_w=2,
        stride_d=2,
        stride_h=2,
        stride_w=1,
        pad_d=0,
        pad_h=0,
        pad_w=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.nn.functional.avg_pool3d(
        x,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 1),
    )
    _assert_close(kernel(x.flatten(0, 1).contiguous()), expected)


def test_max_pool2d_float32_runtime_compare():
    x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    tileops_kernel = get_pool_kernel_class("max_pool2d", "MaxPool2dKernel")(
        n=1,
        c_in=2,
        h_in=4,
        w_in=4,
        kernel_h=2,
        kernel_w=2,
        stride_h=2,
        stride_w=2,
        pad_h=0,
        pad_w=0,
        dilation_h=1,
        dilation_w=1,
        ceil_mode=False,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
    _assert_close(kernel(x.flatten(0, 1).contiguous()), expected)


def test_max_pool2d_with_indices_float32_runtime_compare():
    x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    tileops_kernel = get_pool_kernel_class("max_pool2d", "MaxPool2dWithIndicesKernel")(
        n=1,
        c_in=2,
        h_in=4,
        w_in=4,
        kernel_h=2,
        kernel_w=2,
        stride_h=2,
        stride_w=2,
        pad_h=0,
        pad_w=0,
        dilation_h=1,
        dilation_w=1,
        ceil_mode=False,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected_val, expected_idx = torch.nn.functional.max_pool2d(
        x,
        kernel_size=2,
        stride=2,
        return_indices=True,
    )
    actual_val, actual_idx = kernel(x.flatten(0, 1).contiguous())
    _assert_close(actual_val, expected_val)
    torch.testing.assert_close(actual_idx.reshape(expected_idx.shape), expected_idx)
