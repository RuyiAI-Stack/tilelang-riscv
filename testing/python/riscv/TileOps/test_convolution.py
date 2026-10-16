from __future__ import annotations

import torch
import torch.nn.functional as F

from ._harness import compile_tileops_kernel, get_kernel_class


def test_conv1d_float32_runtime_compare():
    batch, channels_in, length_in = 1, 4, 8
    channels_out, kernel_size = 4, 3
    stride, padding, dilation = 1, (1, 1), 1
    kernel_cls = get_kernel_class("convolution", "Conv1dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        length_in,
        channels_out,
        kernel_size,
        stride,
        padding,
        torch.float32,
        dilation_l=dilation,
        has_bias=False,
        config={
            "block_m": 4,
            "block_n": 8,
            "block_k": 12,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * length_in,
        dtype=torch.float32,
    ).reshape(batch, channels_in, length_in)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in * kernel_size,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, kernel_size)
    weight_flat = weight.permute(0, 2, 1).contiguous().view(channels_out, channels_in * kernel_size)
    actual = kernel(x, weight_flat)
    expected = F.conv1d(x, weight, bias=None, stride=stride, padding=padding[0], dilation=dilation)
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv1d_bias_float32_runtime_compare():
    batch, channels_in, length_in = 1, 4, 8
    channels_out, kernel_size = 4, 3
    stride, padding, dilation = 1, (1, 1), 1
    kernel_cls = get_kernel_class("convolution", "Conv1dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        length_in,
        channels_out,
        kernel_size,
        stride,
        padding,
        torch.float32,
        dilation_l=dilation,
        has_bias=True,
        config={
            "block_m": 4,
            "block_n": 8,
            "block_k": 12,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * length_in,
        dtype=torch.float32,
    ).reshape(batch, channels_in, length_in)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in * kernel_size,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, kernel_size)
    weight_flat = weight.permute(0, 2, 1).contiguous().view(channels_out, channels_in * kernel_size)
    bias = torch.linspace(-0.1, 0.1, channels_out, dtype=torch.float32)

    actual = kernel(x, weight_flat, bias)
    expected = F.conv1d(x, weight, bias=bias, stride=stride, padding=padding[0], dilation=dilation)
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv1d_pointwise_float32_runtime_compare():
    batch, channels_in, length_in = 1, 4, 8
    channels_out = 4
    kernel_cls = get_kernel_class("convolution", "Conv1dPointwiseKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        length_in,
        channels_out,
        torch.float32,
        has_bias=True,
        config={
            "block_m": 4,
            "block_n": 8,
            "block_k": 4,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * length_in,
        dtype=torch.float32,
    ).reshape(batch, channels_in, length_in)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, 1)
    bias = torch.linspace(-0.1, 0.1, channels_out, dtype=torch.float32)

    actual = kernel(x, weight[:, :, 0].contiguous(), bias)
    expected = F.conv1d(x, weight, bias=bias)
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_group_conv1d_depthwise_float32_runtime_compare():
    batch, channels_in, length_in = 1, 4, 8
    groups = channels_in
    channels_out, kernel_size = 4, 3
    stride, padding, dilation = 1, (1, 1), 1
    kernel_cls = get_kernel_class("convolution", "GroupConv1dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        length_in,
        channels_out,
        kernel_size,
        stride,
        padding,
        torch.float32,
        dilation_l=dilation,
        has_bias=False,
        groups=groups,
        config={
            "block_m": 16,
            "block_n": 8,
            "block_k": 16,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * length_in,
        dtype=torch.float32,
    ).reshape(batch, channels_in, length_in)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * (channels_in // groups) * kernel_size,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in // groups, kernel_size)
    actual = kernel(x, weight)
    expected = F.conv1d(
        x,
        weight,
        bias=None,
        stride=stride,
        padding=padding[0],
        dilation=dilation,
        groups=groups,
    )
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv2d_float32_runtime_compare():
    batch, channels_in, height, width = 1, 3, 5, 5
    channels_out, kernel_h, kernel_w = 4, 2, 2
    stride_h = stride_w = 1
    pad_h = pad_w = 0
    dilation_h = dilation_w = 1
    kernel_cls = get_kernel_class("convolution", "Conv2dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        height,
        width,
        channels_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        torch.float32,
        has_bias=False,
        config={
            "block_m": 4,
            "block_n": 16,
            "block_k": 12,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in * kernel_h * kernel_w,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, kernel_h, kernel_w)
    actual = kernel(x, weight)
    expected = F.conv2d(
        x,
        weight,
        bias=None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
    )
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv2d_bias_float32_runtime_compare():
    batch, channels_in, height, width = 1, 3, 5, 5
    channels_out, kernel_h, kernel_w = 4, 2, 2
    stride_h = stride_w = 1
    pad_h = pad_w = 0
    dilation_h = dilation_w = 1
    kernel_cls = get_kernel_class("convolution", "Conv2dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        height,
        width,
        channels_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        torch.float32,
        has_bias=True,
        config={
            "block_m": 4,
            "block_n": 16,
            "block_k": 12,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in * kernel_h * kernel_w,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, kernel_h, kernel_w)
    bias = torch.linspace(-0.1, 0.1, channels_out, dtype=torch.float32)

    actual = kernel(x, weight, bias)
    expected = F.conv2d(
        x,
        weight,
        bias=bias,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
    )
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv2d_symmetric_float32_runtime_compare():
    batch, channels_in, height, width = 1, 4, 3, 3
    channels_out, kernel_size = 4, 2
    stride, pad, dilation = 1, 0, 1
    kernel_cls = get_kernel_class("convolution", "Conv2dSymmetricKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        height,
        width,
        channels_out,
        kernel_size,
        stride,
        pad,
        dilation,
        torch.float32,
        has_bias=True,
        config={
            "block_m": 4,
            "block_n": 4,
            "block_k": 4,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in * kernel_size * kernel_size,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, kernel_size, kernel_size)
    bias = torch.linspace(-0.1, 0.1, channels_out, dtype=torch.float32)
    x_nhwc = torch.empty((batch, height, width, channels_in), dtype=torch.float32)
    weight_krsc = torch.empty(
        (channels_out, kernel_size, kernel_size, channels_in),
        dtype=torch.float32,
    )
    expected = F.conv2d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=pad,
        dilation=dilation,
    )
    actual = kernel(x, weight, x_nhwc, weight_krsc, bias)
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv2d_1x1_float32_runtime_compare():
    batch, channels_in, height, width = 1, 4, 4, 4
    channels_out = 4
    kernel_cls = get_kernel_class("convolution", "Conv2d1x1Kernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        height,
        width,
        channels_out,
        1,
        1,
        0,
        0,
        torch.float32,
        has_bias=True,
        config={
            "block_m": 4,
            "block_n": 16,
            "block_k": 4,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, 1, 1)
    bias = torch.linspace(-0.1, 0.1, channels_out, dtype=torch.float32)

    actual = kernel(x, weight.view(channels_out, channels_in).contiguous(), bias)
    expected = F.conv2d(x, weight, bias=bias)
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv3d_float32_runtime_compare():
    batch, channels_in, depth, height, width = 1, 2, 3, 3, 3
    channels_out, kernel_d, kernel_h, kernel_w = 4, 2, 2, 2
    stride_d = stride_h = stride_w = 1
    pad_d = pad_h = pad_w = 0
    dilation_d = dilation_h = dilation_w = 1
    kernel_cls = get_kernel_class("convolution", "Conv3dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        depth,
        height,
        width,
        channels_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        torch.float32,
        has_bias=False,
        config={
            "block_m": 4,
            "block_n": 8,
            "block_k": 16,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * depth * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, depth, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in * kernel_d * kernel_h * kernel_w,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, kernel_d, kernel_h, kernel_w)
    actual = kernel(x, weight)
    expected = F.conv3d(
        x,
        weight,
        bias=None,
        stride=(stride_d, stride_h, stride_w),
        padding=(pad_d, pad_h, pad_w),
        dilation=(dilation_d, dilation_h, dilation_w),
    )
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_conv3d_bias_float32_runtime_compare():
    batch, channels_in, depth, height, width = 1, 2, 3, 3, 3
    channels_out, kernel_d, kernel_h, kernel_w = 4, 2, 2, 2
    stride_d = stride_h = stride_w = 1
    pad_d = pad_h = pad_w = 0
    dilation_d = dilation_h = dilation_w = 1
    kernel_cls = get_kernel_class("convolution", "Conv3dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        depth,
        height,
        width,
        channels_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        torch.float32,
        has_bias=True,
        config={
            "block_m": 4,
            "block_n": 8,
            "block_k": 16,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * depth * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, depth, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * channels_in * kernel_d * kernel_h * kernel_w,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in, kernel_d, kernel_h, kernel_w)
    bias = torch.linspace(-0.1, 0.1, channels_out, dtype=torch.float32)

    actual = kernel(x, weight, bias)
    expected = F.conv3d(
        x,
        weight,
        bias=bias,
        stride=(stride_d, stride_h, stride_w),
        padding=(pad_d, pad_h, pad_w),
        dilation=(dilation_d, dilation_h, dilation_w),
    )
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_group_conv2d_depthwise_float32_runtime_compare():
    batch, channels_in, height, width = 1, 4, 4, 4
    groups = channels_in
    channels_out, kernel_h, kernel_w = 4, 2, 2
    stride_h = stride_w = 1
    pad_h = pad_w = 0
    dilation_h = dilation_w = 1
    kernel_cls = get_kernel_class("convolution", "GroupConv2dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        height,
        width,
        channels_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        torch.float32,
        has_bias=False,
        groups=groups,
        config={
            "block_m": 16,
            "block_n": 16,
            "block_k": 4,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * (channels_in // groups) * kernel_h * kernel_w,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in // groups, kernel_h, kernel_w)
    actual = kernel(x, weight)
    expected = F.conv2d(
        x,
        weight,
        bias=None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)


def test_group_conv3d_depthwise_float32_runtime_compare():
    batch, channels_in, depth, height, width = 1, 2, 3, 3, 3
    groups = channels_in
    channels_out, kernel_d, kernel_h, kernel_w = 2, 2, 2, 2
    stride_d = stride_h = stride_w = 1
    pad_d = pad_h = pad_w = 0
    dilation_d = dilation_h = dilation_w = 1
    kernel_cls = get_kernel_class("convolution", "GroupConv3dKernel")
    tileops_kernel = kernel_cls(
        batch,
        channels_in,
        depth,
        height,
        width,
        channels_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        torch.float32,
        has_bias=False,
        groups=groups,
        config={
            "block_m": 16,
            "block_n": 8,
            "block_k": 8,
            "num_stages": 1,
            "threads": 16,
            "enable_rasterization": False,
        },
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.linspace(
        -0.5,
        0.5,
        batch * channels_in * depth * height * width,
        dtype=torch.float32,
    ).reshape(batch, channels_in, depth, height, width)
    weight = torch.linspace(
        -0.25,
        0.25,
        channels_out * (channels_in // groups) * kernel_d * kernel_h * kernel_w,
        dtype=torch.float32,
    ).reshape(channels_out, channels_in // groups, kernel_d, kernel_h, kernel_w)
    actual = kernel(x, weight)
    expected = F.conv3d(
        x,
        weight,
        bias=None,
        stride=(stride_d, stride_h, stride_w),
        padding=(pad_d, pad_h, pad_w),
        dilation=(dilation_d, dilation_h, dilation_w),
        groups=groups,
    )
    torch.testing.assert_close(actual, expected.contiguous(), rtol=1e-5, atol=1e-5)
