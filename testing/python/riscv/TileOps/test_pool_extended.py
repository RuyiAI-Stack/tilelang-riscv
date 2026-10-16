"""Pool families omitted from the original RISC-V smoke suite."""

import pytest
import torch

from ._harness import compile_tileops_kernel, get_pool_kernel_class


@pytest.mark.parametrize("with_indices", [False, True])
@pytest.mark.parametrize("dimension", [1, 3])
def test_max_pool_runtime_compare(dimension, with_indices):
    torch.manual_seed(21)
    shape = (1, 2, 9) if dimension == 1 else (1, 2, 3, 3, 3)
    x = torch.randn(shape)
    name = f"MaxPool{dimension}d" + ("WithIndicesKernel" if with_indices else "Kernel")
    params = dict(n=1, c_in=2, dtype=x.dtype, ceil_mode=True)
    axes = ["w"] if dimension == 1 else ["d", "h", "w"]
    for axis, size in zip(axes, shape[2:]):
        params.update({f"{axis}_in": size, f"kernel_{axis}": 3, f"stride_{axis}": 2, f"pad_{axis}": 1, f"dilation_{axis}": 1})
    if dimension == 1:
        params["l_in"] = params.pop("w_in")
    else:
        params["config"] = {"block_m": 8, "threads": 32}
    instance = get_pool_kernel_class(f"max_pool{dimension}d", name)(**params)
    kernel = compile_tileops_kernel(instance)
    try:
        actual = kernel(x.flatten(0, 1).contiguous())
    finally:
        kernel.close()
    expected = getattr(torch.nn.functional, f"max_pool{dimension}d")(
        x, kernel_size=3, stride=2, padding=1, ceil_mode=True, return_indices=with_indices
    )
    if not with_indices:
        actual, expected = (actual,), (expected,)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.reshape(reference.shape), reference)


@pytest.mark.parametrize(
    "kind,name,with_indices",
    [
        ("avg", "AdaptiveAvgPool2dKernel", False),
        ("max", "AdaptiveMaxPool2dKernel", False),
        ("max", "AdaptiveMaxPool2dWithIndicesKernel", True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_adaptive_pool_runtime_compare(kind, name, with_indices, dtype):
    torch.manual_seed(22)
    x = torch.randn(1, 2, 5, 7).to(dtype)
    instance = get_pool_kernel_class(f"adaptive_{kind}_pool2d", name)(n=1, c_in=2, h_in=5, w_in=7, out_h=3, out_w=2, dtype=dtype)
    kernel = compile_tileops_kernel(instance)
    try:
        actual = kernel(x.flatten(0, 1).contiguous() if kind == "max" else x)
    finally:
        kernel.close()
    kwargs = {"return_indices": with_indices} if kind == "max" else {}
    expected = getattr(torch.nn.functional, f"adaptive_{kind}_pool2d")(x.float(), (3, 2), **kwargs)
    if not with_indices:
        actual, expected = (actual,), (expected,)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.reshape(reference.shape), reference.to(result.dtype))
