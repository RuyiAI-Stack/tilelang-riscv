from __future__ import annotations

import pytest
import torch

from ._harness import compile_tileops_jit, _ensure_minimal_tileops_kernel_modules
import importlib


def _compile_rows(module, kind, m, n, config):
    """Build the upstream row program with a fixed, reproducible test schedule."""
    _ensure_minimal_tileops_kernel_modules()
    source = importlib.import_module(f"tileops.kernels.reduction.{module}")
    config = dict(config)
    if module == "reduce":
        if kind in ("std", "var", "var_mean"):
            factory = source._welford_reduce_kernel(m, n, kind, 1, "float32")
        elif kind == "prod":
            factory = source._prod_reduce_kernel(m, n, "float32", "float32", 128, 8)
            config = {}
        else:
            factory = source._simple_reduce_kernel(m, n, kind, "float32", "float32")
    elif module == "softmax":
        factory = source._softmax_kernel_single(m, n, kind, "float32", "float32")
        config.pop("tile_n", None)
    elif module == "vector_norm":
        factory = source._vector_norm_kernel(m, n, kind, "float32", "float32")
    elif module == "logical_reduce":
        factory = source._logical_fold_kernel(1, m, n, kind, "float32", 1, 1, 1, "int64" if kind == "count_nonzero" else "int8", False)
        config = {"threads": 128}
    elif module == "argreduce":
        factory = source._argreduce_cta_kernel(m, n, kind, "float32", 4)
        config = {"threads": 128}
    elif module == "logsumexp":
        tile_n = config.pop("tile_n", None)
        factory = source._logsumexp_kernel_tiled(m, n, "float32", tile_n) if tile_n else source._logsumexp_kernel_single(m, n, "float32")
    elif module == "cumulative":
        factory = source._cumulative_kernel(m, n, kind, "float32", 1)
    else:
        raise ValueError(module)
    return compile_tileops_jit(factory, config)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("sum", lambda x: x.sum(dim=1)),
        ("mean", lambda x: x.mean(dim=1)),
        ("amax", lambda x: x.amax(dim=1)),
        ("amin", lambda x: x.amin(dim=1)),
        ("prod", lambda x: torch.prod(x, dim=1)),
        ("std", lambda x: torch.std(x, dim=1, correction=1)),
        ("var", lambda x: torch.var(x, dim=1, correction=1)),
    ],
)
def test_reduce_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    if op_kind == "prod":
        x = torch.linspace(0.99, 1.01, m * n, dtype=torch.float32).reshape(m, n)
    else:
        x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)

    kernel = _compile_rows("reduce", op_kind, m, n, {"block_m": 1, "threads": 128})

    _assert_close(kernel(x.contiguous()), reference(x))


def test_reduce_var_mean_float32_runtime_compare():
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    kernel = _compile_rows("reduce", "var_mean", m, n, {"block_m": 1, "threads": 128})

    actual_var, actual_mean = kernel(x.contiguous())
    expected_var, expected_mean = torch.var_mean(x, dim=1, correction=1)
    _assert_close(actual_var, expected_var)
    _assert_close(actual_mean, expected_mean)


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("softmax", lambda x: torch.softmax(x, dim=1)),
        ("log_softmax", lambda x: torch.log_softmax(x, dim=1)),
    ],
)
def test_softmax_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    kernel = _compile_rows("softmax", op_kind, m, n, {"block_m": 1, "threads": 128})

    _assert_close(kernel(x.contiguous()), reference(x))


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("softmax", lambda x: torch.softmax(x, dim=1)),
        ("log_softmax", lambda x: torch.log_softmax(x, dim=1)),
    ],
)
def test_softmax_large_row_float32_runtime_compare(op_kind, reference):
    # A ragged width retains the original generic loads instead of its opaque
    # CUDA cache-policy vector-load fast path.
    m, n = 2, 513
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    kernel = _compile_rows("softmax", op_kind, m, n, {"block_m": 1, "threads": 128, "tile_n": 256})

    _assert_close(kernel(x.contiguous())[:, :n], reference(x))


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("l1", lambda x: torch.linalg.vector_norm(x, ord=1, dim=1)),
        ("l2", lambda x: torch.linalg.vector_norm(x, ord=2, dim=1)),
        ("inf", lambda x: torch.linalg.vector_norm(x, ord=float("inf"), dim=1)),
    ],
)
def test_vector_norm_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    kernel = _compile_rows("vector_norm", op_kind, m, n, {"block_m": 1, "threads": 128})

    _assert_close(kernel(x.contiguous()), reference(x))


@pytest.mark.parametrize("op_kind", ["any", "all", "count_nonzero"])
def test_logical_reduce_float32_runtime_compare(op_kind):
    m, n = 2, 256
    x = torch.ones((m, n), dtype=torch.float32)
    x[1, 0] = 0
    kernel = _compile_rows("logical_reduce", op_kind, m, n, {"block_m": 1, "threads": 128})
    actual = kernel(x.unsqueeze(0).contiguous())

    if op_kind == "any":
        expected = torch.any(x != 0, dim=1)
        actual = actual.to(torch.bool)
    elif op_kind == "all":
        expected = torch.all(x != 0, dim=1)
        actual = actual.to(torch.bool)
    else:
        expected = torch.count_nonzero(x, dim=1)
    _assert_close(actual, expected)


@pytest.mark.parametrize("n", [256, 513, 4096])
@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("argmax", lambda x: torch.argmax(x, dim=1)),
        ("argmin", lambda x: torch.argmin(x, dim=1)),
    ],
)
def test_argreduce_float32_runtime_compare(op_kind, reference, n):
    # Negative/positive rows plus extrema tied across lanes and across warps.
    m = 4
    ramp = torch.linspace(0.1, 2.0, n, dtype=torch.float32)
    x = torch.stack((-ramp, ramp, -ramp, ramp))
    x[2, [17, 193]] = 3
    x[3, [11, 201]] = -3
    kernel = _compile_rows("argreduce", op_kind, m, n, {"block_m": 1, "threads": 128})

    _assert_close(kernel(x.contiguous()), reference(x))


def test_logsumexp_float32_runtime_compare():
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    kernel = _compile_rows("logsumexp", "logsumexp", m, n, {"block_m": 1, "threads": 128})

    _assert_close(kernel(x.contiguous()), torch.logsumexp(x, dim=1))


def test_logsumexp_tiled_float32_runtime_compare():
    m, n = 2, 512
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    kernel = _compile_rows("logsumexp", "logsumexp", m, n, {"block_m": 1, "threads": 128, "tile_n": 256})

    _assert_close(kernel(x.contiguous()), torch.logsumexp(x, dim=1))


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("sum", lambda x: torch.cumsum(x, dim=1)),
        ("prod", lambda x: torch.cumprod(x, dim=1)),
    ],
)
def test_cumulative_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    if op_kind == "prod":
        x = torch.linspace(0.99, 1.01, m * n, dtype=torch.float32).reshape(m, n)
    else:
        x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    kernel = _compile_rows("cumulative", op_kind, m, n, {"block_m": 1, "block_n": 256, "threads": 128})

    _assert_close(kernel(x.contiguous()), reference(x))
