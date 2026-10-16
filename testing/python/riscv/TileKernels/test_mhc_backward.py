"""Exercise all mHC backward factories against independent autograd references."""

import pytest
import torch
import tilelang
import tilelang.language as T

from tile_kernels.mhc.expand_kernel import expand_to_mhc_bwd_tl
from tile_kernels.mhc.head_compute_mix_kernel import _mhc_head_compute_mix_bwd
from tile_kernels.mhc.pre_split_mixes_kernel import _mhc_pre_split_mixes_bwd
from tile_kernels.mhc.pre_apply_mix_kernel import _mhc_pre_apply_mix_bwd
from tile_kernels.mhc.post_kernel import _mhc_post_bwd
from tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_bwd
from tile_kernels.mhc.norm_fn_kernel import (
    _mhc_fn_normw_merge_bwd,
    _mhc_pre_norm_fn_fwd_mul,
    _mhc_pre_norm_fn_fwd_norm,
    _mhc_pre_norm_fn_bwd_norm,
    _mhc_pre_norm_fn_bwd_mul,
)


def _tensor(*shape, bf16=False, grad=False):
    # Random (seeded per test) inputs avoid cancellation and constant gradients.
    return (torch.randn(*shape) * 0.2).to(torch.bfloat16 if bf16 else torch.float32).requires_grad_(grad)


def _run(factory, config, inputs, outputs):
    kernel = tilelang.compile(factory.get_tir(*config), out_idx=outputs, target="riscv")
    try:
        return kernel(*inputs)
    finally:
        kernel.close()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


def test_expand_backward():
    dy = _tensor(3, 4, 17, bf16=True)
    actual = _run(expand_to_mhc_bwd_tl, (17, 4), [dy], [1])
    torch.testing.assert_close(actual, dy.float().sum(1).bfloat16())


def test_head_compute_mix_backward():
    x, scale, base = _tensor(5, 4, grad=True), _tensor(1, grad=True), _tensor(4, grad=True)
    dy = _tensor(5, 4)
    y = torch.sigmoid(x * scale + base)
    expected = torch.autograd.grad(y, (x, scale, base), dy)
    dx, ds, db = _run(_mhc_head_compute_mix_bwd, (4, 2, 2), [dy, x, scale, base], [4, 5, 6])
    for actual, ref in zip((dx, ds.sum(0), db.sum(0)), expected):
        torch.testing.assert_close(actual, ref, atol=1e-6, rtol=1e-4)


def test_pre_split_mixes_backward():
    x, scale, base = _tensor(4, 24, grad=True), _tensor(3, grad=True), _tensor(24, grad=True)
    pre = torch.sigmoid(x[:, :4] * scale[0] + base[:4])
    post = 2 * torch.sigmoid(x[:, 4:8] * scale[1] + base[4:8])
    comb = x[:, 8:] * scale[2] + base[8:]
    grads = [_tensor(*y.shape) for y in (pre, post, comb)]
    expected = torch.autograd.grad((pre, post, comb), (x, scale, base), grads)
    dx, ds, db = _run(_mhc_pre_split_mixes_bwd, (4, 2.0, 2, 2, T.float32), [*grads, x, post.contiguous(), scale, base], [7, 8, 9])
    for actual, ref in zip((dx, ds.sum(0), db.sum(0)), expected):
        torch.testing.assert_close(actual, ref, atol=1e-6, rtol=1e-4)


def test_pre_apply_mix_backward_accumulates():
    x, mix = _tensor(3, 4, 32, bf16=True, grad=True), _tensor(3, 4, grad=True)
    dy = _tensor(3, 32, bf16=True)
    expected = torch.autograd.grad((x.float() * mix.unsqueeze(-1)).sum(1).bfloat16(), (x, mix), dy)
    initial = _tensor(3, 4, 32, bf16=True)
    dx = initial.clone()
    dm = _run(_mhc_pre_apply_mix_bwd, (4, 32, 128, 16), [dy, x, mix, dx], [4])
    torch.testing.assert_close(dx, (initial.float() + mix.detach().unsqueeze(-1) * dy.float().unsqueeze(1)).bfloat16())
    torch.testing.assert_close(dm, expected[1], atol=1e-6, rtol=1e-4)


def test_post_backward():
    a, b = _tensor(2, 4, 4, grad=True), _tensor(2, 4, 32, bf16=True, grad=True)
    c, d = _tensor(2, 4, grad=True), _tensor(2, 32, bf16=True, grad=True)
    dy = _tensor(2, 4, 32, bf16=True)
    y = (a.transpose(1, 2) @ b.float() + c.unsqueeze(-1) * d.float().unsqueeze(1)).bfloat16()
    expected = torch.autograd.grad(y, (a, b, c, d), dy)
    actual = _run(_mhc_post_bwd, (4, 32, 128, 16), [dy, a, b, c, d], [5, 6, 7, 8])
    for result, ref in zip(actual, expected):
        torch.testing.assert_close(result, ref, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("repeat", [1, 3])
def test_sinkhorn_backward(repeat):
    x = _tensor(3, 4, 4, grad=True)
    dy = _tensor(3, 4, 4)
    y = torch.softmax(x, -1) + 1e-6
    y = y / (y.sum(-2, keepdim=True) + 1e-6)
    for _ in range(repeat - 1):
        y = y / (y.sum(-1, keepdim=True) + 1e-6)
        y = y / (y.sum(-2, keepdim=True) + 1e-6)
    (expected,) = torch.autograd.grad(y, x, dy)
    actual = _run(_mhc_sinkhorn_bwd, (4, 1, repeat, 1e-6), [dy, x], [2])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)


def test_normw_merge_backward_accumulates():
    fn, normw = _tensor(4, 256, grad=True), _tensor(256, grad=True)
    dy = _tensor(4, 256)
    expected = torch.autograd.grad(fn * normw, (fn, normw), dy)
    df, dw = _tensor(4, 256), _tensor(256)
    initial_df, initial_dw = df.clone(), dw.clone()
    _run(_mhc_fn_normw_merge_bwd, (4, 256, T.float32), [fn, normw, dy, df, dw], [])
    torch.testing.assert_close(df, initial_df + expected[0])
    torch.testing.assert_close(dw, initial_dw + expected[1])


def test_pre_norm_fn_forward_backward_pipeline():
    tokens, groups, hidden, mixes = 2, 2, 256, 24
    x = _tensor(tokens, groups * hidden, bf16=True, grad=True)
    fn = _tensor(mixes, groups * hidden, grad=True)
    dy = _tensor(tokens, mixes)
    xg, wg = x.float().reshape(tokens, groups, hidden), fn.reshape(mixes, groups, hidden)
    mul = torch.einsum("tgh,mgh->tgm", xg, wg)
    sqrsum = xg.square().sum(-1)
    y = (mul * torch.rsqrt(sqrsum / hidden + 1e-6).unsqueeze(-1)).sum(1)
    expected_dx, expected_df = torch.autograd.grad(y, (x, fn), dy)
    actual_mul, actual_sqrsum = _run(_mhc_pre_norm_fn_fwd_mul, (mixes, groups, hidden, 8, 256), [x, fn], [2, 3])
    torch.testing.assert_close(actual_mul, mul, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual_sqrsum, sqrsum, atol=1e-5, rtol=1e-4)
    merged_mul, merged_sqrsum, actual_y = _run(
        _mhc_pre_norm_fn_fwd_norm, (mixes, groups, hidden, 1e-6, 1), [actual_mul.unsqueeze(0), actual_sqrsum.unsqueeze(0)], [2, 3, 4]
    )
    torch.testing.assert_close(actual_y, y, atol=1e-5, rtol=1e-4)
    dmul, dsqrsum = _run(_mhc_pre_norm_fn_bwd_norm, (mixes, groups, hidden, 1e-6), [dy, merged_mul, merged_sqrsum], [3, 4])
    dx = torch.zeros_like(x)
    df = _run(_mhc_pre_norm_fn_bwd_mul, (mixes, groups, hidden, 8, 128), [dmul, dsqrsum, x, fn, dx], [5])
    torch.testing.assert_close(dx, expected_dx, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(df, expected_df, atol=1e-5, rtol=1e-4)
