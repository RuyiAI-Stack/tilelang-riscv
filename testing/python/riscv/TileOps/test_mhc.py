from __future__ import annotations

import math

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_mhc_pre_float32_serial_runtime_compare():
    batch, n_expand, c_x = 1, 2, 4
    phi_dim = n_expand * n_expand + 2 * n_expand
    phi = torch.linspace(-0.2, 0.2, n_expand * c_x * phi_dim).reshape(
        n_expand * c_x,
        phi_dim,
    )
    x = torch.linspace(-0.4, 0.4, batch * n_expand * c_x).reshape(batch, -1)
    bias = torch.linspace(-0.1, 0.1, phi_dim)
    alpha_pre, alpha_post, alpha_res = 0.7, 0.6, 0.8
    sinkhorn_repeat, sinkhorn_eps = 3, 0.02

    kernel_cls = get_kernel_class("mhc.mhc_pre", "MHCPreKernel")
    tileops_kernel = kernel_cls(
        batch,
        n_expand,
        c_x,
        torch.float32,
        config={"block_x_b": 1, "block_C": 2, "num_stages": 1, "threads": 1},
    )
    actual_res, actual_layer, actual_post = tileops_kernel(
        phi,
        x,
        bias,
        alpha_pre,
        alpha_post,
        alpha_res,
        sinkhorn_repeat,
        sinkhorn_eps,
    )

    norm = torch.sqrt((x * x).sum(dim=1)) / math.sqrt(n_expand * c_x) + 0.0001
    h = x.float() @ phi
    h_pre = torch.sigmoid(alpha_pre * h[:, :n_expand] / norm[:, None] + bias[:n_expand])
    expected_post = 2 * torch.sigmoid(alpha_post * h[:, n_expand : 2 * n_expand] / norm[:, None] + bias[n_expand : 2 * n_expand])
    torch.testing.assert_close(actual_post, expected_post, rtol=1e-4, atol=1e-4)
    h_res = h[:, 2 * n_expand :].reshape(batch, n_expand, n_expand)
    h_res = alpha_res * h_res / norm[:, None, None] + bias[2 * n_expand :].reshape(n_expand, n_expand)
    h_res = torch.exp(h_res - h_res.max(dim=-1, keepdim=True).values)
    for _ in range(sinkhorn_repeat):
        h_res = h_res / (h_res.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        h_res = h_res / (h_res.sum(dim=-2, keepdim=True) + sinkhorn_eps)

    x_reshaped = x.reshape(batch, n_expand, c_x)
    expected_res = (h_res @ x_reshaped).reshape_as(x)
    expected_layer = (h_pre[:, None, :] @ x_reshaped).squeeze(1)

    torch.testing.assert_close(actual_res, expected_res, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_layer, expected_layer, rtol=1e-4, atol=1e-4)


def test_mhc_post_float32_runtime_compare():
    batch, n_expand, c_x = 2, 4, 8
    x_layer_out = torch.linspace(-1.0, 1.0, batch * c_x, dtype=torch.float32).reshape(batch, c_x)
    h_post = torch.linspace(0.5, 1.0, batch * n_expand, dtype=torch.float32).reshape(batch, n_expand)
    x_res = torch.linspace(-2.0, 2.0, batch * n_expand * c_x, dtype=torch.float32).reshape(
        batch,
        n_expand * c_x,
    )

    mhc_post_cls = get_kernel_class("mhc.mhc_post", "MHCPostKernel")
    tileops_kernel = mhc_post_cls(
        batch,
        n_expand,
        c_x,
        torch.float32,
        config={"block_x_b": 1, "block_C": 4, "num_stages": 2, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x_layer_out.contiguous(), h_post.contiguous(), x_res.contiguous())

    expected = torch.empty_like(x_res)
    for b in range(batch):
        for i in range(n_expand):
            expected[b, i * c_x : (i + 1) * c_x] = h_post[b, i] * x_layer_out[b] + x_res[b, i * c_x : (i + 1) * c_x]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
