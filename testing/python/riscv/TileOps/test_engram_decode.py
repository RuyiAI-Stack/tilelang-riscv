from __future__ import annotations

import torch
import torch.nn.functional as F

from ._harness import get_kernel_class


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x_f = x.float()
    rrms = (x_f * x_f).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x_f * rrms * weight.float(), rrms.squeeze(-1)


def _reference(
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
    max_conv_len: int,
    dilation: int,
    eps: float,
):
    batch, d = h_t.shape
    w = conv_w.shape[0]
    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()
    h_norm, _ = _rmsnorm(h_t.unsqueeze(1), rms_w_h, eps)
    k_norm, _ = _rmsnorm(k.unsqueeze(1).to(h_t.dtype), rms_w_h, eps)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)
    alpha = torch.sigmoid((h_norm * k_norm).sum(dim=-1, keepdim=True) / (d**0.5))
    v_hat = alpha * v
    v_hat_norm, _ = _rmsnorm(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v, eps)
    v_hat_norm = v_hat_norm.squeeze(1)
    conv_out = torch.zeros(batch, d, dtype=torch.float32)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * conv_state[:, state_idx, :].float()
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm
    if max_conv_len > conv_state.shape[1]:
        new_conv_state = torch.cat([conv_state, v_hat_norm.unsqueeze(1).to(conv_state.dtype)], dim=1)
    else:
        new_conv_state = torch.cat([conv_state[:, 1:, :], v_hat_norm.unsqueeze(1).to(conv_state.dtype)], dim=1)
    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state


def test_engram_decode_float32_runtime_compare():
    batch, d_mem, d = 1, 8, 256
    max_conv_len, conv_kernel_size, dilation = 3, 4, 1
    eps = 1e-6
    kernel_cls = get_kernel_class("engram.engram_decode", "EngramDecodeKernel")
    tileops_kernel = kernel_cls(
        batch,
        d_mem,
        d,
        max_conv_len,
        conv_kernel_size,
        dilation,
        eps,
        torch.float32,
        config={"threads": 128},
    )

    e_t = torch.linspace(-0.5, 0.5, batch * d_mem, dtype=torch.float32).reshape(batch, d_mem)
    h_t = torch.linspace(-0.4, 0.4, batch * d, dtype=torch.float32).reshape(batch, d)
    conv_state = torch.linspace(
        -0.3,
        0.3,
        batch * max_conv_len * d,
        dtype=torch.float32,
    ).reshape(batch, max_conv_len, d)
    W_K = torch.linspace(-0.25, 0.25, d_mem * d, dtype=torch.float32).reshape(d_mem, d)
    W_V = torch.linspace(-0.2, 0.2, d_mem * d, dtype=torch.float32).reshape(d_mem, d)
    rms_w_h = torch.linspace(0.8, 1.2, d, dtype=torch.float32)
    rms_w_v = torch.linspace(0.9, 1.1, d, dtype=torch.float32)
    conv_w = torch.linspace(-0.1, 0.1, conv_kernel_size * d, dtype=torch.float32).reshape(conv_kernel_size, d)

    actual = tileops_kernel(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
    expected = _reference(
        e_t,
        h_t,
        conv_state,
        W_K,
        W_V,
        rms_w_h,
        rms_w_v,
        conv_w,
        max_conv_len,
        dilation,
        eps,
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)
