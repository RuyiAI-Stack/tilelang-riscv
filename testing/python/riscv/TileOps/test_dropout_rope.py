from __future__ import annotations

import pytest
import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_dropout_float32_runtime_compare():
    x = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
    dropout_cls = get_kernel_class("dropout", "DropoutKernel")
    kwargs = {
        "N_total": x.numel(),
        "dtype": x.dtype,
        "p": 0.5,
        "seed": 7,
        "config": {"threads": 128, "num_per_thread": 2},
    }
    kernel_a = compile_tileops_kernel(dropout_cls(**kwargs))
    kernel_b = compile_tileops_kernel(dropout_cls(**kwargs))
    actual_a = kernel_a(x.contiguous())
    actual_b = kernel_b(x.contiguous())

    torch.testing.assert_close(actual_a, actual_b, rtol=0.0, atol=0.0)
    expected_values = torch.stack((torch.zeros_like(x), x * 2.0))
    distances = (actual_a[:, None] - expected_values.T).abs().amin(dim=1)
    torch.testing.assert_close(distances, torch.zeros_like(distances), rtol=0.0, atol=1e-6)
    assert bool((actual_a == 0).any())
    assert bool((actual_a != 0).any())


def _rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    neox: bool,
) -> torch.Tensor:
    output = x.clone()
    if neox:
        half = x.shape[-1] // 2
        output[:, :half] = x[:, :half] * cos - x[:, half:] * sin
        output[:, half:] = x[:, half:] * cos + x[:, :half] * sin
    else:
        output[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
        output[:, 1::2] = x[:, 1::2] * cos + x[:, 0::2] * sin
    return output


@pytest.mark.parametrize(
    ("class_name", "neox"),
    [
        ("RoPENeoxKernel", True),
        ("RoPENonNeoxKernel", False),
    ],
)
def test_rope_float32_runtime_compare(class_name, neox):
    seq_len, head_dim = 64, 8
    x = torch.linspace(-1.0, 1.0, seq_len * head_dim, dtype=torch.float32).reshape(
        seq_len,
        head_dim,
    )
    angles = torch.linspace(0.1, 1.0, seq_len * (head_dim // 2), dtype=torch.float32)
    angles = angles.reshape(seq_len, head_dim // 2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rope_cls = get_kernel_class("rope", class_name)
    tileops_kernel = rope_cls(
        seq_len,
        head_dim,
        x.dtype,
        config={"threads": 128, "num_per_thread": 4},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    expected = _rope_reference(x, cos, sin, neox=neox)
    torch.testing.assert_close(
        kernel(x.contiguous(), cos.contiguous(), sin.contiguous()),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_rope_neox_position_ids_float32_runtime_compare():
    num_tokens, num_heads, head_dim = 4, 2, 8
    rotary_dim, max_position = 4, 8
    x = torch.linspace(
        -1.0,
        1.0,
        num_tokens * num_heads * head_dim,
        dtype=torch.float32,
    ).reshape(num_tokens, num_heads, head_dim)
    angles = torch.linspace(
        0.1,
        1.0,
        max_position * (rotary_dim // 2),
        dtype=torch.float32,
    ).reshape(max_position, rotary_dim // 2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    position_ids = torch.tensor([0, 3, 1, 6], dtype=torch.int32)
    status = torch.zeros(1, dtype=torch.int32)

    rope_cls = get_kernel_class("rope", "RoPENeoxPositionIdsKernel")
    tileops_kernel = rope_cls(
        num_tokens,
        num_heads,
        head_dim,
        rotary_dim,
        max_position,
        x.dtype,
        config={"threads": 128, "num_per_thread": 4},
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(
        x.contiguous().reshape(-1),
        cos.contiguous(),
        sin.contiguous(),
        position_ids.contiguous(),
        status,
    ).reshape(x.shape)
    assert status.item() == 0

    expected = x.clone()
    half = rotary_dim // 2
    for token in range(num_tokens):
        c = cos[position_ids[token]]
        s = sin[position_ids[token]]
        expected[token, :, :half] = x[token, :, :half] * c - x[token, :, half:rotary_dim] * s
        expected[token, :, half:rotary_dim] = x[token, :, half:rotary_dim] * c + x[token, :, :half] * s
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
