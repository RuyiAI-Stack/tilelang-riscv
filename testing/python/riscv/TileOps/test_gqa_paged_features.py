"""Original paged GQA programs: RoPE, masks and sinks, including split reduction."""

import pytest
import torch

from ._harness import compile_tileops_jit


def _rotate(x, positions, cos, sin, rotary_dim, layout):
    # Compute rotations independently in FP64, then store at the input precision.
    values = x.double()
    result = values.clone()
    half = rotary_dim // 2
    a = torch.arange(half) if layout == "neox" else torch.arange(0, rotary_dim, 2)
    b = a + half if layout == "neox" else a + 1
    c, s = cos[positions].double()[:, None], sin[positions].double()[:, None]
    result[..., a] = values[..., a] * c - values[..., b] * s
    result[..., b] = values[..., b] * c + values[..., a] * s
    return result.to(x.dtype)


def _reference(q, k_pages, v_pages, lengths, pages, offsets, cos, sin, sinks, case):
    output = torch.empty_like(q)
    group = q.shape[1] // k_pages.shape[2]
    for request, length in enumerate(lengths.tolist()):
        start, end = offsets[request : request + 2].tolist()
        if start == end:
            continue
        query = q[start:end]
        keys = k_pages[pages[request].long()].flatten(0, 1)[:length]
        values = v_pages[pages[request].long()].flatten(0, 1)[:length].double()
        query_positions = torch.arange(length - (end - start), length)
        key_positions = torch.arange(length)
        if case["layout"] != "none":
            query = _rotate(query, query_positions, cos, sin, case["rotary_dim"], case["layout"])
            keys = _rotate(keys, key_positions, cos, sin, case["rotary_dim"], case["layout"])
        visible = torch.ones((end - start, length), dtype=torch.bool)
        if case["causal"]:
            visible &= key_positions[None] <= query_positions[:, None]
        if case["left"] >= 0:
            visible &= key_positions[None] >= query_positions[:, None] - case["left"]
        if case["right"] >= 0:
            visible &= key_positions[None] <= query_positions[:, None] + case["right"]
        for head in range(q.shape[1]):
            kv_head = head // group
            scores = query[:, head].double() @ keys[:, kv_head].double().T * case["scale"]
            if case["softcap"]:
                scores = case["softcap"] * torch.tanh(scores / case["softcap"])
            scores.masked_fill_(~visible, -torch.inf)
            if torch.isposinf(sinks[head]):
                output[start:end, head] = 0
            else:
                # A raw sink logit contributes only to the denominator. It is
                # neither multiplied by the score scale nor softcapped.
                logits = torch.cat((scores, sinks[head].double().expand(end - start, 1)), dim=1)
                weights = logits.softmax(dim=1)[:, :length]
                weights[~visible.any(dim=1)] = 0
                output[start:end, head] = (weights @ values[:, kv_head]).to(q.dtype)
    return output


CASES = [
    pytest.param(
        dict(layout="neox", rotary_dim=32, causal=True, left=17, right=-1, scale=-0.25, softcap=0.0, page_size=7),
        id="neox-causal-negative-scale",
    ),
    pytest.param(
        dict(layout="interleaved", rotary_dim=16, causal=False, left=17, right=3, scale=0.25, softcap=0.5, page_size=7),
        id="partial-interleaved-two-sided-softcap",
    ),
    pytest.param(
        dict(layout="none", rotary_dim=0, causal=False, left=17, right=1, scale=0.0, softcap=0.0, page_size=16),
        id="plain-two-sided-zero-scale",
    ),
    pytest.param(
        dict(layout="none", rotary_dim=0, causal=False, left=0, right=0, scale=0.25, softcap=0.0, page_size=16),
        id="plain-single-key-window",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 3], ids=["unsplit", "split"])
@pytest.mark.parametrize("case", CASES)
def test_paged_features_runtime_compare(dtype, splits, case):
    from tileops.kernels.attention.gqa.paged import (
        _gqa_paged_varlen_kernel,
        _gqa_paged_varlen_split_kernel,
    )

    rng = torch.Generator().manual_seed(94)
    batch, heads, heads_kv, dim = 3, 8, 2, 32
    block_m, block_n, max_position = 16, 16, 53
    page_size = case["page_size"]
    max_pages = (max_position + page_size - 1) // page_size
    pool_pages = batch * max_pages + 2
    factory_args = dict(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        page_size=page_size,
        max_pages_per_req=max_pages,
        is_causal=case["causal"],
        window_size_left=case["left"],
        window_size_right=case["right"],
        sm_scale=case["scale"],
        softcap=case["softcap"],
        dtype=str(dtype).split(".")[-1],
        fuse_rope=case["layout"] != "none",
        max_position=max_position,
        rotary_dim=case["rotary_dim"],
        rope_layout=case["layout"],
        has_sinks=True,
        sm90=False,
    )
    config = dict(block_M=block_m, block_N=block_n, num_stages=1, threads=128)
    if splits == 1:
        factory = _gqa_paged_varlen_kernel(**factory_args, producer_warpgroup=False)
    else:
        factory = _gqa_paged_varlen_split_kernel(**factory_args)
        config["num_split"] = splits
    kernel = compile_tileops_jit(factory, config)
    half = case["rotary_dim"] // 2
    angles = torch.arange(max_position)[:, None] * torch.linspace(0.1, 0.7, half)[None]
    cos = angles.cos().to(dtype) if half else torch.empty((1, 1), dtype=dtype)
    sin = angles.sin().to(dtype) if half else torch.empty((1, 1), dtype=dtype)
    lengths = torch.empty(batch, dtype=torch.int32)
    offsets = torch.empty(batch + 1, dtype=torch.int32)
    pages = torch.empty((batch, max_pages), dtype=torch.int32)
    sinks = torch.tensor([2.5, -torch.inf, torch.inf, -1.5, -0.3, 3.0, -torch.inf, 0.8])
    try:
        # Same compiled program and metadata buffers, new packing and logical
        # rotary positions. Empty requests change index between calls.
        for q_lens, kv_lens in (([19, 0, 5], [53, 0, 35]), ([5, 17, 2], [29, 41, 27]), ([0, 1, 0], [0, 1, 0])):
            lengths.copy_(torch.tensor(kv_lens, dtype=torch.int32))
            offsets.copy_(torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32))
            pages.copy_(torch.randperm(pool_pages, generator=rng)[: batch * max_pages].reshape_as(pages))
            q = (torch.randn(sum(q_lens), heads, dim, generator=rng) * 0.8).to(dtype)
            k = torch.full((pool_pages, page_size, heads_kv, dim), torch.nan, dtype=dtype)
            v = torch.full_like(k, torch.inf)
            for request, length in enumerate(kv_lens):
                for pos in range(length):
                    slot = (pages[request, pos // page_size].item(), pos % page_size)
                    k[slot] = torch.randn(heads_kv, dim, generator=rng) * 0.8
                    v[slot] = torch.randn(heads_kv, dim, generator=rng)
            inputs = [q, k.flatten(0, 1), v.flatten(0, 1), lengths, pages, offsets, cos, sin, sinks]
            before = [tensor.view(torch.uint8).clone() for tensor in inputs]
            scratch = []
            if splits > 1:
                tiles = sum(q_lens) * (heads // heads_kv) // block_m + batch
                scratch = [
                    torch.full((tiles, heads_kv, splits, block_m), torch.nan),
                    torch.full((tiles, heads_kv, splits, block_m, dim), torch.nan, dtype=dtype),
                ]
            actual = kernel(*inputs, *scratch)
            expected = _reference(q, k, v, lengths, pages, offsets, cos, sin, sinks, case)
            torch.testing.assert_close(actual, expected, atol=8e-3 if dtype == torch.bfloat16 else 2e-3, rtol=2e-2)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual[:, 2], torch.zeros_like(actual[:, 2]), atol=0, rtol=0)
            for tensor, original in zip(inputs, before):
                assert torch.equal(tensor.view(torch.uint8), original)
    finally:
        kernel.close()
