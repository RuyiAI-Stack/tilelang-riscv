"""Original speculative sampler, with independent prefix and residual references."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class
from ._sampling_reference import _philox_uniform


@pytest.mark.parametrize("threads,pace", [(32, 1), (64, 2)])
def test_chain_speculative_sampling_ragged_runtime_compare(threads, pace):
    get_kernel_class("sampling.chain_speculative_sampling", "ChainSpeculativeSamplingFwdKernel")
    from tileops.kernels.sampling.chain_speculative_sampling import _chain_speculative_sampling_kernel

    batch, drafts, vocab = 4, 3, 33
    seed_value, offset_value = 0x123456789ABCDEF, 0x102030405060708
    torch.manual_seed(134)

    # Dyadic probabilities make sums exact across both launch schedules.
    def probabilities(rows):
        return torch.stack([torch.bincount(torch.randint(vocab, (64,)), minlength=vocab).float() / 64 for _ in range(rows)])

    draft = probabilities(batch * drafts).reshape(batch, drafts, vocab)
    target = probabilities(batch * (drafts + 1)).reshape(batch, drafts + 1, vocab)
    tokens = draft.argmax(dim=-1).to(torch.int32)
    # Row 0 accepts all drafts; rows 1/2 reject at positions 0/2. Row 3
    # exercises the stochastic acceptance test against unrelated distributions.
    target[:3, :drafts] = draft[:3]
    for row, position in ((1, 0), (2, 2)):
        token = int(tokens[row, position])
        target[row, position, (token + 1) % vocab] += target[row, position, token]
        target[row, position, token] = 0
    seed = torch.tensor([seed_value], dtype=torch.int64)
    offset = torch.tensor([offset_value], dtype=torch.int64)
    inputs = (draft, tokens, target, seed, offset)
    originals = [value.clone() for value in inputs]
    partial = torch.empty(batch, dtype=torch.float64)
    output = torch.full((batch, drafts + 1), -99, dtype=torch.int32)
    accepted = torch.full((batch,), -99, dtype=torch.int32)
    factory = _chain_speculative_sampling_kernel(batch, drafts, vocab, 1, threads, 1, pace)
    kernel = compile_tileops_jit(factory, {})
    try:
        kernel(*(value.reshape(-1) for value in inputs), partial, output.reshape(-1), accepted)
    finally:
        kernel.close()

    expected = torch.full_like(output, -1)
    expected_accepted = torch.empty_like(accepted)
    for row in range(batch):
        prefix = 0
        for position in range(drafts):
            token = int(tokens[row, position])
            uniform = torch.tensor(_philox_uniform(seed_value, offset_value, row, position))
            if uniform * draft[row, position, token] >= target[row, position, token]:
                break
            prefix += 1
        residual = target[row, prefix].clone()
        if prefix < drafts:
            residual = (residual - draft[row, prefix]).clamp_min(0)
        point = _philox_uniform(seed_value, offset_value, row, drafts) * float(residual.double().sum())
        cumulative = 0.0
        for token, weight in enumerate(residual.tolist()):  # noqa: B007 - use the final token after the search
            cumulative += weight
            if cumulative > point:
                break
        expected[row, :prefix] = tokens[row, :prefix]
        expected[row, prefix] = token
        expected_accepted[row] = prefix
    assert expected_accepted[:3].tolist() == [3, 0, 2]
    torch.testing.assert_close(accepted, expected_accepted, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
