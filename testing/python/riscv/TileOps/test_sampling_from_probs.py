"""Original upstream categorical sampler with a pure-Python Philox reference."""

import pytest
import torch

from ._harness import compile_tileops_jit, get_kernel_class
from ._sampling_reference import _philox_uniform


@pytest.mark.parametrize("vocab,threads", [(33, 32), (129, 64)])
def test_sampling_from_probs_ragged_runtime_compare(vocab, threads):
    get_kernel_class("sampling.sampling_from_probs", "SamplingFromProbsFwdKernel")
    from tileops.kernels.sampling.sampling_from_probs import _sampling_from_probs_kernel

    batch = 4
    seed_value, offset_value = 0x123456789ABCDEF, 0x102030405060708
    torch.manual_seed(132)
    probs = torch.rand(batch, vocab, dtype=torch.float32)
    probs[:, ::7] = 0
    probs /= probs.sum(dim=-1, keepdim=True)
    before = probs.clone()
    seed = torch.tensor([seed_value], dtype=torch.int64)
    offset = torch.tensor([offset_value], dtype=torch.int64)
    partial = torch.empty(batch, dtype=torch.float64)
    samples = torch.empty(batch, dtype=torch.int32)
    factory = _sampling_from_probs_kernel(batch, vocab, 1, threads, 1, 1)
    kernel = compile_tileops_jit(factory, {})
    try:
        kernel(probs.reshape(-1), seed, offset, partial, samples)
    finally:
        kernel.close()

    expected = []
    for row in range(batch):
        weights = probs[row].double().tolist()
        point = _philox_uniform(seed_value, offset_value, row) * sum(weights)
        total = 0.0
        for col, weight in enumerate(weights):
            total += weight
            if total > point:
                expected.append(col)
                break
    torch.testing.assert_close(samples, torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(probs, before, rtol=0, atol=0)
