"""Exercise upstream FFT plans and their original interleaved programs."""

import pytest
import torch
from ._harness import get_kernel_class


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_fft_c2c_runtime_compare(n, dtype):
    kernel_cls = get_kernel_class("fft", "FFTC2COneCTAKernel")
    kernel = kernel_cls(n, dtype)
    # Reuse one dynamic-batch kernel across partial and multiple CTAs. Random
    # real/imaginary parts exercise both butterfly outputs independently.
    generator = torch.Generator().manual_seed(96)
    for batch in (2, 259):
        x = torch.randn(batch, n, dtype=dtype, generator=generator)
        before = x.clone()
        torch.testing.assert_close(kernel(x), torch.fft.fft(x, dim=-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
