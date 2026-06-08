from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang.language as T

try:
    from .riscv_example_utils import run_example
except ImportError:
    from riscv_example_utils import run_example


M = 2
N = 4
K = 3


@T.prim_func
def matmul(
    A: T.Tensor((M, K), "float32"),
    B: T.Tensor((K, N), "float32"),
    C: T.Tensor((M, N), "float32"),
):
    for i, j, kk in T.grid(M, N, K):
        with T.block("matmul"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            vk = T.axis.reduce(K, kk)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def make_torch_case():
    lhs = torch.arange(M * K, dtype=torch.float32).reshape(M, K)
    rhs = torch.linspace(-1.0, 2.0, steps=K * N, dtype=torch.float32).reshape(K, N)
    return (lhs, rhs), lhs @ rhs


def make_numpy_case():
    lhs = np.arange(M * K, dtype=np.float32).reshape(M, K)
    rhs = np.linspace(-1.0, 2.0, num=K * N, dtype=np.float32).reshape(K, N)
    out = np.zeros((M, N), dtype=np.float32)
    return (lhs, rhs, out), lhs @ rhs, 2


if __name__ == "__main__":
    run_example(
        name="matmul",
        description="RISC-V small matmul validation example.",
        func=matmul,
        out_idx=[2],
        make_torch_case=make_torch_case,
        make_numpy_case=make_numpy_case,
    )
