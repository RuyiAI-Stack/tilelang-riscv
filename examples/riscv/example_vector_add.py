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


N = 8


@T.prim_func
def vector_add(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    C: T.Tensor((N,), "float32"),
):
    for i in T.serial(N):
        with T.block("add"):
            vi = T.axis.spatial(N, i)
            C[vi] = A[vi] + B[vi]


def make_torch_case():
    lhs = torch.linspace(-2.0, 2.0, steps=N, dtype=torch.float32)
    rhs = torch.linspace(0.25, 1.75, steps=N, dtype=torch.float32)
    return (lhs, rhs), lhs + rhs


def make_numpy_case():
    lhs = np.linspace(-2.0, 2.0, num=N, dtype=np.float32)
    rhs = np.linspace(0.25, 1.75, num=N, dtype=np.float32)
    out = np.zeros_like(lhs)
    return (lhs, rhs, out), lhs + rhs, 2


if __name__ == "__main__":
    run_example(
        name="vector_add",
        description="RISC-V vector add validation example.",
        func=vector_add,
        out_idx=[2],
        make_torch_case=make_torch_case,
        make_numpy_case=make_numpy_case,
    )
