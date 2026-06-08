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


ROWS = 4
COLS = 8


@T.prim_func
def reduce_sum_rows(
    A: T.Tensor((ROWS, COLS), "float32"),
    B: T.Tensor((ROWS,), "float32"),
):
    for i, k in T.grid(ROWS, COLS):
        with T.block("sum"):
            vi = T.axis.spatial(ROWS, i)
            vk = T.axis.reduce(COLS, k)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + A[vi, vk]


def make_torch_case():
    data = torch.linspace(-3.0, 5.0, steps=ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
    return (data,), data.sum(dim=1)


def make_numpy_case():
    data = np.linspace(-3.0, 5.0, num=ROWS * COLS, dtype=np.float32).reshape(ROWS, COLS)
    out = np.zeros((ROWS,), dtype=np.float32)
    return (data, out), data.sum(axis=1), 1


if __name__ == "__main__":
    run_example(
        name="reduce_sum",
        description="RISC-V row-wise reduce-sum validation example.",
        func=reduce_sum_rows,
        out_idx=[1],
        make_torch_case=make_torch_case,
        make_numpy_case=make_numpy_case,
    )
