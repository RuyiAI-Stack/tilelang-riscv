"""Minimal TileLang INT8 GEMM: C += A @ B.T, one logical workgroup."""

import tilelang.language as T

M = 16
N = 16
K = 64


@T.prim_func
def bosc_gemm(A: T.Tensor((M, K), T.int8), B: T.Tensor((N, K), T.int8), C: T.Tensor((M, N), T.int32)):
    with T.Kernel(1, threads=1):
        T.gemm(A, B, C, transpose_B=True)
