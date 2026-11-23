import shutil
import subprocess

import pytest
import tilelang
import tilelang.language as T
import torch
from tilelang import tvm


@T.prim_func
def update_tile(A: T.Tensor((3, 4), "int32"), indices: T.Tensor((2,), "int32"), B: T.Tensor((3, 4), "int32")):
    with T.Kernel(1, threads=1):
        scratch = T.alloc_shared((3, 4), "int32")
        T.copy(A, scratch)
        T.fill(scratch[0:3, 1:2], -7)
        scratch[indices[0], indices[1]] = A[2, 3] + 50
        T.copy(scratch, B)


@pytest.mark.parametrize("structured", [False, True])
def test_private_writes_preserve_unwritten_elements(structured):
    config = {"tl.riscv_structured_private_writes": structured}
    with tvm.transform.PassContext(opt_level=3, config=config):
        source = tilelang.lower(update_tile, target="riscv").kernel_source
    if structured:
        assert "linalg.fill" in source
        assert "memref.store" not in source
        assert "[1, 1] [1, 1]" in source  # point subview size and strides
    else:
        assert "memref.store" in source

    kernel = tilelang.compile(update_tile, out_idx=[2], target="riscv", pass_configs=config)
    try:
        data = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        for row, col in [(0, 0), (1, 1), (2, 3)]:
            expected = data.clone()
            expected[:, 1] = -7
            expected[row, col] = data[2, 3] + 50
            actual = kernel(data, torch.tensor([row, col], dtype=torch.int32))
            torch.testing.assert_close(actual, expected)
    finally:
        kernel.close()


def test_external_scalar_store_stays_external():
    @T.prim_func
    def write(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            B[2] = 17

    with tvm.transform.PassContext(opt_level=3, config={"tl.riscv_structured_private_writes": True}):
        source = tilelang.lower(write, target="riscv").kernel_source
    assert "memref.store" in source
    assert "linalg.fill" not in source


@pytest.mark.parametrize("initialized", [True, False])
def test_buddy_point_write_requires_initialized_siblings(initialized):
    buddy = shutil.which("buddy-opt")
    if buddy is None or "--kernel-contract-verify" not in subprocess.check_output([buddy, "--help"], text=True):
        pytest.skip("requires Buddy neutral kernel contract verification")

    @T.prim_func
    def update(out: T.Tensor((4,), "int32"), value: T.int32):
        with T.Kernel(1, threads=1):
            scratch = T.alloc_local((4,), "int32")
            if initialized:
                T.fill(scratch, T.int32(7))
            scratch[2] = value
            T.copy(scratch, out)

    with tvm.transform.PassContext(opt_level=3, config={"tl.riscv_structured_private_writes": True}):
        source = tilelang.lower(update, target="riscv").kernel_source
    result = subprocess.run(
        [buddy, "--canonicalize", "--cse", "--kernel-contract-annotate", "--kernel-contract-verify", "-o", "/dev/null"],
        input=source, capture_output=True, text=True,
    )
    if initialized:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "V22" in result.stderr and "initializ" in result.stderr
