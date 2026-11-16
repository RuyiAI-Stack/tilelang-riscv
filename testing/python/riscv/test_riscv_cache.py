from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.cache import _dispatch_map
from tilelang.env import env
from tilelang.jit.kernel import JITKernel


pytest.importorskip("tilelang.tladapter._native")


N = 8


@pytest.fixture()
def clean_riscv_cache_env(tmp_path):
    original_cache_dir = env.TILELANG_CACHE_DIR
    original_tmp_dir = env.TILELANG_TMP_DIR
    tilelang.enable_cache()

    cache_dir = tmp_path / "tilelang_cache"
    tmp_dir = tmp_path / "tilelang_tmp"
    cache_dir.mkdir()
    tmp_dir.mkdir()

    env.TILELANG_CACHE_DIR = str(cache_dir)
    env.TILELANG_TMP_DIR = str(tmp_dir)
    _dispatch_map["tvm_ffi"]._memory_cache.clear()

    yield cache_dir

    _dispatch_map["tvm_ffi"]._memory_cache.clear()
    env.TILELANG_CACHE_DIR = original_cache_dir
    env.TILELANG_TMP_DIR = original_tmp_dir


def test_riscv_disk_cache_reloads_without_recompile(clean_riscv_cache_env, monkeypatch):
    unique_id = uuid.uuid4().hex[:8]

    @T.prim_func
    def tile_copy(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((N,), "float32")
            T.copy(A, A_shared)
            T.copy(A_shared, B)

    kernel_func = tile_copy.with_attr("global_symbol", f"tile_copy_cache_{unique_id}")

    kernel1 = tilelang.compile(kernel_func, out_idx=[1], target="riscv")
    data = torch.arange(N, dtype=torch.float32)
    out1 = kernel1(data)
    torch.testing.assert_close(out1, data)
    kernel1.close()

    cache_files = list(Path(clean_riscv_cache_env).rglob("*.*"))
    assert cache_files, "Expected riscv cache files to be created"

    _dispatch_map["tvm_ffi"]._memory_cache.clear()

    def _unexpected_recompile(self, tilelang_func, out_idx):
        raise AssertionError("Disk cache miss: riscv tried to recompile instead of loading from cache")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", _unexpected_recompile)

    kernel2 = tilelang.compile(kernel_func, out_idx=[1], target="riscv")
    out2 = kernel2(data)
    kernel2.close()

    assert "func.func" in kernel2.get_kernel_source()
    torch.testing.assert_close(out2, data)


@pytest.mark.parametrize("target", ["riscv", "llvm"])
def test_cache_key_tracks_optimizer_only_for_riscv(monkeypatch, tmp_path, target):
    llvm_root = tmp_path / "llvm"
    buddy_root = tmp_path / "buddy-mlir"
    for root, name in ((llvm_root, "mlir-opt"), (buddy_root, "buddy-opt")):
        (root / "bin").mkdir(parents=True)
        executable = root / "bin" / name
        executable.write_text("#!/usr/bin/env bash\n")
        executable.chmod(0o755)
    monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(llvm_root))
    monkeypatch.setenv("TILELANG_RISCV_BUDDY_ROOT", str(buddy_root))
    monkeypatch.delenv("TILELANG_RISCV_OPT", raising=False)

    @T.prim_func
    def tile_copy(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    cache = _dispatch_map["tvm_ffi"]
    kwargs = dict(func=tile_copy, out_idx=[1], args=(), target=tilelang.tvm.target.Target(target))
    default_key = cache._generate_key(**kwargs)
    monkeypatch.setenv("TILELANG_RISCV_OPT", "mlir-opt")
    mlir_key = cache._generate_key(**kwargs)

    assert (default_key != mlir_key) == (target == "riscv")
    monkeypatch.delenv("TILELANG_RISCV_OPT")
    assert cache._generate_key(**kwargs) == default_key
