import json
import os
import platform
from pathlib import Path
from tempfile import mkdtemp

import torch
import tilelang
from tilelang.transform import PassConfigKey

from tile_kernels.engram.engram_hash_kernel import get_engram_hash_kernel
from tile_kernels.torch.engram import engram_hash_ref, make_offsets


# Original TileKernels source: tile_kernels/engram/engram_hash_kernel.py
def test_engram_hash_kernel_runtime_compare():
    seed = 2
    torch.manual_seed(seed)
    ngram_token_ids = torch.randint(0, 1000, (7, 3), dtype=torch.int32)
    multipliers = torch.randint(1, 1000, (2, 3), dtype=torch.int64)
    vocab_sizes = torch.randint(100, 200, (2, 2, 8), dtype=torch.int32)
    offsets = make_offsets(vocab_sizes)
    expected = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets)

    prim_func = get_engram_hash_kernel.get_tir(3, 2, 8)
    artifact_dir = None
    pass_configs = None
    if output_dir := os.environ.get("TILELANG_ENGRAM_HASH_ARTIFACTS"):
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        artifact_dir = Path(mkdtemp(prefix="run-", dir=output_root))
        print(f"Engram Hash artifacts: {artifact_dir}")
        (artifact_dir / "input.tir.py").write_text(prim_func.script() + "\n", encoding="utf-8")
        pass_configs = {
            PassConfigKey.TL_ENABLE_DUMP_IR: True,
            PassConfigKey.TL_DUMP_IR_DIR: str(artifact_dir / "tir-passes"),
        }

    kernel = tilelang.compile(prim_func, out_idx=[4], target="riscv", pass_configs=pass_configs)
    try:
        if artifact_dir is not None:
            mlir = kernel.get_kernel_source()
            (artifact_dir / "kernel.mlir").write_text(mlir, encoding="utf-8")
            assert "func.func" in mlir and "Placeholder MLIR module" not in mlir, (
                "Expected real RISC-V MLIR. Check that the native MLIR backend is enabled."
            )
        actual = kernel(ngram_token_ids, multipliers, vocab_sizes, offsets)
    finally:
        kernel.close()

    passed = False
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        passed = True
    finally:
        if artifact_dir is not None:
            result = {
                "case": "engram_hash_kernel",
                "seed": seed,
                "target": "riscv",
                "execution_arch": platform.machine(),
                "reference": "tile_kernels.torch.engram.engram_hash_ref",
                "passed": passed,
                "actual": {"dtype": str(actual.dtype), "shape": list(actual.shape), "values": actual.tolist()},
                "expected": {"dtype": str(expected.dtype), "shape": list(expected.shape), "values": expected.tolist()},
            }
            (artifact_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
