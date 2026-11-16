# GEMM

先激活已启用 MLIR 的 TileLang 环境，准备匹配的 Buddy/LLVM。
在仓库根目录执行，将 `/path/to/...` 替换为实际路径：

```bash
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm-build
export TILELANG_RISCV_BUDDY_ROOT=/path/to/buddy-build
export PATH="$TILELANG_RISCV_BUDDY_ROOT/bin:$TILELANG_RISCV_LLVM_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TILELANG_RISCV_LLVM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python examples/riscv/gemm/gemm.py
```

程序在宿主上编译并执行 GEMM，打印计算结果。
