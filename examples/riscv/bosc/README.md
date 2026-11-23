# BOSC GEMM

本例构建 INT8 `C += A @ B.T` 固件（A/B：16×64，C：16×16），包含四组板端数值自检。

先激活已启用 MLIR 的 TileLang 环境，准备匹配的 Buddy/LLVM（NR runtime 需支持 UART v1）。
在仓库根目录执行，将 `/path/to/...` 替换为实际路径：

```bash
export BOSC_BUDDY_SOURCE=/path/to/buddy-mlir
export TILELANG_RISCV_LLVM_ROOT=/path/to/llvm-build
export TILELANG_RISCV_BUDDY_ROOT=/path/to/buddy-build
source examples/riscv/bosc/env.sh
python examples/riscv/bosc/build.py --output /path/to/output
```

`--output` 须为新目录或空目录，省略时输出到 `examples/riscv/bosc/build/<时间戳>/`。
将生成的 `bosc_gemm.bin` 复制到已配置的 FPGA 客户端后执行：

```bash
fpga-run /path/to/bosc_gemm.bin --timeout 600
```

成功时数值检查为 `errors=0`，任务状态为 `completed`，应用退出码为 `0`。

修改算子时编辑 [kernel.py](kernel.py) 和 [launch.c](launch.c)，同步调整 shape、ABI 和参考计算；
[build.py](build.py) 完成编译、链接及 ISA 检查。构建仅输出 ELF/bin。

向 Buddy 导出其他算子的 MLIR 时，可以启用同一项配置：

```python
with tvm.transform.PassContext(opt_level=3, config={"tl.riscv_structured_private_writes": True}):
    mlir = tilelang.lower(kernel, target="riscv").kernel_source
```

编译器将可证明在界内的 tile 填充写到对应区域。对于私有缓冲区的普通标量写入，
它保留原来的下标，把目标表示为只覆盖一个元素的子视图。Buddy 可以据此检查
部分写入与先前内容的依赖。可用以下命令检查生成的 IR：

```bash
buddy-opt kernel.mlir --canonicalize --cse --kernel-contract-annotate --kernel-contract-verify -o contract.mlir
```

该步骤检查中性契约。目标规划、AME 指令生成和运行时数值验证由后续阶段完成。
